"""Clinical Summary Verification System Streamlit app.

The app verifies saved original-summary claims and the selected patient's
controlled mutation claims through the same retrieval, deterministic
verification, and optional Gemini semantic-adjudication pipeline. Source labels
and separate counts keep experimental mutations distinct from original-summary
claims. Supported claims are retained in the downloadable research report but
are omitted from the review-focused UI.
"""

import base64
import html
import json
import mimetypes
from pathlib import Path

import streamlit as st

from src.claim_extractor import extract_claims
from src.verification_pipeline import verify_claims_batch

from markdown_it import MarkdownIt

MARKDOWN_RENDERER = MarkdownIt(
    "commonmark",
    {"html": False},
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GROUND_TRUTH_PATH = (
    BASE_DIR
    / "data"
    / "patient_records"
    / "synthetic_discharge_ground_truth.json"
)
SUMMARY_DIR = BASE_DIR / "data" / "discharge_summaries"
CLAIMS_DIR = BASE_DIR / "data" / "claims"
ADDITIONAL_MUTATION_DIR = BASE_DIR / "results" / "additional_error_injection"
ADDITIONAL_MUTATION_CLAIMS_DIR = ADDITIONAL_MUTATION_DIR / "claims"
ADDITIONAL_MUTATION_REPORT_PATH = (
    ADDITIONAL_MUTATION_DIR / "additional_mutation_evaluation_report.json"
)

# Put your own hospital seal here when it is ready. Supported formats are
# determined from the file extension, e.g. PNG, JPG, JPEG, or WebP.
SEAL_IMAGE_PATH = BASE_DIR / "assets" / "quickcare_img.png"


# ============================================================
# PAGE CONFIG + STYLING
# ============================================================

st.set_page_config(
    page_title="Clinical Summary Verification System",
    page_icon="🏥",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
      }
      .app-subtitle {
        color: #5d6876;
        font-size: 1.02rem;
        line-height: 1.65;
        max-width: 950px;
        margin-top: -0.4rem;
      }
      .summary-document {
        background: #ffffff !important;
        color: #202733 !important;
        border: 1px solid #d9dee7;
        border-radius: 10px;
        box-shadow: 0 5px 20px rgba(22, 34, 51, 0.08);
        padding: 14px 22px 16px 22px;
        margin: 0.15rem 0 0.3rem 0;
      }
      .summary-doc-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
        border-bottom: 1px solid #e8ebf0;
        padding-bottom: 8px;
        margin-bottom: 8px;
      }
      .summary-doc-label {
        color: #657184;
        font-family: Arial, sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding-top: 7px;
      }
      .summary-seal {
        display: block;
        width: 76px;
        height: 76px;
        object-fit: contain;
        flex: 0 0 auto;
      }
      .summary-body {
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        color: #202733 !important;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 0.93rem;
        line-height: 1.08;
        margin: 0;
        padding: 0;
      }
      .small-muted {
        color: #6b7280;
        font-size: 0.88rem;
      }
      .evidence-fact {
        border: 1px solid #e2e7ee;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 8px;
        background: #fbfcfe;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data

def load_ground_truth(path_string: str):
    """Read the synthetic ground-truth records from the project data folder."""
    path = Path(path_string)
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Ground-truth JSON must contain a list of records.")

    return records


def load_text(file_path: Path) -> str:
    """Read a saved summary without modifying or regenerating it."""
    return file_path.read_text(encoding="utf-8")


def get_case_id(record: dict) -> str:
    return str(record["case_metadata"]["case_id"])


def get_original_summary(case_id: str):
    """Find an already-existing original summary for a case."""
    possible_files = (
        SUMMARY_DIR / f"{case_id}.txt",
        SUMMARY_DIR / f"{case_id}_summary.txt",
        SUMMARY_DIR / f"{case_id}_discharge_summary.txt",
    )

    for file_path in possible_files:
        if file_path.is_file():
            return file_path

    matches = sorted(SUMMARY_DIR.glob(f"{case_id}*.txt"))
    return matches[0] if matches else None


def get_claims_path(summary_path: Path) -> Path:
    """Map a summary to its persisted claims file."""
    return CLAIMS_DIR / f"{summary_path.stem}_claims.json"


# ============================================================
# SAVED 90-CASE CONTROLLED MUTATION EVALUATION
# ============================================================

def mutation_case_id(summary_reference: str, case_ids: list[str]) -> str | None:
    """Recover the case identifier from an ADD mutation summary filename."""
    name = Path(str(summary_reference or "")).name
    for case_id in case_ids:
        if name.endswith(f"_{case_id}.txt"):
            return case_id
    return None


def safe_project_path(relative_path: str) -> Path | None:
    """Resolve a project-relative report path while preventing path traversal."""
    candidate = (BASE_DIR / str(relative_path)).resolve()
    try:
        candidate.relative_to(BASE_DIR.resolve())
    except ValueError:
        return None
    return candidate


def mismatch_description(field_results: dict) -> str:
    """Summarize fields classified as contradicted for the results table."""
    if not isinstance(field_results, dict):
        return ""

    mismatches = []
    for field_name, details in field_results.items():
        if not isinstance(details, dict):
            continue
        if str(details.get("verdict", "")).upper() != "CONTRADICTED":
            continue

        claimed = details.get("claim_value", "")
        source = details.get("source_value", "")
        fact_id = details.get("matched_fact_id", "")
        description = f"{field_name}: {claimed!s} → {source!s}"
        if fact_id:
            description += f" [{fact_id}]"
        mismatches.append(description)

    return " | ".join(mismatches)


def load_additional_mutation_claims(
    case_records: dict[str, dict], selected_case: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Load prepared mutation claims without verifying them here.

    All returned claims are sent through the same reusable verification batch
    as the original-summary claims. Mutation IDs/labels are kept in parallel
    UI metadata and are not passed to the verifier or Gemini.
    """
    loaded_claims: list[dict] = []
    metadata: list[dict] = []
    load_errors: list[str] = []

    if not ADDITIONAL_MUTATION_CLAIMS_DIR.is_dir():
        return loaded_claims, metadata, load_errors

    for claim_path in sorted(ADDITIONAL_MUTATION_CLAIMS_DIR.glob("ADD-*_claims.json")):
        try:
            payload = json.loads(claim_path.read_text(encoding="utf-8"))
            claims_for_file = (
                payload.get("claims", []) if isinstance(payload, dict) else payload
            )
            summary_reference = (
                payload.get("summary_file", "") if isinstance(payload, dict) else ""
            )
            if not isinstance(claims_for_file, list) or not claims_for_file:
                continue

            case_id = mutation_case_id(summary_reference, list(case_records))
            if case_id != selected_case:
                continue

            for claim_index, claim in enumerate(claims_for_file):
                if not isinstance(claim, dict):
                    load_errors.append(
                        f"{claim_path.name}: claim {claim_index + 1} is not an object."
                    )
                    continue
                loaded_claims.append(dict(claim))
                metadata.append({
                    "source_type": "Controlled mutation test",
                    "test_id": claim_path.stem.removesuffix("_claims"),
                    "display_claim_id": str(claim.get("claim_id", "")),
                })
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            load_errors.append(
                f"{claim_path.name}: {type(exc).__name__}: {exc}"
            )

    return loaded_claims, metadata, load_errors


def read_claims_file(claims_path: Path):
    """Load claims from either the project's wrapper format or a raw list."""
    payload = json.loads(claims_path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        claims = payload
    elif isinstance(payload, dict):
        claims = payload.get("claims")
    else:
        claims = None

    if not isinstance(claims, list):
        raise ValueError(
            f"Saved claims file has an unexpected structure: {claims_path.name}"
        )

    return claims


def save_claims_file(claims_path: Path, summary_path: Path, claims: list):
    """Persist newly extracted claims for reuse; do not overwrite existing work."""
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)

    # Protect against accidentally replacing already extracted claims.
    if claims_path.exists():
        raise FileExistsError(
            f"Claims already exist at {claims_path}. Existing claims were not overwritten."
        )

    try:
        summary_reference = str(summary_path.relative_to(BASE_DIR))
    except ValueError:
        summary_reference = str(summary_path)

    payload = {
        "summary_file": summary_reference,
        "claim_count": len(claims),
        "claims": claims,
    }

    claims_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_seal_data_uri() -> str:
    """Use the user's seal image if present, otherwise a simple SVG placeholder."""
    if SEAL_IMAGE_PATH.is_file():
        mime_type, _ = mimetypes.guess_type(SEAL_IMAGE_PATH.name)
        mime_type = mime_type or "image/png"
        encoded = base64.b64encode(SEAL_IMAGE_PATH.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    # Embedded placeholder image: replace it by adding assets/hospital_seal.png.
    placeholder_svg = """<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240' viewBox='0 0 240 240'>
      <circle cx='120' cy='120' r='108' fill='#ffffff' stroke='#315d78' stroke-width='5'/>
      <circle cx='120' cy='120' r='91' fill='none' stroke='#9db6c5' stroke-width='2' stroke-dasharray='4 5'/>
      <path d='M105 67h30v38h38v30h-38v38h-30v-38H67v-30h38z' fill='#315d78'/>
      <text x='120' y='199' font-family='Arial, sans-serif' font-size='15' font-weight='bold' text-anchor='middle' fill='#315d78'>HOSPITAL SEAL</text>
    </svg>"""
    encoded = base64.b64encode(placeholder_svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


SUMMARY_PREVIEW_CHARACTERS = 1800


def compact_summary_text(summary: str) -> str:
    """Normalize line endings and remove blank rows to keep the document dense."""
    normalized = summary.replace("\r\n", "\n").replace("\r", "\n")
    # Preserve meaningful line breaks (headings, bullets, list items), but remove
    # every empty/whitespace-only row that creates large vertical gaps in the UI.
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    return "\n".join(lines).strip()


def get_summary_preview(summary: str, limit: int = SUMMARY_PREVIEW_CHARACTERS):
    """Return a readable excerpt and whether additional text exists."""
    cleaned = compact_summary_text(summary)
    if len(cleaned) <= limit:
        return cleaned, False

    excerpt = cleaned[:limit]
    last_line_break = excerpt.rfind("\n")
    # Avoid ending on a tiny partial line where practical.
    if last_line_break > int(limit * 0.6):
        excerpt = excerpt[:last_line_break]
    else:
        last_space = excerpt.rfind(" ")
        if last_space > int(limit * 0.75):
            excerpt = excerpt[:last_space]

    return excerpt.rstrip() + "\n\n…", True


def render_summary_document(summary: str):
    """Render a compact white paper-like document with a seal at upper right."""
    safe_summary = html.escape(summary)
    seal_uri = get_seal_data_uri()

    summary_html = MARKDOWN_RENDERER.render(
        safe_summary
    )
    document_html = f"""
      <div class="summary-document">
        <div class="summary-doc-header">
          <div class="summary-doc-label">Clinical discharge document</div>
          <img class="summary-seal" src="{seal_uri}" alt="Hospital seal placeholder" />
        </div>
        <div class="summary-body">{summary_html}</div>
      </div>
    """
    st.markdown(document_html, unsafe_allow_html=True)


# ============================================================
# LOAD RECORDS + HEADER
# ============================================================

try:
    records = load_ground_truth(str(GROUND_TRUTH_PATH))
except Exception as exc:
    st.error(f"Could not load patient records: {exc}")
    st.stop()

if not records:
    st.error("The ground-truth file contains no patient records.")
    st.stop()

st.title("🏥 Clinical Summary Verification System")
st.markdown(
    """
    <div class="app-subtitle">
      Select a patient case and run one unified verification. Original-summary
      claims and controlled experimental claims use the same evidence retrieval,
      deterministic verification, and optional Gemini adjudication pipeline.
      The source label distinguishes real summary claims from test inputs.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# SIDEBAR: PATIENT CASE ONLY
# ============================================================

st.sidebar.header("Patient Case")
case_ids = [get_case_id(record) for record in records]
selected_case = st.sidebar.selectbox("Patient Case", case_ids)
selected_record = next(record for record in records if get_case_id(record) == selected_case)

# The app always uses the saved original summary for the selected patient case.
# Original/error-injected summary selection is intentionally removed from the UI.
selected_summary_file = get_original_summary(selected_case)
if selected_summary_file is None:
    st.sidebar.error("No saved original discharge summary was found for this case.")
    st.info(
        "Place the original summary in `data/discharge_summaries/` for this case."
    )
    st.stop()

current_summary_key = str(selected_summary_file.resolve())
claims_path = get_claims_path(selected_summary_file)

# When the automatically selected original summary changes, load saved claims.
# Do not call the extraction model if a saved claims file already exists.
if st.session_state.get("active_summary_key") != current_summary_key:
    st.session_state["active_summary_key"] = current_summary_key
    st.session_state["claims_summary_file"] = current_summary_key
    st.session_state.pop("claims", None)
    st.session_state.pop("claims_load_error", None)
    st.session_state.pop("claims_loaded_from_disk", None)
    st.session_state.pop("summary_verification_results", None)
    st.session_state.pop("summary_verification_results_key", None)

    if claims_path.is_file():
        try:
            st.session_state["claims"] = read_claims_file(claims_path)
            st.session_state["claims_loaded_from_disk"] = True
        except Exception as exc:
            st.session_state["claims_load_error"] = str(exc)


# ============================================================
# 1. PATIENT RECORDS
# ============================================================

st.header("1. Patient Records")
metadata = selected_record.get("case_metadata", {})

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Case ID", selected_case)
with col2:
    st.metric("Specialty", metadata.get("primary_specialty", "N/A"))
with col3:
    st.metric("Admission Type", metadata.get("admission_type", "N/A"))

with st.expander("View Complete Ground-Truth Patient Record"):
    st.json(selected_record)

st.divider()


# ============================================================
# 2. DISCHARGE SUMMARY
# ============================================================

st.header("2. AI-Generated Discharge Summary")
st.caption(
    "Original saved discharge summary for the selected patient case. "
)

try:
    summary = load_text(selected_summary_file)
except Exception as exc:
    st.error(f"Could not read the selected summary: {exc}")
    st.stop()

summary_expanded_key = f"summary_expanded::{current_summary_key}"
if summary_expanded_key not in st.session_state:
    st.session_state[summary_expanded_key] = False

summary_preview, summary_has_more = get_summary_preview(summary)
summary_display_text = (
    compact_summary_text(summary)
    if st.session_state[summary_expanded_key]
    else summary_preview
)
render_summary_document(summary_display_text)

if summary_has_more:
    if not st.session_state[summary_expanded_key]:
        if st.button("Read More...", key=f"read_more::{selected_summary_file.stem}"):
            st.session_state[summary_expanded_key] = True
            st.rerun()
    else:
        if st.button("Show Less", key=f"show_less::{selected_summary_file.stem}"):
            st.session_state[summary_expanded_key] = False
            st.rerun()

st.caption(
    "Seal placeholder shown by default. To use your hospital seal, add your image at "
    "`assets/hospital_seal.png` relative to app.py."
)

st.divider()


# ============================================================
# 3. CLAIM EXTRACTION + DISK PERSISTENCE
# ============================================================

st.header("3. Atomic Claim Extraction")
claims = st.session_state.get("claims")

if st.session_state.get("claims_load_error"):
    st.error(
        "A saved claims file exists but could not be loaded. It was not overwritten "
        "or regenerated. Details: "
        + st.session_state["claims_load_error"]
    )
elif claims is not None and st.session_state.get("claims_loaded_from_disk"):
    st.success(
        f"Loaded {len(claims)} previously extracted claims from `{claims_path.name}`. "
        "No claim extraction was performed."
    )
elif claims is not None:
    st.success(f"{len(claims)} claims are available for this summary.")
elif claims_path.is_file():
    # Defensive case: file exists but session state has not been hydrated.
    try:
        claims = read_claims_file(claims_path)
        st.session_state["claims"] = claims
        st.session_state["claims_loaded_from_disk"] = True
        st.success(f"Loaded {len(claims)} existing claims from `{claims_path.name}`.")
    except Exception as exc:
        st.error(f"Could not load saved claims: {exc}")
else:
    st.info(
        "No saved claims were found for this summary. Extract Claims will run the "
        "claim extractor once and save the results so they can be reused."
    )

if claims_path.is_file():
    if st.button("Reload Saved Claims", help="Reloads the claims JSON file only; it does not call the extractor."):
        try:
            st.session_state["claims"] = read_claims_file(claims_path)
            st.session_state["claims_loaded_from_disk"] = True
            st.session_state.pop("claims_load_error", None)
            st.session_state.pop("summary_verification_results", None)
            st.session_state.pop("summary_verification_results_key", None)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load saved claims: {exc}")
elif not st.session_state.get("claims_load_error"):
    if st.button("Extract Claims and Save", type="primary"):
        # Re-check the file immediately before extraction so an existing file is
        # always preferred over generating a second set of claims.
        if claims_path.is_file():
            try:
                st.session_state["claims"] = read_claims_file(claims_path)
                st.session_state["claims_loaded_from_disk"] = True
                st.rerun()
            except Exception as exc:
                st.error(f"A claims file exists but could not be loaded: {exc}")
        else:
            with st.spinner("Extracting factual claims for the first time..."):
                try:
                    extracted_claims = extract_claims(summary)
                    if not isinstance(extracted_claims, list):
                        raise ValueError(
                            "The claim extractor did not return a list of claims."
                        )
                    save_claims_file(claims_path, selected_summary_file, extracted_claims)
                    st.session_state["claims"] = extracted_claims
                    st.session_state["claims_loaded_from_disk"] = True
                    st.session_state.pop("claims_load_error", None)
                    st.session_state.pop("summary_verification_results", None)
                    st.session_state.pop("summary_verification_results_key", None)
                    st.success(
                        f"Saved {len(extracted_claims)} claims to `{claims_path.name}`. "
                        "Future runs will reuse this file."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Claim extraction or saving failed: {exc}")

claims = st.session_state.get("claims")

if claims is not None:
    st.caption(f"Claims file: `{claims_path}`")
    if not claims:
        st.warning("The saved claim list is empty.")
    else:
        claim_rows = []
        for claim in claims:
            claim_rows.append(
                {
                    "Claim ID": claim.get("claim_id", ""),
                    "Category": claim.get("category", ""),
                    "Subject": claim.get("subject", ""),
                    "Relation": claim.get("relation", ""),
                    "Value": claim.get("value", ""),
                    "Unit": claim.get("unit", ""),
                    "Route": claim.get("route", ""),
                    "Frequency": claim.get("frequency", ""),
                    "Date": claim.get("date_time", ""),
                }
            )
        st.dataframe(claim_rows, use_container_width=True, hide_index=True)

st.divider()


# ============================================================
# 4. SELECTED CLAIM DETAILS
# ============================================================

st.header("4. Selected Claim Details")
selected_claim = None

if not claims:
    st.info("Load or extract claims to inspect an individual claim.")
else:
    def claim_label(index: int) -> str:
        claim = claims[index]
        claim_id = claim.get("claim_id", f"Claim {index + 1}")
        subject = claim.get("subject", "Unspecified subject")
        return f"{claim_id} · {subject}"

    selected_claim_index = st.selectbox(
        "Select a claim",
        options=list(range(len(claims))),
        format_func=claim_label,
        key=f"claim_selector_{selected_case}_{selected_summary_file.stem}",
    )
    selected_claim = claims[selected_claim_index]

    st.subheader("Extracted Claim")
    claim_col1, claim_col2, claim_col3 = st.columns(3)
    with claim_col1:
        st.write(f"**Claim ID:** {selected_claim.get('claim_id', '')}")
        st.write(f"**Category:** {selected_claim.get('category', '')}")
        st.write(f"**Subject:** {selected_claim.get('subject', '')}")
    with claim_col2:
        st.write(f"**Relation:** {selected_claim.get('relation', '')}")
        st.write(f"**Value:** {selected_claim.get('value', '')}")
        st.write(f"**Unit:** {selected_claim.get('unit', '')}")
    with claim_col3:
        st.write(f"**Route:** {selected_claim.get('route', '')}")
        st.write(f"**Frequency:** {selected_claim.get('frequency', '')}")
        st.write(f"**Date/time:** {selected_claim.get('date_time', '')}")

    st.subheader("Exact Source Text From Summary")
    source_text = selected_claim.get("source_text", "")
    if source_text:
        st.info(source_text)
    else:
        st.info("No source text was stored for this claim.")

    with st.expander("View Complete Claim JSON"):
        st.json(selected_claim)

st.divider()


# ============================================================
# 5. UNIFIED CLAIM VERIFICATION
# ============================================================

st.header("5. Unified Claim Verification")
st.markdown(
    "Original discharge-summary claims and the selected patient's existing "
    "controlled mutation claims are evaluated through the same verification "
    "pipeline. The source column identifies which claims come from the original "
    "summary and which are experimental test inputs."
)

if not claims:
    st.info("Load or extract claims before running verification.")
else:
    verification_key = current_summary_key
    saved_results_key = st.session_state.get("summary_verification_results_key")

    # Load mutation inputs up front only to show the combined run size. The
    # mutation labels/test IDs remain UI metadata and are never sent to Gemini.
    case_record_map = {get_case_id(record): record for record in records}
    preview_mutation_claims, preview_mutation_metadata, preview_mutation_errors = (
        load_additional_mutation_claims(case_record_map, selected_case)
    )
    original_claim_count = len(claims)
    mutation_claim_count = len(preview_mutation_claims)
    total_input_claim_count = original_claim_count + mutation_claim_count

    if st.button(
        f"Verify All Claims ({total_input_claim_count})",
        type="primary",
        key=f"verify_all_claims::{selected_summary_file.stem}",
    ):
        # Create one mixed input list, but replace claim IDs with opaque unique
        # per-run IDs. This prevents collisions between original claims and
        # controlled mutation claims that may reuse the same C### identifiers.
        # Test IDs and source labels are kept outside the model input.
        combined_claims: list[dict] = []
        result_metadata: list[dict] = []
        run_claim_ids: list[str] = []

        for index, original_claim in enumerate(claims):
            display_id = str(original_claim.get("claim_id", f"C{index + 1:03d}"))
            internal_id = f"VERIFY-{index + 1:06d}"
            prepared_claim = dict(original_claim)
            prepared_claim["claim_id"] = internal_id
            combined_claims.append(prepared_claim)
            run_claim_ids.append(internal_id)
            result_metadata.append({
                "source_type": "Original discharge summary",
                "test_id": "",
                "display_claim_id": display_id,
            })

        for index, mutation_claim in enumerate(preview_mutation_claims):
            metadata = preview_mutation_metadata[index]
            display_id = str(
                metadata.get("display_claim_id") or f"Mutation claim {index + 1}"
            )
            internal_id = f"VERIFY-{len(combined_claims) + 1:06d}"
            prepared_claim = dict(mutation_claim)
            prepared_claim["claim_id"] = internal_id
            combined_claims.append(prepared_claim)
            run_claim_ids.append(internal_id)
            result_metadata.append(metadata)

        ground_truth_facts = selected_record.get("ground_truth_facts", [])
        progress = st.progress(0, text="Preparing unified verification...")
        total_claims = len(combined_claims)

        def update_verification_progress(completed: int, total: int):
            progress.progress(
                completed / total if total else 1.0,
                text=f"Retrieved evidence and verified {completed} of {total} claims...",
            )

        # One call: all claims go through the same retrieval, deterministic
        # verification, and optional evidence-constrained Gemini fallback.
        batch_results, semantic_status = verify_claims_batch(
            combined_claims,
            ground_truth_facts,
            record=selected_record,
            top_k=20,
            progress_callback=update_verification_progress,
        )

        # Restore human-readable claim IDs and attach source metadata only after
        # semantic adjudication has finished. Gemini never receives mutation
        # labels, expected verdicts, test IDs, or the scoring manifest.
        for index, item in enumerate(batch_results):
            metadata = result_metadata[index]
            claim_payload = item.get("claim") or {}
            claim_payload["claim_id"] = metadata.get("display_claim_id", "")
            item["claim"] = claim_payload
            item["source_type"] = metadata.get("source_type", "Original discharge summary")
            item["test_id"] = metadata.get("test_id", "")
            item["display_claim_id"] = metadata.get("display_claim_id", "")
            item["pipeline_claim_id"] = run_claim_ids[index]
            result_payload = item.get("result")
            if isinstance(result_payload, dict):
                result_payload["claim_id"] = metadata.get("display_claim_id", "")

        progress.empty()
        st.session_state["summary_verification_results"] = batch_results
        st.session_state["summary_verification_results_key"] = verification_key
        st.session_state["summary_verification_semantic_status"] = semantic_status
        st.session_state["summary_verification_semantic_status_key"] = verification_key
        st.session_state["summary_verification_summary_claim_count"] = original_claim_count
        st.session_state["summary_verification_mutation_count"] = mutation_claim_count
        st.session_state["summary_verification_mutation_load_errors"] = preview_mutation_errors
        st.session_state["summary_verification_results_case_id"] = selected_case
        st.rerun()

    stored_batch_results = st.session_state.get("summary_verification_results")
    results_case_id = st.session_state.get("summary_verification_results_case_id")
    if (
        isinstance(stored_batch_results, list)
        and saved_results_key == verification_key
        and results_case_id == selected_case
    ):
        batch_results = stored_batch_results
        semantic_status = st.session_state.get("summary_verification_semantic_status", {})
        semantic_status_key = st.session_state.get("summary_verification_semantic_status_key")
        mutation_load_errors = st.session_state.get(
            "summary_verification_mutation_load_errors", []
        )

        if semantic_status_key == verification_key and semantic_status.get("enabled"):
            if semantic_status.get("error"):
                st.warning(
                    "Gemini semantic adjudication did not complete ("
                    + str(semantic_status["error"])
                    + "). Deterministic verification results were retained."
                )
            elif semantic_status.get("considered", 0):
                accepted_count = int(
                    semantic_status.get("accepted", semantic_status.get("applied", 0)) or 0
                )
                st.info(
                    "Gemini reviewed "
                    + str(semantic_status.get("considered", 0))
                    + " eligible claim(s). Accepted decisions: "
                    + str(accepted_count)
                    + " ("
                    + str(semantic_status.get("supported", 0))
                    + " SUPPORTED, "
                    + str(semantic_status.get("contradicted", 0))
                    + " CONTRADICTED, "
                    + str(semantic_status.get("abstained", 0))
                    + " INSUFFICIENT_EVIDENCE)."
                )
                if semantic_status.get("limit_reached"):
                    st.caption(
                        "The configured Gemini review limit was reached; other eligible "
                        "claims retain their deterministic results."
                    )
            else:
                st.caption(
                    "No claims met the current rules for Gemini semantic adjudication; "
                    "deterministic results were retained."
                )

        if mutation_load_errors:
            st.warning(
                f"{len(mutation_load_errors)} controlled mutation file(s) could not be loaded. "
                "See the full report for details."
            )

        valid_results = [item for item in batch_results if item.get("result") is not None]
        error_results = [item for item in batch_results if item.get("error")]

        def item_verdict(item: dict) -> str:
            return str((item.get("result") or {}).get("verdict", "UNKNOWN")).upper()

        def semantic_verdict_changed(item: dict) -> bool:
            result = item.get("result") or {}
            adjudication = result.get("semantic_adjudication") or {}
            if not adjudication.get("applied"):
                return False
            if "verdict_changed" in adjudication:
                return bool(adjudication.get("verdict_changed"))
            previous = adjudication.get("previous_verdict")
            final = result.get("verdict")
            return bool(
                previous and final
                and str(previous).upper() != str(final).upper()
            )

        def source_label(item: dict) -> str:
            if item.get("source_type") == "Controlled mutation test":
                return "Controlled mutation test"
            return "Original discharge summary"

        def make_result_row(item: dict) -> dict:
            claim = item.get("claim") or {}
            result = item.get("result") or {}
            field_results = result.get("field_results") or {}
            mismatches = mismatch_description(field_results)
            claim_value = " ".join(
                str(claim.get(key, "")).strip()
                for key in ("value", "unit")
                if str(claim.get(key, "")).strip()
            )
            return {
                "Source": source_label(item),
                "Test ID": item.get("test_id", ""),
                "Claim ID": item.get("display_claim_id") or claim.get("claim_id", ""),
                "Subject": claim.get("subject", ""),
                "Final verdict": item_verdict(item),
                "Review": "by Gemini" if semantic_verdict_changed(item) else "",
                "Category": claim.get("category", ""),
                "Relation": claim.get("relation", ""),
                "Claimed value": claim_value,
                "Route": claim.get("route", ""),
                "Frequency": claim.get("frequency", ""),
                "Discrepancy type": result.get("error_type") or "",
                "Mismatch details": mismatches,
            }

        original_count = sum(
            item.get("source_type") != "Controlled mutation test" for item in batch_results
        )
        mutation_count = sum(
            item.get("source_type") == "Controlled mutation test" for item in batch_results
        )
        contradicted_items = [
            item for item in valid_results if item_verdict(item) == "CONTRADICTED"
        ]
        review_items = [
            item for item in valid_results
            if item_verdict(item) in {"INSUFFICIENT_EVIDENCE", "PARTIALLY_SUPPORTED"}
        ]
        gemini_changed_to_supported = sum(
            1 for item in valid_results
            if semantic_verdict_changed(item) and item_verdict(item) == "SUPPORTED"
        )

        st.subheader("Unified Verification Results")
        # Present the original-summary and controlled-mutation claims as one
        # combined verification batch. Source provenance remains available on
        # individual result rows and in the downloadable research report.
        metric_cols = st.columns(2)
        with metric_cols[0]:
            st.metric("Total claims evaluated", len(batch_results))
        with metric_cols[1]:
            st.metric("Processing errors", len(error_results))

        if gemini_changed_to_supported:
            st.caption(
                f"Gemini resolved {gemini_changed_to_supported} claim(s) as SUPPORTED. "
                "They are omitted from the review lists; all outcomes remain in the downloadable report."
            )

        # Doctors/reviewers see only claims that require attention. SUPPORTED
        # claims are retained in the full JSON report for research analysis.
        st.subheader(f"Detected Discrepancies ({len(contradicted_items)})")
        if contradicted_items:
            st.caption(
                "These claims were classified as CONTRADICTED. The Source column "
                "distinguishes original-summary claims from controlled experimental inputs."
            )
            st.dataframe(
                [make_result_row(item) for item in contradicted_items],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Inspect discrepancy evidence"):
                for item in contradicted_items:
                    claim = item.get("claim") or {}
                    result = item.get("result") or {}
                    title_prefix = (
                        f"{item.get('test_id')} · "
                        if item.get("test_id") else ""
                    )
                    label = (
                        f"{title_prefix}{claim.get('claim_id', 'Claim')} · "
                        f"{claim.get('subject', 'Unspecified subject')}"
                    )
                    with st.expander(label):
                        st.write("**Source:**", source_label(item))
                        if item.get("test_id"):
                            st.write("**Test ID:**", item["test_id"])
                        st.write("**Final verdict:**", "CONTRADICTED")
                        if semantic_verdict_changed(item):
                            st.caption("by Gemini")
                        st.write("**Claim text:**", claim.get("source_text", "") or "Not available")
                        st.write("**Category / relation:**", claim.get("category", ""), "/", claim.get("relation", ""))
                        st.write("**Claimed value:**", claim.get("value", ""), claim.get("unit", ""))
                        st.write("**Error type:**", result.get("error_type") or "Field mismatch")
                        if result.get("semantic_adjudication", {}).get("applied"):
                            rationale = result["semantic_adjudication"].get("rationale", "")
                            if rationale:
                                st.write("**Gemini-generated rationale:**", rationale)
                        st.write(
                            "**Matched fact IDs:**",
                            ", ".join(result.get("matched_fact_ids") or []) or "None",
                        )
                        with st.expander("Field-by-field comparison"):
                            st.json(result.get("field_results", {}))
                        candidate_by_id = {
                            candidate.get("fact", {}).get("fact_id"): candidate.get("fact", {})
                            for candidate in item.get("candidates", [])
                            if isinstance(candidate, dict)
                            and isinstance(candidate.get("fact"), dict)
                        }
                        for fact_id in result.get("matched_fact_ids") or []:
                            fact = candidate_by_id.get(fact_id, {})
                            if fact:
                                st.write(
                                    f"{fact.get('subject', 'Source fact')} — "
                                    f"{fact.get('relation', 'relation')}: "
                                    f"{fact.get('value', '')} {fact.get('unit', '')} "
                                    f"(Fact ID: {fact_id})"
                                )
        else:
            st.success("No claims were classified as contradicted in this run.")

        st.subheader(f"Claims Requiring Further Review ({len(review_items)})")
        if review_items:
            st.caption(
                "These claims could not be fully resolved from the available evidence. "
                "They are not presented as confirmed contradictions."
            )
            st.dataframe(
                [make_result_row(item) for item in review_items],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Inspect claims requiring further review"):
                for item in review_items:
                    claim = item.get("claim") or {}
                    result = item.get("result") or {}
                    test_prefix = f"{item.get('test_id')} · " if item.get("test_id") else ""
                    label = (
                        f"{test_prefix}{claim.get('claim_id', 'Claim')} · "
                        f"{claim.get('subject', 'Unspecified subject')} — {item_verdict(item)}"
                    )
                    with st.expander(label):
                        st.write("**Source:**", source_label(item))
                        st.write("**Final verdict:**", item_verdict(item))
                        if semantic_verdict_changed(item):
                            st.caption("by Gemini")
                        st.write("**Claim text:**", claim.get("source_text", "") or "Not available")
                        adjudication = result.get("semantic_adjudication") or {}
                        if adjudication.get("rationale"):
                            st.write("**Gemini-generated rationale:**", adjudication["rationale"])
                        st.write("**Matched fact IDs:**", ", ".join(result.get("matched_fact_ids") or []) or "None")
                        with st.expander("Field-by-field comparison"):
                            st.json(result.get("field_results", {}))

        if error_results:
            st.warning(
                f"Verification could not be completed for {len(error_results)} claim(s). "
                "These claims are preserved in the full report and should not be treated as supported."
            )

        # Full research report intentionally retains SUPPORTED claims as well as
        # review-needed outcomes. The UI above hides SUPPORTED claims to reduce
        # reviewer workload, but the data remains available for metrics/auditing.
        report_payload = {
            "case_id": selected_case,
            "original_summary_file": selected_summary_file.name,
            "evaluation_scope": "one_unified_batch_original_summary_plus_preextracted_controlled_mutations",
            "original_summary_claims_checked": original_count,
            "controlled_mutation_claims_checked": mutation_count,
            "total_claims_checked": len(batch_results),
            "contradicted_count": len(contradicted_items),
            "needs_review_count": len(review_items),
            "processing_error_count": len(error_results),
            "gemini_changed_to_supported_count": gemini_changed_to_supported,
            "semantic_adjudication_status": semantic_status,
            "mutation_input_load_errors": mutation_load_errors,
            "results": [
                {
                    "source_type": source_label(item),
                    "test_id": item.get("test_id", ""),
                    "claim": item.get("claim", {}),
                    "verdict_result": item.get("result"),
                    "processing_error": item.get("error"),
                    "candidate_evidence": [
                        {
                            "score": candidate.get("score"),
                            "match_reasons": candidate.get("match_reasons", []),
                            "fact": candidate.get("fact", {}),
                        }
                        for candidate in item.get("candidates", [])
                    ],
                }
                for item in batch_results
            ],
        }
        st.download_button(
            "Download Full Unified Verification Report (JSON)",
            data=json.dumps(report_payload, indent=2, ensure_ascii=False),
            file_name=f"{selected_summary_file.stem}_unified_verification_report.json",
            mime="application/json",
            key=f"download_unified_verification_report::{selected_summary_file.stem}",
        )
    else:
        st.info(
            "Click **Verify All Claims** to run the original summary claims and the "
            "selected patient's available controlled mutation claims through one shared pipeline."
        )

st.divider()

st.caption(
    "Research note: the existing controlled mutation set modifies already-extracted "
    "claims. It tests retrieval, deterministic comparison, and semantic adjudication, "
    "but does not test whether the LLM extractor can detect errors introduced into "
    "the complete discharge-summary text. A separate document-level mutation study "
    "should inject errors into copies of the summaries and rerun claim extraction."
)
