"""Clinical Summary Verification System Streamlit app.

The app reads existing summaries and patient records from disk. Extracted claims
are persisted under data/claims/ and reused on subsequent runs; existing
summaries and claims are never regenerated automatically. The main verification
view combines saved summary claims with controlled mutation claims for the selected case.
"""

import base64
import html
import json
import mimetypes
from pathlib import Path

import streamlit as st

from src.claim_extractor import extract_claims
from src.evidence_builder import retrieve_candidate_facts
from src.verification_engine import verify_claim_against_facts
from src.semantic_adjudicator import maybe_apply_semantic_adjudications

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


def load_mutation_claim(prediction: dict) -> dict:
    """Load the saved atomic claim corresponding to a controlled mutation."""
    claim_path = safe_project_path(prediction.get("claim_file", ""))
    if claim_path is None or not claim_path.is_file():
        return {}
    try:
        payload = json.loads(claim_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    claims_payload = payload.get("claims") if isinstance(payload, dict) else payload
    return claims_payload[0] if isinstance(claims_payload, list) and claims_payload else {}


def mismatch_description(field_results: dict) -> str:
    """Summarize contradicted fields without using evaluation mutation labels."""
    mismatches = []
    if not isinstance(field_results, dict):
        return ""
    for field_name, details in field_results.items():
        if not isinstance(details, dict):
            continue
        if str(details.get("verdict", "")).upper() == "CONTRADICTED":
            claimed = details.get("claim_value", "")
            source = details.get("source_value", "")
            fact_id = details.get("matched_fact_id", "")
            part = f"{field_name}: {claimed!s} → {source!s}"
            if fact_id:
                part += f" [{fact_id}]"
            mismatches.append(part)
    return " | ".join(mismatches)


def load_saved_mutation_predictions() -> list[dict]:
    """Load previously computed predictions, never scoring-manifest labels."""
    if not ADDITIONAL_MUTATION_REPORT_PATH.is_file():
        return []
    try:
        report = json.loads(
            ADDITIONAL_MUTATION_REPORT_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return []
    predictions = report.get("predictions", []) if isinstance(report, dict) else []
    return predictions if isinstance(predictions, list) else []


def run_additional_mutation_verification(
    case_records: dict[str, dict], selected_case: str
) -> list[dict]:
    """Rerun only the saved mutation claims belonging to the selected patient case.

    Mutation labels and expected verdicts are not read by this verification path.
    """
    outcomes = []
    if not ADDITIONAL_MUTATION_CLAIMS_DIR.is_dir():
        return outcomes

    for claim_path in sorted(ADDITIONAL_MUTATION_CLAIMS_DIR.glob("ADD-*_claims.json")):
        try:
            payload = json.loads(claim_path.read_text(encoding="utf-8"))
            claims_for_file = payload.get("claims", []) if isinstance(payload, dict) else payload
            summary_reference = payload.get("summary_file", "") if isinstance(payload, dict) else ""
            if not isinstance(claims_for_file, list) or not claims_for_file:
                continue
            case_id = mutation_case_id(summary_reference, list(case_records))
            if case_id != selected_case:
                continue
            record = case_records[case_id]
            claim = claims_for_file[0]
            candidates = retrieve_candidate_facts(
                claim, record.get("ground_truth_facts", []), top_k=20, record=record
            )
            result = verify_claim_against_facts(claim, candidates, record=record)
            outcomes.append({
                "test_id": claim_path.stem.removesuffix("_claims"),
                "case_id": case_id,
                "claim": claim,
                "result": result,
                "candidates": candidates,
                "processing_error": None,
            })
        except Exception as exc:
            outcomes.append({
                "test_id": claim_path.stem.removesuffix("_claims"),
                "case_id": selected_case,
                "claim": {},
                "result": None,
                "candidates": [],
                "processing_error": f"{type(exc).__name__}: {exc}",
            })
    return outcomes


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
      Select a patient case and run one verification action. The results combine
      discrepancies found in the saved original summary with controlled mutation
      test claims for that patient, clearly identifying each source.
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
# 5. INDEPENDENT EVIDENCE RETRIEVAL + WHOLE-SUMMARY VERIFICATION
# ============================================================

st.header("5. Claim Verification")
st.markdown(
    "Click **Verify Entire Discharge Summary** to check the selected patient's "
    "summary claims against the ground-truth record. "
    "All claims classified as CONTRADICTED appear in the discrepancy list. "
)

if not claims:
    st.info("Load or extract claims before verifying the complete discharge summary.")
else:
    verification_key = current_summary_key
    saved_results_key = st.session_state.get("summary_verification_results_key")

    if st.button(
        f"Verify Entire Discharge Summary ({len(claims)} claims)",
        type="primary",
        key=f"verify_entire_summary::{selected_summary_file.stem}",
    ):
        ground_truth_facts = selected_record.get("ground_truth_facts", [])
        batch_results = []

        progress = st.progress(0, text="Preparing full-summary verification...")
        total_claims = len(claims)

        for index, claim in enumerate(claims):
            claim_id = claim.get("claim_id", f"Claim {index + 1}")
            try:
                candidates = retrieve_candidate_facts(
                    claim,
                    ground_truth_facts,
                    top_k=20,
                    record=selected_record,
                )
                result = verify_claim_against_facts(
                    claim,
                    candidates,
                    record=selected_record,
                )
                batch_results.append(
                    {
                        "claim": claim,
                        "candidates": candidates,
                        "result": result,
                        "error": None,
                        "source_type": "Discharge summary",
                        "test_id": "",
                    }
                )
            except Exception as exc:
                batch_results.append(
                    {
                        "claim": claim,
                        "candidates": [],
                        "result": None,
                        "error": str(exc),
                        "source_type": "Discharge summary",
                        "test_id": "",
                    }
                )

            progress.progress(
                (index + 1) / total_claims,
                text=f"Verified {index + 1} of {total_claims} claims...",
            )

        # Optional, single-batch Gemini fallback is disabled by default. It only
        # considers uncertain claims with exactly one unresolved field and never
        # overrides a deterministic CONTRADICTED result. Any failure leaves the
        # deterministic results intact.
        try:
            batch_results, semantic_status = maybe_apply_semantic_adjudications(
                batch_results
            )
        except Exception as exc:
            semantic_status = {
                "enabled": True,
                "applied": 0,
                "error": type(exc).__name__,
                "reason": "semantic fallback failed; deterministic results retained",
            }

        # Include controlled mutation claims for the selected patient in this same
        # result batch, rather than displaying them in a separate UI section.
        # The verifier receives each claim and ground-truth facts, never the mutation
        # manifest, expected verdict, or mutation label.
        case_record_map = {get_case_id(record): record for record in records}
        mutation_results = run_additional_mutation_verification(
            case_record_map, selected_case=selected_case
        )
        for mutation_item in mutation_results:
            batch_results.append(
                {
                    "claim": mutation_item.get("claim", {}),
                    "candidates": mutation_item.get("candidates", []),
                    "result": mutation_item.get("result"),
                    "error": mutation_item.get("processing_error"),
                    "source_type": "Controlled mutation test",
                    "test_id": mutation_item.get("test_id", ""),
                }
            )

        progress.empty()
        st.session_state["summary_verification_results"] = batch_results
        st.session_state["summary_verification_results_key"] = verification_key
        st.session_state["summary_verification_semantic_status"] = semantic_status
        st.session_state["summary_verification_semantic_status_key"] = verification_key
        st.session_state["summary_verification_summary_claim_count"] = len(claims)
        st.session_state["summary_verification_mutation_count"] = len(mutation_results)
        st.session_state["summary_verification_results_case_id"] = selected_case
        st.rerun()

    # Display the most recent results only when they belong to this summary.
    batch_results = st.session_state.get("summary_verification_results")
    results_case_id = st.session_state.get("summary_verification_results_case_id")
    if batch_results and saved_results_key == verification_key and results_case_id == selected_case:
        semantic_status = st.session_state.get("summary_verification_semantic_status", {})
        semantic_status_key = st.session_state.get("summary_verification_semantic_status_key")
        if semantic_status_key == verification_key and semantic_status.get("enabled"):
            if semantic_status.get("error"):
                st.warning(
                    "Optional Gemini semantic fallback did not run successfully "
                    "(" + str(semantic_status["error"]) + "). Deterministic verification "
                    "results were retained."
                )
            elif semantic_status.get("applied", 0):
                st.info(
                    "Optional Gemini semantic fallback supported "
                    + str(semantic_status.get("applied", 0))
                    + " previously uncertain claim(s), using cited retrieved evidence. "
                    "Deterministic contradictions were not overridden."
                )
                if semantic_status.get("limit_reached"):
                    st.caption(
                        "The configured semantic adjudication limit was reached; remaining "
                        "uncertain claims keep their deterministic verdicts."
                    )
            else:
                st.caption(
                    "Optional Gemini semantic fallback ran but did not promote any claims; "
                    "deterministic verdicts were retained."
                )

        valid_results = [item for item in batch_results if item.get("result") is not None]
        error_results = [item for item in batch_results if item.get("error")]

        def item_verdict(item):
            return str(item.get("result", {}).get("verdict", "UNKNOWN")).upper()

        def item_verdict(item):
            return str(item.get("result", {}).get("verdict", "UNKNOWN")).upper()

        def make_result_row(item):
            claim = item.get("claim", {})
            result = item.get("result") or {}
            value_parts = [
                str(claim.get("value", "")).strip(),
                str(claim.get("unit", "")).strip(),
            ]
            claim_value = " ".join(part for part in value_parts if part)
            field_results = result.get("field_results") or {}
            mismatches = []
            for field_name, field_result in field_results.items():
                if str(field_result.get("verdict", "")).upper() == "CONTRADICTED":
                    claimed = field_result.get("claim_value", "")
                    source = field_result.get("source_value", "")
                    fact_id = field_result.get("matched_fact_id", "")
                    mismatch = f"{field_name}: {claimed!s} → {source!s}"
                    if fact_id:
                        mismatch += f" [{fact_id}]"
                    mismatches.append(mismatch)

            return {
                "Source": item.get("source_type", "Discharge summary"),
                "Mutation Test ID": item.get("test_id", ""),
                "Claim ID": claim.get("claim_id", ""),
                "Subject": claim.get("subject", ""),
                "Category": claim.get("category", ""),
                "Relation": claim.get("relation", ""),
                "Value": claim_value,
                "Route": claim.get("route", ""),
                "Frequency": claim.get("frequency", ""),
                "Discrepancy type": result.get("error_type") or "Field mismatch",
                "Mismatch details": " | ".join(mismatches),
                "Matched fact IDs": ", ".join(result.get("matched_fact_ids") or []),
                "Source text": claim.get("source_text", ""),
                "Processing error": item.get("error") or "",
            }

        contradicted_items = [
            item for item in valid_results if item_verdict(item) == "CONTRADICTED"
        ]

        st.subheader("Discrepancy Detection Results")
        mutation_case_count = sum(
            item.get("source_type") == "Controlled mutation test" for item in batch_results
        )
        summary_claim_count = sum(
            item.get("source_type", "Discharge summary") == "Discharge summary"
            for item in batch_results
        )
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric("Summary claims checked", summary_claim_count)
        with summary_cols[2]:
            st.metric("Potential discrepancies detected", len(contradicted_items))

        

        if error_results:
            st.warning(
                f"Verification could not be completed for {len(error_results)} claim(s). "
                "Those claims are retained in the downloadable full verification report."
            )

        with st.expander("Discrepancy categories covered by the 90-case controlled test"):
            st.caption(
                "These categories were detected in a controlled test of existing atomic claims. "
                "This is a tested scope, not a guarantee that every real-world instance will be detected."
            )
            tested_categories = [
                ("Medication dose mismatch", 12),
                ("Medication route mismatch", 12),
                ("Medication frequency mismatch", 12),
                ("Multiple medication-field mismatch", 12),
                ("Laboratory numeric-value mismatch", 12),
                ("Unit mismatch", 12),
                ("Wrong event date", 6),
                ("Wrong laboratory timepoint/value", 7),
                ("Narrative polarity or negation", 3),
                ("Allergy polarity", 1),
                ("Duration mismatch", 1),
            ]
            st.dataframe(
                [
                    {"Tested discrepancy category": category, "Controlled errors": count}
                    for category, count in tested_categories
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "The 90/90 result applies only to these constructed verification-stage cases. "
                "The LLM claim extractor was not rerun for that experiment."
            )

        st.subheader(f"Detected Discrepancies ({len(contradicted_items)})")
        st.caption(
            "Review the extracted claim alongside the matched source fact. These are "
            "potential factual discrepancies identified by the verifier and should be "
            "interpreted with their evidence, not as automatically confirmed clinical errors."
        )
        if contradicted_items:
            st.dataframe(
                [make_result_row(item) for item in contradicted_items],
                use_container_width=True,
                hide_index=True,
            )
            st.subheader("Discrepancy Evidence")
            st.caption(
                "Expand a discrepancy to inspect its source text, field-level comparison, "
                "matched facts and retrieved candidates."
            )
            for item in contradicted_items:
                claim = item.get("claim", {})
                result = item.get("result") or {}
                source_prefix = (
                    f"{item.get('test_id')} · "
                    if item.get("source_type") == "Controlled mutation test"
                    else "Summary · "
                )
                claim_label = (
                    f"{source_prefix}{claim.get('claim_id', 'Claim')} · "
                    f"{claim.get('subject', 'Unspecified subject')} — "
                    f"{result.get('error_type') or 'CONTRADICTED'}"
                )
                with st.expander(claim_label):
                    st.write("**Source:**", item.get("source_type", "Discharge summary"))
                    if item.get("test_id"):
                        st.write("**Mutation Test ID:**", item.get("test_id"))
                    st.write("**Claim ID:**", claim.get("claim_id", ""))
                    st.write("**Category:**", claim.get("category", ""))
                    st.write("**Relation:**", claim.get("relation", ""))
                    st.write("**Claim value:**", claim.get("value", ""), claim.get("unit", ""))
                    st.write("**Route:**", claim.get("route", ""))
                    st.write("**Frequency:**", claim.get("frequency", ""))
                    st.write("**Duration:**", claim.get("duration", ""))
                    st.write("**Source text:**", claim.get("source_text", ""))
                    st.write("**Discrepancy type:**", result.get("error_type") or "Field mismatch")
                    if result.get("semantic_adjudication", {}).get("applied"):
                        adjudication = result["semantic_adjudication"]
                        st.caption(
                            "Semantic fallback adjudicated the unresolved field "
                            f"`{adjudication.get('unresolved_field', '')}` using "
                            f"retrieved fact IDs: {', '.join(adjudication.get('supporting_fact_ids', []))}. "
                            "Deterministic field checks remain visible below."
                        )
                    st.write(
                        "**Matched fact IDs:**",
                        ", ".join(result.get("matched_fact_ids") or []) or "None",
                    )

                    with st.expander("Field-by-field comparison"):
                        field_results = result.get("field_results", {})
                        if field_results:
                            st.json(field_results)
                        else:
                            st.info("No field-level comparison was returned.")

                    candidate_by_id = {
                        candidate.get("fact", {}).get("fact_id"): candidate
                        for candidate in item.get("candidates", [])
                        if isinstance(candidate, dict)
                        and isinstance(candidate.get("fact"), dict)
                    }
                    matched_fact_ids = result.get("matched_fact_ids") or []
                    if matched_fact_ids:
                        st.markdown("**Matched source evidence**")
                        for fact_id in matched_fact_ids:
                            candidate = candidate_by_id.get(fact_id, {})
                            fact = candidate.get("fact", {})
                            st.markdown(
                                f"- **{fact.get('subject', 'Source fact')}** — "
                                f"{fact.get('relation', 'relation not specified')}: "
                                f"{fact.get('value', '')} {fact.get('unit', '')} "
                                f"(Fact ID: `{fact_id}`, section: "
                                f"`{fact.get('source_section', 'N/A')}`)"
                            )

                    with st.expander("Retrieved candidate evidence"):
                        candidates = item.get("candidates", [])
                        if candidates:
                            candidate_rows = []
                            for candidate in candidates:
                                fact = candidate.get("fact", {})
                                candidate_rows.append(
                                    {
                                        "Score": candidate.get("score", ""),
                                        "Fact ID": fact.get("fact_id", ""),
                                        "Subject": fact.get("subject", ""),
                                        "Category": fact.get("category", ""),
                                        "Relation": fact.get("relation", ""),
                                        "Value": fact.get("value", ""),
                                        "Unit": fact.get("unit", ""),
                                        "Section": fact.get("source_section", ""),
                                        "Match Reasons": ", ".join(
                                            str(reason)
                                            for reason in candidate.get("match_reasons", [])
                                        ),
                                    }
                                )
                            st.dataframe(
                                candidate_rows,
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("No candidate facts were retrieved.")
        else:
            st.success("No claims were classified as contradicted in this run.")

        # Export every claim verdict and its retrieved source candidates so that
        # unexpected contradictions can be audited outside the UI.
        report_payload = {
            "case_id": selected_case,
            "summary_file": selected_summary_file.name,
            "summary_claims_checked": summary_claim_count,
            "mutation_cases_checked": mutation_case_count,
            "total_checks": len(batch_results),
            "contradicted_count": len(contradicted_items),
            "results": [
                {
                    "source_type": item.get("source_type", "Discharge summary"),
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
            "Download Full Verification Report (JSON)",
            data=json.dumps(report_payload, indent=2, ensure_ascii=False),
            file_name=f"{selected_summary_file.stem}_verification_report.json",
            mime="application/json",
            key=f"download_verification_report::{selected_summary_file.stem}",
        )
    else:
        st.info(
            "Click **Verify Entire Discharge Summary** to evaluate the selected summary's "
            "saved claims and this patient's controlled mutation claims together. Only "
            "contradicted results will be displayed below."
        )

st.divider()

