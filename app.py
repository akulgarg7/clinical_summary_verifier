import json
from pathlib import Path

import streamlit as st

from src.claim_extractor import extract_claims
from src.verification_engine import verify_claim


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

SUMMARY_DIR = (
    BASE_DIR
    / "data"
    / "discharge_summaries"
)

ERROR_DIR = (
    BASE_DIR
    / "data"
    / "error_injected"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Clinical Summary Verification System",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_ground_truth():

    with open(
        GROUND_TRUTH_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        records = json.load(f)

    if not isinstance(records, list):

        raise ValueError(
            "Ground-truth JSON must contain a list of records."
        )

    return records


def load_text(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


records = load_ground_truth()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_case_id(record):

    return record[
        "case_metadata"
    ][
        "case_id"
    ]


def get_original_summary(case_id):

    possible_files = [

        SUMMARY_DIR / f"{case_id}.txt",

        SUMMARY_DIR / f"{case_id}_summary.txt",

        SUMMARY_DIR / f"{case_id}_discharge_summary.txt",
    ]

    for file_path in possible_files:

        if file_path.exists():

            return file_path


    # Fallback
    matches = sorted(
        SUMMARY_DIR.glob(
            f"{case_id}*.txt"
        )
    )

    if matches:

        return matches[0]

    return None


def get_error_summaries(case_id):

    return sorted(
        ERROR_DIR.glob(
            f"{case_id}_*.txt"
        )
    )


def get_summary_options(case_id):

    options = []

    original = get_original_summary(
        case_id
    )

    if original:

        options.append(
            (
                "Original discharge summary",
                original
            )
        )


    for error_file in get_error_summaries(
        case_id
    ):

        options.append(
            (
                error_file.stem,
                error_file
            )
        )

    return options


# ============================================================
# CONTROLLED TEST INFORMATION
# ============================================================

# IMPORTANT:
# This information is ONLY used by the current Phase A
# demonstration interface.
#
# It will be removed/replaced when independent evidence
# retrieval is implemented.

CONTROLLED_TESTS = {

    "SYN-DS-0006_E001": {

        "fact_id": "SYN-DS-0006-F139",

        "error_type": "Medication dose",

        "wrong_value": "20 mg",

        "expected_value": "10 mg",

        "description":
            "Rivaroxaban dose is incorrectly written as 20 mg "
            "instead of 10 mg."
    },


    "SYN-DS-0006_E002": {

        "fact_id": "SYN-DS-0006-F141",

        "error_type": "Medication frequency",

        "wrong_value": "twice daily",

        "expected_value": "once daily",

        "description":
            "Rivaroxaban frequency is incorrectly written as "
            "twice daily instead of once daily."
    },


    "SYN-DS-0003_E003": {

        "fact_id": "SYN-DS-0003-F055",

        "error_type": "Laboratory value",

        "wrong_value": "2.9 mg/dL",

        "expected_value": "1.9 mg/dL",

        "description":
            "Admission creatinine is incorrectly written as "
            "2.9 mg/dL instead of 1.9 mg/dL."
    }
}


# ============================================================
# HEADER
# ============================================================

st.title(
    "🏥 Clinical Summary Verification System"
)

st.markdown(
    """
### Automated factual verification of AI-generated discharge summaries

The system takes a structured clinical record, examines an
AI-generated discharge summary, extracts atomic factual claims,
and evaluates those claims against the available source information.
"""
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Case Selection"
)


case_ids = [
    get_case_id(record)
    for record in records
]


selected_case = st.sidebar.selectbox(
    "Patient Case",
    case_ids
)


selected_record = next(
    record
    for record in records
    if get_case_id(record)
    == selected_case
)


# ============================================================
# SUMMARY SELECTION
# ============================================================

summary_options = get_summary_options(
    selected_case
)


if not summary_options:

    st.sidebar.error(
        "No discharge summaries found for this case."
    )

    st.stop()


summary_labels = [
    label
    for label, _ in summary_options
]


selected_summary_label = st.sidebar.selectbox(
    "Discharge Summary",
    summary_labels
)


selected_summary_file = next(
    file_path
    for label, file_path in summary_options
    if label == selected_summary_label
)


# ============================================================
# RESET CLAIMS WHEN SUMMARY CHANGES
# ============================================================

current_summary_key = str(
    selected_summary_file
)


if (
    "claims_summary_file"
    not in st.session_state
    or st.session_state[
        "claims_summary_file"
    ] != current_summary_key
):

    st.session_state.pop(
        "claims",
        None
    )

    st.session_state[
        "claims_summary_file"
    ] = current_summary_key


# ============================================================
# 1. CASE INFORMATION
# ============================================================

st.header(
    "1. Patient / Ground-Truth Record"
)


metadata = selected_record.get(
    "case_metadata",
    {}
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Case ID",
        selected_case
    )


with col2:

    st.metric(
        "Complexity",
        metadata.get(
            "complexity",
            "N/A"
        )
    )


with col3:

    st.metric(
        "Specialty",
        metadata.get(
            "primary_specialty",
            "N/A"
        )
    )


with col4:

    st.metric(
        "Admission Type",
        metadata.get(
            "admission_type",
            "N/A"
        )
    )


with st.expander(
    "View Ground-Truth Patient Record"
):

    st.json(
        selected_record
    )


st.divider()


# ============================================================
# 2. DISCHARGE SUMMARY
# ============================================================

st.header(
    "2. AI-Generated Discharge Summary"
)


if selected_summary_label == (
    "Original discharge summary"
):

    st.info(
        "Original LLM-generated discharge summary"
    )

else:

    st.warning(
        f"Controlled error-injected summary: "
        f"{selected_summary_label}"
    )


summary = load_text(
    selected_summary_file
)


st.text_area(
    "Discharge Summary",
    summary,
    height=400,
    disabled=True
)


st.divider()


# ============================================================
# 3. ATOMIC CLAIM EXTRACTION
# ============================================================

st.header(
    "3. Atomic Claim Extraction"
)


if st.button(
    "Extract Claims",
    type="primary"
):

    with st.spinner(
        "Extracting factual claims..."
    ):

        try:

            claims = extract_claims(
                summary
            )

            st.session_state[
                "claims"
            ] = claims

            st.session_state[
                "claims_summary_file"
            ] = current_summary_key

        except Exception as e:

            st.error(
                f"Claim extraction failed: {e}"
            )

            st.stop()


claims = st.session_state.get(
    "claims"
)


if claims is None:

    st.info(
        "Click **Extract Claims** to extract "
        "atomic factual claims from the summary."
    )

else:

    st.success(
        f"{len(claims)} factual claims extracted."
    )


    # --------------------------------------------------------
    # CLAIM TABLE
    # --------------------------------------------------------

    claim_rows = []

    for claim in claims:

        claim_rows.append(
            {
                "Claim ID":
                    claim.get(
                        "claim_id",
                        ""
                    ),

                "Category":
                    claim.get(
                        "category",
                        ""
                    ),

                "Subject":
                    claim.get(
                        "subject",
                        ""
                    ),

                "Relation":
                    claim.get(
                        "relation",
                        ""
                    ),

                "Value":
                    claim.get(
                        "value",
                        ""
                    ),

                "Unit":
                    claim.get(
                        "unit",
                        ""
                    ),

                "Route":
                    claim.get(
                        "route",
                        ""
                    ),

                "Frequency":
                    claim.get(
                        "frequency",
                        ""
                    ),

                "Date":
                    claim.get(
                        "date_time",
                        ""
                    )
            }
        )


    st.dataframe(
        claim_rows,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ============================================================
# 4. CLAIM DETAILS
# ============================================================

st.header(
    "4. Selected Claim Details"
)


if claims is None:

    st.info(
        "Extract claims first to inspect "
        "individual claims."
    )

else:

    claim_ids = [
        claim.get(
            "claim_id",
            ""
        )
        for claim in claims
    ]


    selected_claim_id = st.selectbox(
        "Select a claim",
        claim_ids,
        key=f"claim_selector_{selected_case}_{selected_summary_label}"
    )


    selected_claim = next(
        claim
        for claim in claims
        if claim.get(
            "claim_id"
        )
        == selected_claim_id
    )


    # --------------------------------------------------------
    # CLAIM INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Extracted Claim"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.write(
            f"**Claim ID:** "
            f"{selected_claim.get('claim_id', '')}"
        )

        st.write(
            f"**Category:** "
            f"{selected_claim.get('category', '')}"
        )

        st.write(
            f"**Subject:** "
            f"{selected_claim.get('subject', '')}"
        )


    with col2:

        st.write(
            f"**Relation:** "
            f"{selected_claim.get('relation', '')}"
        )

        st.write(
            f"**Value:** "
            f"{selected_claim.get('value', '')}"
        )

        st.write(
            f"**Unit:** "
            f"{selected_claim.get('unit', '')}"
        )


    with col3:

        st.write(
            f"**Route:** "
            f"{selected_claim.get('route', '')}"
        )

        st.write(
            f"**Frequency:** "
            f"{selected_claim.get('frequency', '')}"
        )

        st.write(
            f"**Date:** "
            f"{selected_claim.get('date_time', '')}"
        )


    # --------------------------------------------------------
    # SOURCE TEXT
    # --------------------------------------------------------

    st.subheader(
        "Exact Source Text From Summary"
    )


    st.info(
        selected_claim.get(
            "source_text",
            "No source text available."
        )
    )


    # --------------------------------------------------------
    # COMPLETE CLAIM JSON
    # --------------------------------------------------------

    with st.expander(
        "View Complete Claim JSON"
    ):

        st.json(
            selected_claim
        )


st.divider()


# ============================================================
# 5. VERIFICATION
# ============================================================

st.header(
    "5. Verification Result"
)


st.warning(
    """
### Current prototype limitation

The current controlled experiment uses the known target
ground-truth fact for these injected-error cases.

The next research stage will remove this shortcut and make
the system independently retrieve the relevant evidence from
the patient's complete ground-truth record.
"""
)


# ============================================================
# CURRENT CONTROLLED VERIFICATION
# ============================================================

summary_name = selected_summary_file.stem


controlled_test = CONTROLLED_TESTS.get(
    summary_name
)


if claims is None:

    st.info(
        "Extract claims first to perform the "
        "current verification demonstration."
    )


elif controlled_test is None:

    st.info(
        """
This summary is not currently configured as a
controlled verification experiment.

You can still inspect the extracted claims above.
"""
    )


else:

    facts = selected_record.get(
        "ground_truth_facts",
        []
    )


    target_fact_id = controlled_test[
        "fact_id"
    ]


    target_fact = next(
        (
            fact
            for fact in facts
            if fact.get("fact_id")
            == target_fact_id
        ),
        None
    )


    if target_fact is None:

        st.error(
            "Configured ground-truth fact was not found."
        )

    else:

        # ----------------------------------------------------
        # FIND THE CLAIM CONTAINING THE INJECTED ERROR
        # ----------------------------------------------------

        wrong_value = controlled_test[
            "wrong_value"
        ]


        candidate_claim = None


        for claim in claims:

            fields_to_check = [

                claim.get(
                    "value",
                    ""
                ),

                claim.get(
                    "unit",
                    ""
                ),

                claim.get(
                    "frequency",
                    ""
                ),

                claim.get(
                    "route",
                    ""
                ),

                claim.get(
                    "duration",
                    ""
                ),

                claim.get(
                    "source_text",
                    ""
                )
            ]


            combined_text = " ".join(
                str(field).lower()
                for field in fields_to_check
                if field
            )


            if wrong_value.lower() in combined_text:

                candidate_claim = claim

                break


        if candidate_claim is None:

            st.info(
                "The injected error claim was not found "
                "among the extracted claims."
            )

        else:

            # ------------------------------------------------
            # RUN CURRENT VERIFICATION ENGINE
            # ------------------------------------------------

            result = verify_claim(
                candidate_claim,
                target_fact
            )


            verdict = result.get(
                "verdict",
                "UNKNOWN"
            )


            # ------------------------------------------------
            # VERDICT
            # ------------------------------------------------

            if verdict == "SUPPORTED":

                st.success(
                    "🟢 SUPPORTED"
                )

            elif verdict == "CONTRADICTED":

                st.error(
                    "🔴 CONTRADICTED"
                )

            else:

                st.warning(
                    f"🟡 {verdict}"
                )


            # ------------------------------------------------
            # CLAIM VS GROUND TRUTH
            # ------------------------------------------------

            st.subheader(
                "Claim vs Ground Truth"
            )


            comparison_col1, comparison_col2 = (
                st.columns(2)
            )


            with comparison_col1:

                st.markdown(
                    "### 🔴 What the summary says"
                )


                st.write(
                    f"**Claim:** "
                    f"{candidate_claim.get('source_text', '')}"
                )


                st.write(
                    f"**Value:** "
                    f"{candidate_claim.get('value', '')}"
                )


                if candidate_claim.get("unit"):

                    st.write(
                        f"**Unit:** "
                        f"{candidate_claim.get('unit')}"
                    )


                if candidate_claim.get("frequency"):

                    st.write(
                        f"**Frequency:** "
                        f"{candidate_claim.get('frequency')}"
                    )


            with comparison_col2:

                st.markdown(
                    "### 🟢 What the ground truth says"
                )


                st.write(
                    f"**Fact:** "
                    f"{target_fact.get('subject', '')}"
                )


                st.write(
                    f"**Value:** "
                    f"{target_fact.get('value', '')}"
                )


                if target_fact.get("unit"):

                    st.write(
                        f"**Unit:** "
                        f"{target_fact.get('unit')}"
                    )


                st.write(
                    f"**Relation:** "
                    f"{target_fact.get('relation', '')}"
                )


                if target_fact.get("date_time"):

                    st.write(
                        f"**Date:** "
                        f"{target_fact.get('date_time')}"
                    )


            # ------------------------------------------------
            # EXPLICIT CONTRADICTION EXPLANATION
            # ------------------------------------------------

            if verdict == "CONTRADICTED":

                st.subheader(
                    "⚠️ Detected Contradiction"
                )


                error_type = controlled_test[
                    "error_type"
                ]


                wrong_value = controlled_test[
                    "wrong_value"
                ]


                expected_value = controlled_test[
                    "expected_value"
                ]


                st.error(
                    f"**{error_type}:** "
                    f"The summary states **{wrong_value}**, "
                    f"but the ground truth specifies "
                    f"**{expected_value}**."
                )


                st.write(
                    controlled_test[
                        "description"
                    ]
                )


            # ------------------------------------------------
            # VERIFICATION METRICS
            # ------------------------------------------------

            st.subheader(
                "Verification Details"
            )


            col1, col2, col3, col4 = (
                st.columns(4)
            )


            with col1:

                st.metric(
                    "Verdict",
                    result.get(
                        "verdict",
                        "N/A"
                    )
                )


            with col2:

                confidence = result.get(
                    "confidence",
                    0
                )


                st.metric(
                    "Confidence",
                    f"{confidence * 100:.0f}%"
                )


            with col3:

                st.metric(
                    "Severity",
                    result.get(
                        "severity",
                        "N/A"
                    )
                )


            with col4:

                st.metric(
                    "Error Type",
                    result.get(
                        "error_type",
                        "None"
                    )
                )


            # ------------------------------------------------
            # EVIDENCE
            # ------------------------------------------------

            st.subheader(
                "Ground-Truth Evidence"
            )


            evidence_col1, evidence_col2 = (
                st.columns(2)
            )


            with evidence_col1:

                st.write(
                    f"**Fact ID:** "
                    f"{target_fact.get('fact_id', '')}"
                )

                st.write(
                    f"**Category:** "
                    f"{target_fact.get('category', '')}"
                )

                st.write(
                    f"**Subject:** "
                    f"{target_fact.get('subject', '')}"
                )

                st.write(
                    f"**Relation:** "
                    f"{target_fact.get('relation', '')}"
                )


            with evidence_col2:

                st.write(
                    f"**Value:** "
                    f"{target_fact.get('value', '')}"
                )

                st.write(
                    f"**Unit:** "
                    f"{target_fact.get('unit', '')}"
                )

                st.write(
                    f"**Date:** "
                    f"{target_fact.get('date_time', '')}"
                )

                st.write(
                    f"**Severity if incorrect:** "
                    f"{target_fact.get('severity_if_incorrect', '')}"
                )


            with st.expander(
                "View Complete Ground-Truth Fact"
            ):

                st.json(
                    target_fact
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Phase A prototype | Synthetic clinical data only | "
    "Independent evidence retrieval is the next development stage"
)