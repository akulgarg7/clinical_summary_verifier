
import argparse
import json
from pathlib import Path

from src.claim_extractor import extract_claims


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUMMARY_DIRECTORIES = [
    PROJECT_ROOT / "data" / "discharge_summaries",
    PROJECT_ROOT / "data" / "error_injected",
]

CLAIMS_DIRECTORY = PROJECT_ROOT / "data" / "claims"


def get_summary_files():
    """Find original and error-injected text summaries."""

    summary_files = []

    for directory in SUMMARY_DIRECTORIES:
        if directory.exists():
            summary_files.extend(directory.glob("*.txt"))
        else:
            print(f"Warning: Directory not found: {directory}")

    return sorted(summary_files)


def normalize_claims(result):
    """Accept the claim extractor's supported output formats."""

    if isinstance(result, dict):
        claims = result.get("claims")
    elif isinstance(result, list):
        claims = result
    else:
        claims = None

    if not isinstance(claims, list):
        raise ValueError("Claim extractor did not return a list of claims.")

    return claims


def process_summary(summary_path, force=False):
    """Extract claims and save them unless already saved."""

    output_path = CLAIMS_DIRECTORY / f"{summary_path.stem}_claims.json"

    if output_path.exists() and not force:
        print(f"SKIPPED: {summary_path.name} (claims already saved)")
        return "skipped"

    try:
        summary = summary_path.read_text(encoding="utf-8")

        print(f"\nProcessing: {summary_path.name}")

        result = extract_claims(summary)
        claims = normalize_claims(result)

        saved_data = {
            "summary_file": str(summary_path.relative_to(PROJECT_ROOT)),
            "claim_count": len(claims),
            "claims": claims,
        }

        output_path.write_text(
            json.dumps(saved_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"Claims extracted: {len(claims)}")
        print(f"Saved to: {output_path.relative_to(PROJECT_ROOT)}")

        return "success"

    except Exception as error:
        print(f"ERROR processing {summary_path.name}: {error}")
        return "failed"


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save claims from discharge summaries."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate claims even when saved files already exist.",
    )
    args = parser.parse_args()

    CLAIMS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    summary_files = get_summary_files()

    if not summary_files:
        print("No .txt discharge summaries were found.")
        return

    print(f"Found {len(summary_files)} summary files.")

    results = {"success": 0, "skipped": 0, "failed": 0}

    for summary_path in summary_files:
        status = process_summary(summary_path, force=args.force)
        results[status] += 1

    print("\n========== SUMMARY ==========")
    print(f"Successfully extracted: {results['success']}")
    print(f"Already saved and skipped: {results['skipped']}")
    print(f"Failed: {results['failed']}")
    print(f"Output directory: {CLAIMS_DIRECTORY}")


if __name__ == "__main__":
    main()
