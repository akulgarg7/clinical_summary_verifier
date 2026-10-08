from src.claim_extractor import extract_claims


test_summary = """
The patient was admitted with pneumonia and fever.
Chest X-ray showed bilateral lung infiltrates.
The patient was treated with amoxicillin 500 mg orally three times daily.
After treatment, the fever resolved.
The patient was discharged home on 2026-09-15.
"""


claims = extract_claims(test_summary)

print("\nExtracted Claims:\n")

for claim in claims:
    print(claim)