import os
import re
import sys


def validate_evidence_index():
    print("Validating EVIDENCE_INDEX.md...")
    try:
        with open("EVIDENCE_INDEX.md", "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("FAIL: EVIDENCE_INDEX.md not found.")
        return False

    header_found = False
    rows_valid = True

    # Regex for table structure: | ID | Claim | Source | Evidence | Status | Notes |
    row_pattern = re.compile(
        r"^\|\s*(EVID-\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$"
    )

    pending_count = 0
    verified_count = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if "| ID | Claim summary" in line:
            header_found = True
            continue
        if not header_found or line.startswith("|---"):
            continue
        if not line.startswith("|"):
            continue

        match = row_pattern.match(line)
        if match:
            evidence_id = match.group(1)
            status = match.group(5).strip()
            evidence_artifact = match.group(4).strip()

            if status == "PENDING":
                print(f"WARN: {evidence_id} is PENDING.")
                pending_count += 1
            elif status == "VERIFIED":
                verified_count += 1
                if (
                    "pending" in evidence_artifact.lower()
                    or "<commit>" in evidence_artifact
                ):
                    print(
                        f"FAIL: {evidence_id} is VERIFIED but has placeholder artifact: {evidence_artifact}"
                    )
                    rows_valid = False
            elif status == "INVALIDATED":
                pass
            else:
                print(f"FAIL: {evidence_id} has invalid status '{status}'")
                rows_valid = False
        else:
            # Check for malformed rows if they look like table rows but don't match strict regex
            if line.startswith("| EVID-"):
                print(f"FAIL: Malformed row at line {i+1}: {line}")
                rows_valid = False

    print(f"Evidence Index: {verified_count} VERIFIED, {pending_count} PENDING.")
    if pending_count > 0:
        print("FAIL: Evidence Index has PENDING entries. Governance must fail-closed.")
        rows_valid = False
    return rows_valid and header_found


def validate_sot():
    print("\nValidating BIZRA_SOT.md...")
    try:
        with open("BIZRA_SOT.md", "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print("FAIL: BIZRA_SOT.md not found.")
        return False

    status_pattern = re.compile(r"Status:\s*(ACTIVE|APPROVED)", re.IGNORECASE)
    if not status_pattern.search(content):
        print("FAIL: SOT Status is not ACTIVE/APPROVED. Found Draft or other status.")
        return False

    if "Ihsan Metric Definition (Canonical)" not in content:
        print("FAIL: Ihsan Metric Definition (Canonical) section missing.")
        return False

    print("BIZRA_SOT.md looks valid.")
    return True


if __name__ == "__main__":
    # Adjust CWD if running from scripts/ subdirectory
    if not os.path.exists("EVIDENCE_INDEX.md") and os.path.exists(
        "../EVIDENCE_INDEX.md"
    ):
        os.chdir("..")

    v1 = validate_evidence_index()
    v2 = validate_sot()

    if v1 and v2:
        print("\nSUCCESS: Governance artifacts validated.")
        sys.exit(0)
    else:
        print("\nFAILURE: Validation errors found.")
        sys.exit(1)
