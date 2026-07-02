"""
For each sentence group in wilcox2022_embed4_subject_gap_with_modals.csv, one row has
already been updated with a modal+verb (e.g. "will throw"). This script propagates that
modal+verb to all other rows in the same sentence group.

Prints an error for any group where no modal verb is found.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

MODALS = {"can", "could", "should", "would", "may", "might", "must", "will", "shall"}

INPUT = Path(__file__).parent.parent / "data/stimuli/wilcox2022_embed4_subject_gap_with_modals.csv"


def has_modal(verb: str) -> bool:
    return bool(verb) and verb.split()[0].lower() in MODALS


def main():
    with open(INPUT, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Group rows by sentence_group, preserving original order
    group_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        group_indices[row["sentence_group"]].append(i)

    errors = []
    updates = 0

    for group, indices in group_indices.items():
        group_rows = [rows[i] for i in indices]
        modal_verbs = [r["verb"] for r in group_rows if has_modal(r["verb"])]

        if not modal_verbs:
            errors.append(f"ERROR: sentence_group {group} has no modal verb — skipping")
            continue

        # Use the first modal verb found (there should only be one per group)
        modal_verb = modal_verbs[0]
        if len(set(modal_verbs)) > 1:
            print(f"WARNING: sentence_group {group} has multiple distinct modal verbs: {set(modal_verbs)}. Using: {modal_verb!r}")

        for i in indices:
            if rows[i]["verb"] != modal_verb:
                rows[i]["verb"] = modal_verb
                updates += 1

    for err in errors:
        print(err, file=sys.stderr)

    with open(INPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. {updates} rows updated across {len(group_indices) - len(errors)} groups.")
    if errors:
        print(f"{len(errors)} groups had errors and were not updated.", file=sys.stderr)


if __name__ == "__main__":
    main()
