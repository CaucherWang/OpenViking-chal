#!/usr/bin/env bash
set -euo pipefail

results_file="${1:-benchmark/aml/data/results/personamem_v2_search_mcq_32k_full.csv}"

if [[ ! -f "${results_file}" ]]; then
  printf 'Results file not found: %s\n' "${results_file}" >&2
  exit 1
fi

python3 - "${results_file}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(field_limit)
        break
    except OverflowError:
        field_limit //= 10

complete_rows = 0
incomplete_rows = 0
errors = 0
scored = 0
correct = 0

with path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames or []
    correctness_columns = [
        name
        for name in fieldnames
        if name.startswith("is_correct_mcq_")
    ]
    if len(correctness_columns) != 1:
        raise SystemExit(
            "Expected exactly one is_correct_mcq_* column, found: "
            + ", ".join(correctness_columns)
        )
    correctness_column = correctness_columns[0]

    for row in reader:
        if None in row or any(value is None for value in row.values()):
            incomplete_rows += 1
            continue
        complete_rows += 1
        if row.get("evaluation_error", "").strip():
            errors += 1
        value = row.get(correctness_column, "").strip().lower()
        if value not in {"true", "false"}:
            continue
        scored += 1
        correct += value == "true"

accuracy = correct / scored if scored else 0.0
print(f"file: {path}")
print(f"correctness_column: {correctness_column}")
print(f"complete_rows: {complete_rows}")
print(f"incomplete_rows_ignored: {incomplete_rows}")
print(f"errors: {errors}")
print(f"scored: {scored}")
print(f"correct: {correct}")
print(f"incorrect: {scored - correct}")
print(f"accuracy: {accuracy:.6f} ({accuracy:.2%})")
PY
