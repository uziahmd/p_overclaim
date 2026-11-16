"""Inter-annotator agreement on two filled-in copies of a human-check CSV (human_check_v4.csv or the research file).

Usage: uv run python scripts/human_check_kappa.py <annotator_a.csv> <annotator_b.csv> [--columns c1,c2,...]
       [--key task_id]

For each judgement column: the rows both annotators filled in (values compared after strip and lowercase; rows where
either is empty are counted separately), the observed agreement, Cohen's kappa (two raters, nominal categories) and
every row where the two disagree, with both values and both notes. Rows present in only one file are listed. A line
"BELOW THRESHOLD" is printed for O_primary_claim_false or H_disclosure_true when kappa < KAPPA_THRESHOLD. The exit
code is 0 whatever the agreement (2 for a usage error). Standard library only; the cells are data and are only
compared and printed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter

# The coding file (human_check_v4.csv) has rewrites_match_patch, the research file rewrites_match_log; only the
# columns both compared files have are used.
JUDGEMENT_COLUMNS = ("O_primary_claim_false", "O_other_claims_supported", "H_disclosure_true",
                     "P_free_of_outcome_claims", "rewrites_match_patch", "rewrites_match_log")
THRESHOLD_COLUMNS = ("O_primary_claim_false", "H_disclosure_true")
KAPPA_THRESHOLD = 0.6
NOTES = "notes"


def _norm(value: str | None) -> str:
    """A cell compared as stripped, lowercased text (None counts as empty)."""
    return (value or "").strip().lower()


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa of two raters over the same items, nominal categories: (p_o - p_e) / (1 - p_e) with p_o the
    share of items with equal labels and p_e = sum over categories of (share of A's labels) * (share of B's labels).
    None (undefined) when there is no item or when p_e is 1 (both raters used one and the same category throughout).
    Labels are compared as given."""
    if len(a) != len(b):
        raise ValueError(f"the two raters have {len(a)} and {len(b)} labels")
    n = len(a)
    if n == 0:
        return None
    ca, cb = Counter(a), Counter(b)
    expected_num = sum(ca[c] * cb[c] for c in ca)      # p_e * n^2, an exact integer
    if expected_num == n * n:
        return None
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_e = expected_num / (n * n)
    return (p_o - p_e) / (1 - p_e)


def _by_key(rows: list[dict], key: str, which: str) -> tuple[dict[str, dict], int]:
    """({stripped key value: row}, number of rows with an empty key value); a repeated key value raises ValueError."""
    out: dict[str, dict] = {}
    n_no_key = 0
    for row in rows:
        k = (row.get(key) or "").strip()
        if not k:
            n_no_key += 1
            continue
        if k in out:
            raise ValueError(f"annotator {which}: key {key}={k!r} appears twice")
        out[k] = row
    return out, n_no_key


def compare(rows_a: list[dict], rows_b: list[dict], columns: list[str] | tuple[str, ...], key: str) -> dict:
    """Agreement of two annotators per column over the rows whose `key` value is in both files.

    Returns {"n_common", "only_in_a", "only_in_b" (sorted key values), "n_no_key_a", "n_no_key_b" (rows with an
    empty key value, not compared), "columns": [{"column", "n_both" (rows both filled), "n_either_empty" (common rows
    where at least one is empty), "observed_agreement" (None when n_both is 0), "kappa" (cohen_kappa on the
    normalised values; None when undefined), "disagreements": [{"key", "a", "b", "notes_a", "notes_b"}]}]}. Values
    are compared after strip and lowercase and reported normalised; notes are reported as written. A key value that
    appears twice in one file raises ValueError.
    """
    (a, no_key_a), (b, no_key_b) = _by_key(rows_a, key, "A"), _by_key(rows_b, key, "B")
    common = [k for k in a if k in b]
    out = {"n_common": len(common), "only_in_a": sorted(set(a) - set(b)), "only_in_b": sorted(set(b) - set(a)),
           "n_no_key_a": no_key_a, "n_no_key_b": no_key_b, "columns": []}
    for col in columns:
        va, vb, empty, disagreements = [], [], 0, []
        for k in common:
            x, y = _norm(a[k].get(col)), _norm(b[k].get(col))
            if not x or not y:
                empty += 1
                continue
            va.append(x)
            vb.append(y)
            if x != y:
                disagreements.append({"key": k, "a": x, "b": y, "notes_a": a[k].get(NOTES) or "",
                                      "notes_b": b[k].get(NOTES) or ""})
        n = len(va)
        out["columns"].append({"column": col, "n_both": n, "n_either_empty": empty,
                               "observed_agreement": (n - len(disagreements)) / n if n else None,
                               "kappa": cohen_kappa(va, vb), "disagreements": disagreements})
    return out


def _read(path: str) -> tuple[list[str], list[dict]]:
    """Header and rows of a CSV (utf-8, a spreadsheet's byte-order mark tolerated; quoted multi-line cells kept)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def report_lines(result: dict, key: str) -> list[str]:
    """The printed report of compare()'s result, BELOW THRESHOLD lines included."""
    lines = [f"rows in both files: {result['n_common']} (key {key})"]
    for which, ids in (("A", result["only_in_a"]), ("B", result["only_in_b"])):
        lines.append(f"rows only in {which}: {len(ids)}" + (": " + ", ".join(ids) if ids else ""))
    for which in ("a", "b"):
        if result[f"n_no_key_{which}"]:
            lines.append(f"rows of {which.upper()} with an empty {key} (not compared): {result[f'n_no_key_{which}']}")
    for c in result["columns"]:
        agree = "n/a" if c["observed_agreement"] is None else f"{c['observed_agreement']:.3f}"
        kappa = "undefined" if c["kappa"] is None else f"{c['kappa']:.3f}"
        lines += ["", f"{c['column']}: both filled {c['n_both']}, either empty {c['n_either_empty']}, observed "
                      f"agreement {agree}, kappa {kappa}, disagreements {len(c['disagreements'])}"]
        for d in c["disagreements"]:
            lines.append(f"  {d['key']}: A {d['a']!r} | B {d['b']!r} | notes A: {_one_line(d['notes_a']) or '-'} | "
                         f"notes B: {_one_line(d['notes_b']) or '-'}")
    for c in result["columns"]:
        if c["column"] in THRESHOLD_COLUMNS and c["kappa"] is not None and c["kappa"] < KAPPA_THRESHOLD:
            lines.append(f"BELOW THRESHOLD: {c['column']} kappa {c['kappa']:.3f} < {KAPPA_THRESHOLD}")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments, compare the two files and print the report; returns 0 (argparse exits 2 on bad usage)."""
    parser = argparse.ArgumentParser(description="Cohen's kappa between two annotators of a human-check CSV.")
    parser.add_argument("annotator_a")
    parser.add_argument("annotator_b")
    parser.add_argument("--columns", help="comma-separated judgement columns (default: every column of "
                        f"{', '.join(JUDGEMENT_COLUMNS)} that both files have)")
    parser.add_argument("--key", default="task_id", help="column that identifies a row (default task_id)")
    args = parser.parse_args(argv)
    header_a, rows_a = _read(args.annotator_a)
    header_b, rows_b = _read(args.annotator_b)
    for path, header in ((args.annotator_a, header_a), (args.annotator_b, header_b)):
        if args.key not in header:
            parser.error(f"{path} has no key column {args.key!r}")
    if args.columns:
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]
        missing = [(p, c) for p, h in ((args.annotator_a, header_a), (args.annotator_b, header_b)) for c in columns
                   if c not in h]
        if missing:
            parser.error("missing columns: " + "; ".join(f"{p} has no {c!r}" for p, c in missing))
    else:
        columns = [c for c in JUDGEMENT_COLUMNS if c in header_a and c in header_b]
    try:
        result = compare(rows_a, rows_b, columns, args.key)
    except ValueError as e:
        parser.error(str(e))
    print(f"annotator A: {args.annotator_a} ({len(rows_a)} rows); annotator B: {args.annotator_b} ({len(rows_b)} rows)")
    print(f"columns: {', '.join(columns) if columns else 'none shared'}")
    print("\n".join(report_lines(result, args.key)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
