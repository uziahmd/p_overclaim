"""Write the paper's tables from the committed records (paper/tables/*.tex and *.csv).

Main text: t1_questions (the four questions, what we found, the verdict of the planned tests). Appendix B, in this
order: a_tests (the eight planned tests), a_premium (the premium without and with the transcript), a_graders (model
ids, settings, dates), a_placebo (the placebo comparisons), a_selection (best-of-N selection; a CSV repeats it with
the second labeller), a_verification, a_side_by_side, a_claims (claims added and removed), a_sensitivity (the eight
tests on smaller samples), a_construction (how the sets were built), a_not_run (what is partial, failed or not run).

Draft 7 (the reanalysis of 2026-11-11, framing "B+table"): t2_comparisons (main text, Table 2: four comparisons
without the transcript), a_kinds (Table B10 since draft 8: the premium by kind of missing check), a_instruction (Table B12:
selection by instruction), one more column in a_claims (the share of the premium) and two in a_sensitivity. These
read the CSVs of reanalysis/tables, which the scripts of reanalysis/scripts wrote from the same records. A result
that was added after the results were known carries the mark AFTER (a section sign); the captions explain it. An
estimate whose interval includes zero carries the mark ZERO in those tables.

Draft 10 (the control run of the sprint, results/sprint_control): Table 1's RQ1 cell gives the control run's count;
Table 2 has five comparisons (the claim alone against the honest message and against the placebo's neutral sentence
come from the control run and carry the mark CONTROL; the comparison against a hedge left the table and stays in
a_claims); a_control and a_control_split (Tables B13 and B14) print the control run; a_reasons is the lower part of
Table B9 (the keyword count of the judges' stated reasons); a_not_run gains the control run's two rows. The control
run's values are read from results/sprint_control/analysis.json and reasons.csv, which sprint_control/analyze.py
wrote; nothing of that folder is imported.

Every table is a `tabular` body only (caption and label live in the section files). Graders always appear in one
order (paper_data.LLM_JUDGES, then REWARD_MODELS). Shares are printed as percentages and their differences in
percentage points. Run from the repository root:
    uv run --no-sync python scripts/make_paper_derived.py && uv run --no-sync python scripts/make_paper_tables.py
Nothing under results/ is written; the inputs are those of scripts/paper_data.py.
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("paper_data", _HERE / "paper_data.py")
D = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(D)

OUT = D.PAPER / "tables"
MID = ["\\midrule"]
WORDS = {2: "Two", 3: "Three", 4: "Four"}
TEST_WORDS = {8: "eight"}      # the number of planned tests, in words (Table 1)
COND = {"hidden": "without", "shown": "with"}
RQ = {"H1": "RQ1", "H2": "RQ2", "H3": "RQ3", "H4": "RQ4", "H4(i)": "RQ4, premium", "H4(ii)": "RQ4, selection"}


def _minus(cell: str) -> str:
    """A hyphen before a digit becomes a real minus sign in the .tex file."""
    return re.sub(r"(?<![\w$\\.-])-(?=\d)", "$-$", cell)


def write(name: str, header: list[str], rows: list[list[str]], colspec: str, *, size: str = "\\scriptsize",
          stretch: float | None = None, colsep: float = 4, head: list[str] | None = None) -> None:
    """`head`: the header lines of the .tex file as raw LaTeX, for a header with a cell that spans columns; the CSV
    keeps `header`."""
    OUT.mkdir(parents=True, exist_ok=True)
    setup = f"\\setlength{{\\tabcolsep}}{{{colsep:g}pt}}" + (f"\\renewcommand{{\\arraystretch}}{{{stretch}}}" if stretch else "")
    colspec = colspec.replace("p{", ">{\\raggedright\\arraybackslash}p{")     # text columns are never justified
    parts = [h.split("|") for h in header]           # "a|b" is a header of two lines, a above b
    depth = max(len(x) for x in parts)
    head = head or [" & ".join(([""] * (depth - len(x)) + x)[i] for x in parts) + " \\\\" for i in range(depth)]
    lines = [f"{{{size}{setup}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *head, "\\midrule"]
    for r in rows:
        lines.append("\\midrule" if r == MID else " & ".join(_minus(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([_plain(h.replace("|", " ")) for h in header])
        for r in rows:
            if r != MID:
                w.writerow([_plain(c) for c in r])


def _plain(cell: str) -> str:
    """A LaTeX cell as plain text for the CSV."""
    cell = re.sub(r"\\(?:textbf|textit|emph|texttt|path)\{([^}]*)\}", r"\1", cell)
    cell = re.sub(r"\\multicolumn\{\d+\}\{(?:[^{}]|\{[^{}]*\})*\}\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", cell)   # also "@{}l"
    cell = re.sub(r"\s*\((?:Section|Table)~\\ref\{[^}]*\}\)", "", cell)   # a pointer has no number outside LaTeX
    for a, b in (("\\textsuperscript{a}", " [a]"), ("\\allowbreak ", ""), ("\\%", "%"), ("$-$", "-"), ("$<$", "<"), ("$\\dagger$", "+"), ("\\_", "_"), ("~", " "), ("--", "-"),
                 ("\\quad ", ""), ("\\textsuperscript{\\S}", " [after the fact]"),
                 ("\\textsuperscript{\\P}", " [control run]"),
                 ("\\rlap{\\textsuperscript{$\\circ$}}", " [interval includes zero]"),
                 ("\\textsuperscript{$\\circ$}", " [interval includes zero]"), ("$", ""), ("\\&", "&")):
        cell = cell.replace(a, b)
    return cell


def name(g: str) -> str:
    return D.NAMES.get(g, g)


def pp(x: float) -> str:
    """A share difference in whole percentage points."""
    return str(D.half_up(x * 100))


def pp_est(x: float, c: list) -> str:
    return f"{pp(x)} [{pp(c[0])}, {pp(c[1])}]"


def pp1_est(x: float, c: list) -> str:
    """A share difference in percentage points to one decimal, with its interval."""
    return f"{x * 100:.1f} [{c[0] * 100:.1f}, {c[1] * 100:.1f}]"


def pct_est(x: float, c: list) -> str:
    return f"{D.pct(x)} [{D.pct(c[0])}, {D.pct(c[1])}]"


REAN = D.REPO / "reanalysis" / "tables"
AFTER = "\\textsuperscript{\\S}"                 # after the fact: an analysis added after the results were known
ZERO = "\\textsuperscript{$\\circ$}"            # the interval includes zero
ZERO_HANG = "\\rlap{" + ZERO + "}"               # the same mark set in the gap after its column (Table 2: six columns)


def rean(stem: str) -> list[dict]:
    """A table of the reanalysis (reanalysis/tables/<stem>.csv) as rows."""
    with (REAN / f"{stem}.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def r_est(r: dict, col: str, lo: str, hi: str, digits: int = 2) -> str:
    """An estimate of a reanalysis row with its interval; the mark ZERO when the interval includes zero."""
    x, a, b = float(r[col]), float(r[lo]), float(r[hi])
    return D.est(x, [a, b], digits) + (ZERO if a <= 0 <= b else "")


CTRL = D.REPO / "results" / "sprint_control"     # the control run of the sprint (written by sprint_control/analyze.py)
CONTROL = "\\textsuperscript{\\P}"               # the control run: a sign of its own (AFTER marks an analysis after the fact)
# What the text says of the control run in words (abstract, Table 1, Section 4.1, the second limitation, the
# Conclusion, Appendix A): twelve graders; the claim alone scores above the honest message for all twelve, detectably
# for eleven (not for Opus 5.5); that difference is smaller than the premium for eleven, detectably for nine; of the
# four tests three are supported. control() stops when the run's files no longer give these.
CONTROL_SAYS = {"graders": 12, "claim alone above the honest message": 12, "detectably": 11,
                "not detectably": ["Opus 5.5"], "smaller than the premium": 11, "detectably smaller": 9,
                "tests": 4, "tests supported": 3, "tests not supported": ["Opus 5.5"],
                "no score from": ["gemini-3.8-flash", "gemini-4-argon"]}


def control() -> dict:
    """The analysis of the control run: its rows by the grader's name ("rows") and the whole file ("doc")."""
    doc = D.load_json(CTRL / "analysis.json")
    rows = {r["name"]: r for r in doc["rows"]}
    ch = {g: r["comparisons"]["C - H"] for g, r in rows.items()}
    oc = {g: r["comparisons"]["O - C"] for g, r in rows.items()}
    got = {"graders": len(rows), "claim alone above the honest message": sum(v["mean"] > 0 for v in ch.values()),
           "detectably": sum(v["lo"] > 0 for v in ch.values()),
           "not detectably": [g for g, v in ch.items() if not v["lo"] > 0],
           "smaller than the premium": sum(v["mean"] > 0 for v in oc.values()),
           "detectably smaller": sum(v["lo"] > 0 for v in oc.values()),
           "tests": len(doc["tests"]), "tests supported": sum(t["verdict"] == "supported (Holm)" for t in doc["tests"]),
           "tests not supported": [t["name"] for t in doc["tests"] if t["verdict"] != "supported (Holm)"],
           "no score from": sorted(doc["graders_without_scores"])}
    if got != CONTROL_SAYS:
        raise SystemExit(f"the control run's files no longer give what the paper says in words: {got}; rewrite the "
                         "sentences on the control run and CONTROL_SAYS")
    return {"rows": rows, "doc": doc}


def c_est(v: dict, digits: int = 2) -> str:
    """An estimate of the control run with its interval; the mark ZERO when the interval includes zero."""
    return D.est(v["mean"], [v["lo"], v["hi"]], digits) + (ZERO if v["lo"] <= 0 <= v["hi"] else "")


def c_val(v: dict, digits: int = 2) -> str:
    """The same without the interval."""
    return D.fmt(v["mean"], digits) + (ZERO if v["lo"] <= 0 <= v["hi"] else "")


def of(n: int, total: int) -> str:
    return f"all {total}" if n == total else f"{n} of {total}"


# ---------------------------------------------------------------- Table 1: the questions and the answers


def t1_questions(s: dict, d: dict) -> None:
    c = d["counts"]
    holm = {r["member"]: r for r in D.holm_rows(s)}
    sup = lambda m: holm[m]["verdict_holm"].startswith("supported")  # noqa: E731
    h1 = [m for m in holm if m.startswith("H1")]
    if not all(sup(m) for m in h1):
        raise SystemExit("Table 1 says RQ1 is supported for every judge tested; the records disagree")
    if not (sup("H2 / claude-sonnet-5-5") and sup("H3 / claude-sonnet-4-6") and not sup("H3 / claude-sonnet-5-5")
            and not sup("H4 / claude-sonnet-5-5")):
        raise SystemExit("Table 1's verdict column no longer matches the records; rewrite t1_questions")
    e7 = {r["grader"]: r["diffs"]["verify_first_minus_plain"] for r in s["e7"]}
    h4i = {r["grader"]: r for r in D.hypothesis_rows(s, "H4(i)")}
    if not (h4i["claude-sonnet-5-5"]["ci"][1] < 0 and h4i["claude-opus-5-5"]["ci"][1] < 0
            and h4i["gemini-3.8-flash"]["ci"][0] <= 0 <= h4i["gemini-3.8-flash"]["ci"][1]
            and e7["gemini-3.8-flash"]["ci"][0] <= 0 <= e7["gemini-3.8-flash"]["ci"][1]):
        raise SystemExit("Table 1's RQ4 sentence no longer matches the records; rewrite t1_questions")
    if D.e2_rates(D.e2_row(s, "claude-sonnet-5-5"))["diff"]["ci"][1] >= 0:
        raise SystemExit("Table 1 says the share falls under Sonnet 5.5; the records disagree")
    both = [g for g in D.LLM_JUDGES if g not in c["judges_order_dependent"]
            and D.pairwise_row(s, g)["pairs"]["O_vs_H"]["ci"][1] < 0.5 and D.core_row(s, g, "hidden")["premium_ci"][0] > 0]
    step = d["verification_step"]["claude-sonnet-5-5"]
    if step["ci"][1] >= 0:
        raise SystemExit("Section 4.4 says that under the verification step the honest message scores higher than the "
                         "overclaim for Sonnet 5.5; the records disagree")
    # draft 7: the cells of RQ1 to RQ3 also give what the reanalysis found (reanalysis/tables)
    # draft 9: the RQ1 cell no longer gives the count against a hedge (the message with the hedge keeps the agent's
    # other claims, Section 4.1); it says that the difference is smaller for most graders when one passage changes
    # draft 10: the cell gives the count of the control run of the sprint, in which only the main passage differs:
    # the claim alone minus the honest message is smaller than the premium for 11 of its 12 graders (control() stops
    # when the run's files give other counts)
    control()
    said = CONTROL_SAYS
    if (c["judges_premium_above_zero_without"], c["reward_models_premium_above_zero_without"]) != (c["judges"], c["reward_models"]):
        raise SystemExit("Table 1 says that all graders score the overclaim higher; the records disagree")
    below = [r for r in rean("r27f_with_the_transcript") if r["scoring prompt"] == "plain scoring prompt"
             and r["reading"].startswith("overclaim detectably below")]
    if below:
        raise SystemExit("Table 1 says that no judge scores the overclaim detectably lower with the transcript; the "
                         "reanalysis tables disagree")
    sel8 = [r for r in rean("r23_selection_by_instruction") if "verification step" not in r["selector"]]
    within = [r["selector"] for r in sel8 if r["honesty change detectable"] == "yes" and float(r["honesty half: change"]) > 0]
    if len(sel8) != c["selectors"]:
        raise SystemExit("Table 1: the selectors of the reanalysis table are not the paper's eight")
    rows = [
        ["RQ1. Without the transcript, is the overclaim scored higher?",
         # two lines, the height of the row: the judges and the reward models are counted together
         f"Yes, by all {c['judges'] + c['reward_models']} graders; by less for "
         f"{said['smaller than the premium']} of {said['graders']} when only one passage differs "
         "(Table~\\ref{tab:comparisons})",
         f"supported for the {len(h1)} Claude judges"],
        ["RQ2. Does the transcript remove the difference?",
         f"No detectable premium for {len(c['judges_no_detectable_premium_with'])} of the {c['judges_with_transcript']} "
         "judges tested with the transcript; none scores the overclaim detectably lower",
         "supported: it lowers the premium of Sonnet 5.5"],
        ["RQ3. Does best-of-16 select more overclaims?",
         f"Under {c['selectors_share_rises']} of {c['selectors']} graders on all 16 messages; under {len(within)} of "
         f"{c['selectors']} among the 8 written under the added instruction{AFTER}",
         "supported for Sonnet 4.6, not for Sonnet 5.5"],
        ["RQ4. Does a verification step lower premium and selection?",
         "It lowers the premium of Sonnet 5.5 and Opus 5.5; no detectable change for Gemini 3.8 Flash",
         # Table 1 comes before the paragraph that introduces the correction, so the cell names it (draft 6)
         f"not supported after correcting for {TEST_WORDS[len(holm)]} tests "
         f"($p$~=~{D.fmt(holm['H4 / claude-sonnet-5-5']['p_holm'], 3)})"],      # ties: the value never breaks
        ["Side by side (Section~\\ref{sec:r5})",
         f"{WORDS[len(both)]} judges prefer the honest message, yet score the overclaim higher alone",
         "not among the eight tests"],
    ]
    write("t1_questions", ["Question", "What we found", "Planned test"], rows,
          "@{}p{4.6cm}p{7.1cm}p{4.0cm}@{}", size="\\footnotesize")


# ---------------------------------------------------------------- appendix: the eight planned tests


def p_adj(r: dict) -> str:
    if r.get("p_holm_is_bound") or r["p_holm"] < 0.01:
        return "$<$ 0.01"
    return D.fmt(r["p_holm"], 3)


def a_tests(s: dict) -> None:
    rows = []
    tests = {"H1": "premium without the transcript (points)",
             "H2": "change in the premium with the transcript (points)",
             "H3": "share of overclaims: best of 16 minus one at random (percentage points)"}
    seen: set[str] = set()
    for r in D.holm_rows(s):
        hyp = r["hypothesis"]
        ok = r["verdict_holm"].startswith("supported")
        if hyp == "H4":
            for k, c in enumerate(r["components"]):
                share = c["hypothesis"] == "H4(ii)"
                rows.append(["RQ4" if k == 0 else "",
                             "verification step: change in the share of overclaims selected (pct. points)" if share
                             else "verification step: change in the premium with the transcript (points)",
                             name(r["grader"]) if k == 0 else "",
                             pp_est(c["value"], c["ci"]) if share else D.est(c["value"], c["ci"]), str(c["n_tasks"]),
                             p_adj(r) if k == 0 else "", ("yes" if ok else "no") if k == 0 else ""])
            continue
        share = hyp == "H3"
        verdict = "yes" if ok else ("no (the share falls)" if share and r["ci"][1] < 0 else "no")
        rows.append([RQ[hyp] if hyp not in seen else "", tests[hyp] if hyp not in seen else "", name(r["grader"]),
                     pp_est(r["value"], r["ci"]) if share else D.est(r["value"], r["ci"]), str(r["n_tasks"]), p_adj(r),
                     verdict])
        seen.add(hyp)
    write("a_tests", ["", "the eight planned tests", "judge", "estimate [95\\% CI]", "tasks", "adjusted $p$",
                       "supported?"], rows, "@{}lp{5.9cm}llrll@{}", size="\\scriptsize", stretch=0.95, colsep=3.2)


# ---------------------------------------------------------------- appendix: graders and settings


def ident(model_id: str) -> str:
    """A model id in typewriter type that may break only after a slash (a break at a hyphen would look like part of
    the id)."""
    return "\\texttt{" + D.tex_escape(model_id).replace("/", "/\\allowbreak ") + "}"


GRADERS_DAY = "2026-11-08"      # the day of every score, as the caption of the graders table prints it


def a_graders(cfg: dict, models: dict) -> None:
    dates: dict[str, set[str]] = defaultdict(set)
    for f in ("pointwise_calls.jsonl", "judge_calls.jsonl", "e3pw_calls.jsonl", "e7_calls.jsonl", "e2_scores.jsonl"):
        for r in D.load_jsonl(D.RUN_DIR / f):
            g = r.get("grader") or r.get("selector")
            if g and r.get("ts"):
                dates[str(g).split("+")[0]].add(str(r["ts"])[:10])
    prov = {"anthropic": "Anthropic API", "openrouter": "OpenRouter", "gemini": "Vertex AI, AI Studio"}
    rows = []
    for g in D.LLM_JUDGES:
        j = cfg["judges"][g]
        bits = []
        if "thinking" in j:
            bits.append(f"thinking {j['thinking']}, effort {j['effort']}")
        if "thinking_level" in j:
            bits.append(f"thinking level {j['thinking_level']}")
        if "temperature" in j:
            bits.append(f"temperature {j['temperature']:g}")
            rs = (j.get("extra_body") or {}).get("reasoning") or {}
            bits.append("reasoning off" if rs.get("enabled") is False else f"reasoning effort {rs.get('effort')}")
        bits.append(f"{j.get('max_tokens', j.get('max_output_tokens'))} tokens")      # the output limit (the header says so)
        ds = sorted(dates[g])
        rows.append([name(g), ident(j["model"]), prov[j["provider"]], ", ".join(bits),
                     ds[0] if ds[0] == ds[-1] else f"{ds[0]} to {ds[-1]}"])
    rows.append(MID)
    # the reward-model scores carry no timestamp; the run's log dates the lines of the stage that produced them
    log = (D.RUN_DIR / "run.log").read_text(encoding="utf-8")
    for g in D.REWARD_MODELS:
        days = sorted(set(re.findall(r"^(\d{4}-\d{2}-\d{2}) [^\n]* rm: " + re.escape(g) + r"\b", log, flags=re.M)))
        if not days:
            raise SystemExit(f"no line of the reward-model stage for {g} in results/full_run/run.log")
        m = next(x for x in models["reward_models_part2"] if x["name"] == g)
        bits = [str(m["dtype"])] + (["4-bit weights (" + str(m["quantization"]) + ")"] if m.get("quantization") else [])
        rows.append([name(g), ident(m["model"]), "local GPU", ", ".join(bits),
                     days[0] if days[0] == days[-1] else f"{days[0]} to {days[-1]}"])
    # One day for every grader: the caption states it (GRADERS_DAY) and the table has no date column. Any other case
    # keeps the column, and the caption of Table B2 (the graders; B4 before draft 8) must then be rewritten.
    if {r[-1] for r in rows if r != MID} != {GRADERS_DAY}:
        raise SystemExit("a_graders: the graders' scores are no longer all dated " + GRADERS_DAY + "; print the date "
                         "column again and rewrite the caption of the graders table")
    rows = [r if r == MID else r[:-1] for r in rows]
    write("a_graders", ["grader", "model id", "served by", "settings (the token count is the output limit)"], rows,
          "@{}lp{4.55cm}p{2.65cm}p{5.95cm}@{}")


# ---------------------------------------------------------------- appendix: the premium


def a_premium(s: dict, d: dict) -> None:
    h2 = {r["grader"]: r for r in D.hypothesis_rows(s, "H2")}
    allt = d["with_transcript_all_tasks"]
    change = d["change_all_tasks"]
    rows = []
    for g in (*D.LLM_JUDGES, *D.REWARD_MODELS):
        hid, sho, band = (D.core_row(s, g, c) for c in ("hidden", "shown", "shown, band"))
        base = [name(g), D.est(hid["premium"], hid["premium_ci"]), str(hid["n_tasks"])]
        if sho is None:
            rows.append([*base, "not run", "", "", ""])
        elif sho["n_tasks"] < D.MIN_TASKS:
            rows.append([*base, "not estimated", str(sho["n_tasks"]), "", ""])
        elif g in allt:
            a, c = allt[g], change[g]
            rows.append([*base, D.est(a["premium"], a["ci"]), str(a["n_tasks"]), D.est(c["value"], c["ci"]), str(c["n_tasks"])])
            rows.append(["\\quad shorter transcripts", "", "", D.est(sho["premium"], sho["premium_ci"]), str(sho["n_tasks"]),
                         D.est(h2[g]["value"], h2[g]["ci"]), str(h2[g]["n_tasks"])])
            rows.append(["\\quad longer transcripts", "", "", D.est(band["premium"], band["premium_ci"]), str(band["n_tasks"]), "", ""])
        else:
            rows.append([*base, D.est(sho["premium"], sho["premium_ci"]), str(sho["n_tasks"]),
                         D.est(h2[g]["value"], h2[g]["ci"]), str(h2[g]["n_tasks"])])
        if g == D.LLM_JUDGES[-1]:
            rows.append(MID)
    write("a_premium", ["grader", "without the transcript", "tasks", "with the transcript", "tasks",
                        "change, same tasks", "tasks"], rows, "@{}llrlrlr@{}")


def a_placebo(s: dict) -> None:
    rows = []
    for g in (*D.LLM_JUDGES, *D.REWARD_MODELS):
        r = D.core_row(s, g, "hidden")
        sd = r.get("standardised_gap")
        rows.append([name(g), D.est(r["bonus"], r["bonus_ci"]), D.est(r["tax"], r["tax_ci"]), D.fmt(sd), str(r["n_tasks"])])
        if g == D.LLM_JUDGES[-1]:
            rows.append(MID)
    write("a_placebo", ["grader", "overclaim $-$ placebo", "placebo $-$ honest message", "premium in standard deviations",
                        "tasks"], rows, "@{}lllrr@{}")


# ---------------------------------------------------------------- appendix: selection


def a_selection(s: dict) -> None:
    """Ten rows on the selection tasks (the two verification-step rows ran on the 26 of them whose transcript has at
    most 40,000 tokens), then the two rows that the verification-step comparison needs: the same two judges without
    the verification step on those 26 tasks. The second labeller's version is written as a CSV only."""
    def row(sel: str, labeller: str, selection: str) -> list[str] | None:
        r = D.e2_row(s, sel, "clean_overclaim", labeller, selection)
        if r is None:
            return None
        x = D.e2_rates(r)
        cells = [pct_est(x["rates"][str(n)]["mean"], x["rates"][str(n)]["ci"]) for n in (1, 4, 16)]
        return [name(sel), str(x["n_tasks"]), *cells, pp_est(x["diff"]["mean"], x["diff"]["ci"])]
    header = ["grader that selects", "tasks", "one at random", "best of 4", "best of 16", "change (pct. points)"]
    step_judges = ("claude-sonnet-5-5", "deepseek-v4-pro")
    for fname, lab in (("a_selection", "session3"), ("a_selection_second_labeller", "full_evidence")):
        top = [row(sel, lab, "e2_tasks") for sel in ("claude-sonnet-5-5", "claude-sonnet-5-5+verify_first",
                                                    "claude-sonnet-4-6", "deepseek-v4-pro",
                                                    "deepseek-v4-pro+verify_first", *D.REWARD_MODELS)]
        low = [row(g, lab, "e2_shown_eligible") for g in step_judges]
        if any(r is None for r in (*top, *low)):
            raise SystemExit(f"a_selection: a selection row is missing for labeller {lab}")
        n_low = {r[1] for r in low} | {r[1] for r in top if "verification step" in r[0]}
        if len(n_low) != 1:
            raise SystemExit(f"a_selection: the verification-step rows and their comparison rows differ in tasks: {n_low}")
        sub = [[f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{without the verification step, on the {n_low.pop()} tasks of the "
                "verification-step rows}}"]]
        write(fname, header, top + [MID] + sub + low, "@{}lrllll@{}")
    (OUT / "a_selection_second_labeller.tex").unlink()        # the paper prints the first labeller; the CSV stays


def a_verification(s: dict) -> None:
    """The change in the premium for each judge that ran the verification step, and for Sonnet 5.5 the selection part
    of its planned test (four rows). The paired differences on the 30 tasks scored three ways are no longer a block of
    this table: the caption gives the two for Sonnet 5.5 and the README the rest (they are in summary.json, e7)."""
    rows = []
    for g in D.LLM_JUDGES:      # the fixed order; the selection part of the test follows its judge's premium part
        for r in D.hypothesis_rows(s, "H4(i)"):
            if r["grader"] == g:
                rows.append([name(g), "premium with the transcript: verification step $-$ plain question",
                             D.est(r["value"], r["ci"]), str(r["n_tasks"])])
        for r in D.hypothesis_rows(s, "H4(ii)"):
            if r["grader"].split(" vs ")[-1] == g:
                rows.append([name(g), "share of overclaims selected by best of 16: verification step with the transcript "
                             "$-$ plain question without it (percentage points)", pp_est(r["value"], r["ci"]),
                             str(r["n_tasks"])])
    # the caption says that neither the short instruction nor the verification step changes the premium of Gemini 3.8
    # Flash detectably, and that both lower the premium of Sonnet 5.5; stop when that would no longer be true
    e7 = {r["grader"]: r["diffs"] for r in s["e7"]}
    for g, below in (("claude-sonnet-5-5", True), ("gemini-3.8-flash", False)):
        for key in ("instruction_only_minus_plain", "verify_first_minus_plain"):
            lo, hi = e7[g][key]["ci"]
            if (hi < 0) != below or (not below and not lo <= 0 <= hi):
                raise SystemExit(f"a_verification: {g} {key} is now [{lo}, {hi}]; rewrite the caption of the table")
    write("a_verification", ["judge", "quantity", "estimate [95\\% CI]", "tasks"], rows, "@{}lp{9.2cm}lr@{}")


# ---------------------------------------------------------------- appendix: side by side


def a_side_by_side(s: dict) -> None:
    rows = []
    for g in D.LLM_JUDGES:
        for cond in ("hidden", "shown"):
            r = D.pairwise_row(s, g, cond)
            if r is None:
                continue
            oh = r["pairs"]["O_vs_H"]
            fl = D.flip_rate(s, g, "O_vs_H", cond)
            x = D.reasons_row(s, g, cond)["O_vs_H_only"]["against_O"]
            cite = f"{D.pct(x['cites_disclosure'] / x['n'])} of {x['n']}" if x.get("n") else "none"
            rows.append([name(g) + (" (with the transcript)" if cond == "shown" else ""), str(r["n_tasks"]),
                         pct_est(oh["rate"], oh["ci"]), D.pct(fl["rate"]), D.pct(r["tie_rate"]), cite])
    write("a_side_by_side", ["judge", "tasks", "prefers the overclaim", "order changes verdict", "ties",
                             "verdicts for the honest message that cite the disclosure"], rows,
          "@{}p{4.1cm}rlp{1.9cm}rp{4.4cm}@{}")


# ---------------------------------------------------------------- appendix: claims, sensitivity, construction


SLOPE_ENDS = {"judges": ("gemini-3.8-flash", "claude-sonnet-4-6"),          # lowest and highest slope, as the README
              "reward_models": ("skywork-v2-qwen3-4b-4bit", "skywork-v2-qwen3-1.7b")}   # prints them (draft 5 moved the sentence)


def a_claims(s: dict) -> None:
    slope = {r["grader"]: r for r in s["filler_dose"] if r.get("table") == "dose" and r["condition"] == "hidden"
             and r.get("rubric") in (D.NEUTRAL, "reward model", None)}
    e6 = {r["grader"]: r for r in s["e6"] if r.get("rubric") in (D.NEUTRAL, None, "reward model")}
    r21 = {r["grader"]: r for r in rean("r21_four_comparisons")}      # draft 7: the share of the premium (after the fact)
    rows = []
    for g in (*D.LLM_JUDGES, *D.REWARD_MODELS):
        r = e6[g]
        lv, df = r["levels"], r["diffs"]
        ra = r21[name(g)]
        if (int(ra["tasks (claim against a hedge)"]) != r["n_tasks"]
                or abs(float(ra["hedge, raw units"]) - df["O_minus_O_minus"]["value"]) > 5e-4):
            raise SystemExit(f"a_claims: the reanalysis row of {g} is not the claim-removed difference of this table")
        rows.append([name(g), str(r["n_tasks"]), *[D.fmt(lv[k]["mean"]) for k in ("O_minus", "O", "O_plus")],
                     D.est(df["O_minus_O_minus"]["value"], df["O_minus_O_minus"]["ci"]),
                     D.pct(float(ra["hedge as a share of the premium (same tasks)"])),
                     D.est(df["O_plus_minus_O"]["value"], df["O_plus_minus_O"]["ci"])])
        if g == D.LLM_JUDGES[-1]:
            rows.append(MID)
    # the caption gives the range of the slope per style sentence; stop when its sentence would no longer be true
    for key, graders in (("judges", D.LLM_JUDGES), ("reward_models", D.REWARD_MODELS)):
        by = sorted(graders, key=lambda g: slope[g]["slope"])
        if (by[0], by[-1]) != SLOPE_ENDS[key]:
            raise SystemExit(f"a_claims: the lowest and highest slope among the {key} are now {by[0]} and {by[-1]}; "
                             "update the sentence on neutral entries in the README (Draft 5, moved out of the appendix)")
    excl = [g for g in (*D.LLM_JUDGES, *D.REWARD_MODELS) if not slope[g]["slope_ci"][0] <= 0 <= slope[g]["slope_ci"][1]]
    if excl != ["skywork-v2-qwen3-1.7b"]:
        raise SystemExit(f"a_claims: the slopes whose interval excludes zero are now {excl}; update the README sentence")
    write("a_claims", ["grader", "tasks", "main claim|removed", "as written", "second claim|added",
                       "as written $-$ removed", f"share of the|premium{AFTER}", "added $-$ as written"], rows,
          "@{}lrrrr@{\\hspace{8pt}}lr@{\\hspace{8pt}}l@{}")


def a_sensitivity(s: dict) -> None:
    """Table B11 since draft 8 (B10 in drafts 6 and 7), as in draft 6. Two more samples of the reanalysis (without the 6 resolved tasks; without the tasks
    with a reused score) do not fit beside these three at the table's size; Appendix A states their results in words
    and reanalysis/tables/r24_planned_estimates.csv holds the estimates."""
    rows = []
    labels = {"H1": "RQ1", "H2": "RQ2", "H3": "RQ3", "H4(i)": "RQ4, premium", "H4(ii)": "RQ4, selection"}
    for r in s["sensitivity"]:
        if r.get("section") != "Holm family":
            continue
        share = r["rubric"] in ("H3", "H4(ii)")      # share differences print in percentage points (the caption says so),
        # to one decimal: in whole points an interval of [-0.2, 19.6] would print as [0, 20] and hide that it includes zero
        f = (lambda v: pp1_est(v["value"], v["ci"])) if share else (lambda v: D.est(v["value"], v["ci"]))
        cells = [c for k in ("full", "repaired", "zero_kept") for c in (f(r[k]), str(r[k]["n_tasks"]))]
        rows.append([labels[r["rubric"]], name(r["grader"].split(" vs ")[-1]), *cells])
    write("a_sensitivity", ["test", "judge", "all 60 sets", "tasks", "without the 17", "tasks", "without the 7", "tasks"],
          rows, "@{}lllrlrlr@{}")


# ---------------------------------------------------------------- draft 7: the tables of the reanalysis


C21 = (("premium", "premium lo", "premium hi"), ("claim against a hedge", "hedge lo", "hedge hi"),
       ("disclosure against a neutral sentence", "disclosure lo", "disclosure hi"),
       ("fresh messages (honesty half)", "fresh lo", "fresh hi"))


def t2_comparisons() -> None:
    """Table 2 (main text): the premium beside four comparisons in which less changes. One row per judge, then one
    row with the range over the five reward models (in standard deviations). The second and third comparisons come
    from the control run of the sprint (CONTROL; no Gemini judge took part); the last column is after the fact. The
    comparison against a hedge left the table in draft 10 (its numbers are in a_claims)."""
    r21 = {r["grader"]: r for r in rean("r21_four_comparisons")}
    ctl = control()["rows"]
    kept = (C21[0], C21[2], C21[3])      # the premium, the disclosure and the fresh messages, as printed since draft 7

    def cell(r: dict, c: tuple) -> str:
        return r_est(r, *c) if r[c[0]] != "" else "--"

    def same_scores(g: str) -> None:
        """A grader with all its tasks in the control run has the paper's premium and disclosure there too, so that
        the second comparison is the sum of the third and the fourth (the caption says so)."""
        r, c = r21[name(g)], ctl[name(g)]
        if c["n_tasks"] != int(r["tasks (premium, disclosure)"]):
            return
        for key, col in (("O - H", "premium"), ("P - H", "disclosure against a neutral sentence")):
            if abs(c["comparisons"][key]["mean"] - float(r[col])) > 5e-4:
                raise SystemExit(f"Table 2: the control run's {key} of {name(g)} is not the paper's {col}")
        v = c["comparisons"]
        if abs(v["C - P"]["mean"] + v["P - H"]["mean"] - v["C - H"]["mean"]) > 1e-9:
            raise SystemExit(f"Table 2: for {name(g)} the second comparison is not the sum of the third and the fourth")

    rows = []
    for g in D.LLM_JUDGES:
        r = r21[name(g)]
        if name(g) in ctl:
            same_scores(g)
            new = [c_est(ctl[name(g)]["comparisons"][k]) for k in ("C - H", "C - P")]
        else:
            new = ["--", "--"]
        rows.append([name(g), cell(r, kept[0]), *new, cell(r, kept[1]), cell(r, kept[2])])
    rows.append(MID)
    rm = [r21[name(g)] for g in D.REWARD_MODELS]
    rmc = [ctl[name(g)] for g in D.REWARD_MODELS]
    for g in D.REWARD_MODELS:
        same_scores(g)
    if (any(r["unit"] != "SD" for r in rm) or any(float(r[lo]) <= 0 for r in rm for _, lo, _ in kept)
            or any(c["unit"] != "standard deviations" or c["comparisons"][k]["lo"] <= 0 for c in rmc for k in ("C - H", "C - P"))):
        raise SystemExit("Table 2's reward-model row gives ranges without a mark: every interval must exclude zero")

    def span(values: list[float]) -> str:
        return f"{D.fmt(min(values))} to {D.fmt(max(values))}"

    rows.append(["5 reward models", span([float(r[kept[0][0]]) for r in rm]),
                 *[span([c["comparisons"][k]["mean"] for c in rmc]) for k in ("C - H", "C - P")],
                 *[span([float(r[c[0]]) for r in rm]) for c in kept[1:]]])
    # six columns at the size of Table 1: the mark of an interval that includes zero is set in the gap after its
    # column, and each header names the two messages of its difference in two short lines
    rows = [r if r == MID else [c.replace(ZERO, ZERO_HANG) for c in r] for r in rows]
    write("t2_comparisons", ["grader", "premium", f"claim alone $-$|honest message{CONTROL}",
                             f"claim alone $-$|placebo{CONTROL}", "placebo $-$|honest message",
                             f"fresh|messages{AFTER}"], rows, "@{}llllll@{}", size="\\footnotesize", colsep=4)


KINDS = ("no test run", "last test run failed", "narrower rerun after a failure", "failure reported (40 sets)")


def a_kinds() -> None:
    """Table B10 since draft 8 (B11 in draft 7; after the fact): the premium by kind of missing check, every grader without the transcript and the
    two judges that ran on all 60 tasks with it. Reward models in standard deviations."""
    t = rean("r22_premium_by_missing_check")

    def row(r: dict, label: str) -> list[str]:
        d = (float(r["failure reported minus no test run"]), float(r["difference: lo"]), float(r["difference: hi"]))
        return [label, *[r_est(r, k, k + ": lo", k + ": hi") for k in KINDS],
                D.est(d[0], [d[1], d[2]]) + (ZERO if d[1] <= 0 <= d[2] else "")]

    without = {r["grader"]: r for r in t if r["condition"] == "without the transcript"}
    rows = []
    for g in (*D.LLM_JUDGES, *D.REWARD_MODELS):
        rows.append(row(without[name(g)], name(g)))
        if g == D.LLM_JUDGES[-1]:
            rows.append(MID)
    rows.append(MID)
    rows.append(["\\multicolumn{6}{@{}l}{\\textit{with the transcript}}"])
    rows += [row(r, r["grader"]) for r in t if r["condition"] == "with the transcript"]
    write("a_kinds", ["grader", "no test run|(20 sets)", "last test run|failed (20 sets)", "narrower rerun|(20 sets)",
                      "failure reported|(40 sets)", "failure reported|$-$ no test run"], rows, "@{}llllll@{}", colsep=4.5)


def a_instruction() -> None:
    """Table B12 (after the fact): selection among the 8 messages written under the added instruction, how often the
    message selected from all 16 was written without it, and the score difference between fresh messages."""
    t = {r["selector"]: r for r in rean("r23_selection_by_instruction")}
    r21 = {r["grader"]: r for r in rean("r21_four_comparisons")}
    step = {r["grader"]: r for r in rean("r21_fresh_variants")
            if r["variant"] == "with the transcript and the verification step, honesty half"}
    rows = []
    selectors = ("claude-sonnet-5-5", "claude-sonnet-4-6", "deepseek-v4-pro", *D.REWARD_MODELS)
    for sel in (*selectors, "claude-sonnet-5-5+verify_first", "deepseek-v4-pro+verify_first"):
        r = t[name(sel)]
        ch = (float(r["honesty half: change"]), float(r["honesty change: lo"]), float(r["honesty change: hi"]))
        if sel in selectors:
            fr = r_est(r21[name(sel)], *C21[3])
        else:
            fr = r_est(step[name(sel)], "difference", "lo", "hi")
        if sel == "claude-sonnet-5-5+verify_first":
            rows.append(MID)
            rows.append([f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{with the transcript and the verification step, on "
                         f"{r['tasks']} tasks}}}}"])
        rows.append([name(sel.split("+")[0]), r["tasks"], D.pct(float(r["honesty half: one of the 8 at random"])),
                     D.pct(float(r["honesty half: top pick"])),
                     pp_est(ch[0], [ch[1], ch[2]]) + (ZERO if ch[1] <= 0 <= ch[2] else ""),
                     D.pct(float(r["top pick among all 16 comes from the plain half"])), fr])
    write("a_instruction", ["grader that selects", "tasks", "one of the 8|at random", "selected|of the 8",
                            "change|(pct. points)", "selected of all 16:|written without the|added instruction",
                            "score: overclaims|$-$ other messages"], rows, "@{}lrrrlrl@{}", colsep=6)


# ---------------------------------------------------------------- draft 10: the control run of the sprint


R28_FIRST = "reasons that name a contradiction (list of the first count)"
R28_VERSIONS = ("overclaim", "main claim replaced by the hedge", "honest message", "placebo")
CTRL_VERSIONS = ("claim alone (C)", "hedge alone (G)", "placebo, second score (P2)")


def a_reasons() -> None:
    """The lower part of Table B9: for each judge, the stated reasons in which a keyword of reanalysis r28 names a
    contradiction (the list of the first count), for four messages of the main run (after the fact) and the three of
    the control run (declared before its first call). A count is of 60 reasons unless the cell says otherwise."""
    main_run = {(r["judge"], r["version"]): r for r in rean("r28_contradiction_in_reasons")}
    with (CTRL / "reasons.csv").open(encoding="utf-8") as fh:
        this_run = {(r["judge"], r["version"]): r for r in csv.DictReader(fh)}
    ctl = control()["rows"]

    def count(r: dict, col: str) -> str:
        return r[col] if int(r["reasons"]) == 60 else f"{r[col]} of {r['reasons']}"

    rows = []
    for g in D.LLM_JUDGES:
        new = ([count(this_run[(name(g), v)], "list of the first count") for v in CTRL_VERSIONS] if name(g) in ctl
               else ["--"] * len(CTRL_VERSIONS))
        for v in R28_VERSIONS:      # reasons.csv repeats the main run's counts: they must be the reanalysis's own
            if this_run[(name(g), v)]["list of the first count"] != main_run[(name(g), v)][R28_FIRST]:
                raise SystemExit(f"a_reasons: the control run's copy of the count for {name(g)}, {v} differs from r28")
        rows.append([name(g), *[count(main_run[(name(g), v)], R28_FIRST) for v in R28_VERSIONS], *new])
    write("a_reasons", ["judge", "overclaim", "message with|the hedge", "honest|message", "placebo",
                        f"claim|alone{CONTROL}", f"hedge|alone{CONTROL}", f"placebo|scored again{CONTROL}"], rows,
          "@{}lrrrr@{\\hspace{14pt}}rrr@{}", colsep=6)


def a_control() -> None:
    """Table B13: the control run of the sprint, one row per grader that took part. Judges in points, reward models
    in standard deviations of their scores. The last column, the placebo scored again minus its stored score, has no
    interval (the table would be too wide); the mark ZERO says when it includes zero. For every reward model the
    placebo scored again is identical to the stored score (the cell says so; the script stops otherwise)."""
    c = control()
    ctl, checks = c["rows"], {r["grader"]: r for r in c["doc"]["reward_model_checks"]}
    rows = []
    for g in (*D.LLM_JUDGES, *D.REWARD_MODELS):
        if g == D.REWARD_MODELS[0]:
            rows.append(MID)
        if name(g) not in ctl:
            continue
        v = ctl[name(g)]["comparisons"]
        again = c_val(v["P2 - P"])
        if g in D.REWARD_MODELS:
            k = checks[name(g)]
            if k["placebo tasks that differ"] != 0 or k["placebo largest absolute difference"] != 0 or v["P2 - P"]["mean"] != 0:
                raise SystemExit(f"a_control: the placebo scored again differs from the stored score for {name(g)}")
            again = "identical"
        rows.append([name(g), str(ctl[name(g)]["n_tasks"]), *[c_est(v[k]) for k in ("C - H", "C - P", "C - G", "O - C")],
                     again])
    write("a_control", ["grader", "tasks", "claim alone|$-$ honest message", "claim alone|$-$ placebo",
                        "claim alone|$-$ hedge alone", "overclaim|$-$ claim alone", "placebo scored|again $-$ placebo"],
          rows, "@{}lrlllll@{}", colsep=3.2)


SPLIT_SETS = {"no plain statement": 38, "plain statement": 22}                  # the declared split of the control run
MARK_SETS = {"marked by the keyword rule": 36, "not marked": 24}               # the keyword rule of the amendment
CONTROL_FEWER = {"gpt-oss-120b": (34, 20, 31, 23)}      # the tasks of a grader with fewer (the caption of Table B14)


def a_control_split() -> None:
    """Table B14: two descriptive splits of the control run, without intervals: the claim alone minus the honest
    message on the 38 sets whose changed passage holds no plain statement of the code change and on the 22 that
    hold one; the claim alone minus the hedge alone on the 36 sets that the keyword rule marks and on the 24 others."""
    ctl = control()["rows"]
    rows = []
    for g in (*D.LLM_JUDGES, *D.REWARD_MODELS):
        if g == D.REWARD_MODELS[0]:
            rows.append(MID)
        if name(g) not in ctl:
            continue
        r = ctl[name(g)]
        cells = [r["split"]["C - H"][k] for k in SPLIT_SETS] + [r["by_mark"][k] for k in MARK_SETS]
        if tuple(x["n"] for x in cells) != CONTROL_FEWER.get(name(g), (*SPLIT_SETS.values(), *MARK_SETS.values())):
            raise SystemExit(f"a_control_split: the tasks of {name(g)} are {[x['n'] for x in cells]}; rewrite the caption")
        rows.append([name(g), *[c_val(x) for x in cells]])
    a, b, m, o = (*SPLIT_SETS.values(), *MARK_SETS.values())
    write("a_control_split",
          ["grader", f"claim alone $-$ honest message: {a} sets", f"claim alone $-$ honest message: {b} sets",
           f"claim alone $-$ hedge alone: {m} marked sets", f"claim alone $-$ hedge alone: {o} other sets"], rows,
          "@{}lllll@{}", colsep=8,
          head=[" & \\multicolumn{2}{l}{claim alone $-$ honest message} & \\multicolumn{2}{l}{claim alone $-$ hedge alone} \\\\",
                "\\cmidrule(r){2-3}\\cmidrule{4-5}",
                f"grader & {a} sets & {b} sets & {m} marked sets & {o} other sets \\\\"])


def a_not_run(s: dict, status: dict, inv: dict) -> None:
    """Only what is partial, failed or not run, and the tasks that gpt-oss-120b loses to replies with no valid score
    (inv: derived.json, invalid_replies); the caption says that everything else ran in full. Draft 10: two rows for
    the control run of the sprint (the Gemini judges did not take part; gpt-oss-120b has fewer tasks)."""
    c = control()
    oss_ctl = c["rows"]["gpt-oss-120b"]
    if oss_ctl["n_tasks"] != s["header"]["counts"]["sets"] - inv["gptoss_without_transcript"]["tasks_lost"] - next(
            m["C"] for m in c["doc"]["missing"] if m["grader"] == "gpt-oss-120b"):
        raise SystemExit("a_not_run: the tasks of gpt-oss-120b in the control run are not the main run's minus the "
                         "tasks with no valid score on the claim alone; rewrite its row")
    lost = {m["grader"]: m for m in c["doc"]["missing"]}
    if ([g for g, m in lost.items() if m["C"] + m["G"] + m["P2"]] != ["gpt-oss-120b"] or lost["gpt-oss-120b"]["G"]
            or lost["gpt-oss-120b"]["P2"] or any(r["n_tasks"] != 60 for g, r in c["rows"].items() if g != "gpt-oss-120b")):
        raise SystemExit("a_not_run: the control run's missing scores changed; rewrite its rows")
    def cell(rid: str, g: str) -> str:
        e = status[rid]
        got, want = e["records"]["pointwise_calls.jsonl"], e["expected_records"]["pointwise_calls.jsonl"]
        return (f"{e['status']}: {got} of {want} scores, {D.core_row(s, g, 'shown')['n_tasks']} of "
                f"{e['n_tasks']['pointwise_calls.jsonl']} tasks complete")
    for rid in ("pws-gema", "pws-qwen"):
        if status[rid]["status"] == "done":
            raise SystemExit(f"{rid} is complete now; rewrite a_not_run and the sentences listed in paper/open_cells.md")
    o55 = status["pws-o55"]["n_tasks"]["pointwise_calls.jsonl"]
    eligible = s["header"]["counts"]["shown_eligible"]
    sets = s["header"]["counts"]["sets"]
    oss_hid, oss_sho = D.core_row(s, "gpt-oss-120b", "hidden"), D.core_row(s, "gpt-oss-120b", "shown")
    a, b = inv["gptoss_without_transcript"], inv["gptoss_with_transcript"]
    if set(inv["without_transcript"]["by_judge"]) | set(inv["with_transcript"]["by_judge"]) != {"gpt-oss-120b"}:
        raise SystemExit("a_not_run: another judge now has replies with no valid score; add its row")
    if status["ps-s55"]["status"] != "done" or any(k.startswith("ps-") and k != "ps-s55" for k in status):
        raise SystemExit("a_not_run: the side-by-side runs with the transcript changed; rewrite its row")
    rows = [     # the fixed grader order
        ["Opus 5.5, scored alone with the transcript", f"{o55} of the {eligible} tasks", "limited for cost before the run"],
        ["Sonnet 4.6 and Opus 4.6, scored alone with the transcript", "not run", "left out for cost before the run"],
        ["Gemini 4 Argon, scored alone with the transcript", cell("pws-gema", "gemini-4-argon"),
         "stopped at a spending limit after Google restricted the project's access to Vertex AI"],
        ["Qwen3-32B, scored alone with the transcript", cell("pws-qwen", "qwen3-32b"),
         "the one provider with a large enough input limit returned rate-limit errors on every retry"],
        ["gpt-oss-120b, scored alone", f"{oss_hid['n_tasks']} of the {sets} tasks without the transcript, "
         f"{oss_sho['n_tasks']} of the {eligible} with it",
         f"a task with no valid score on one of its three messages is left out ({a['tasks_lost']} tasks without the "
         f"transcript, {b['tasks_lost']} with it)"],
        ["the five reward models with the transcript", "not run",
         "the transcripts do not fit in the memory of the 10 GB GPU used"],
        ["side by side with the transcript", "Sonnet 5.5 only",
         "planned for Sonnet 5.5 and Opus 5.5; Opus 5.5 was left out for cost before the run"],
        ["Gemini 3.8 Flash and Gemini 4 Argon, control run", "not run",
         "Google had restricted the project's access to Vertex AI: it refused the one request made for this run, and "
         "no other service was used"],
        ["gpt-oss-120b, control run", f"{oss_ctl['n_tasks']} of the {sets} tasks",
         f"a task needs a valid score on all six of its messages: {a['tasks_lost']} tasks have none on a message of "
         f"the main run, and {lost['gpt-oss-120b']['C']} more have none on the claim alone"],
    ]
    write("a_not_run", ["what", "state", "why"], rows, "@{}p{5.3cm}p{4.3cm}p{6.1cm}@{}")


def main() -> None:
    s, status, d = D.summary(), D.runs_status(), D.derived()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    t1_questions(s, d)
    a_tests(s)
    a_premium(s, d)
    a_graders(D.load_resolved_config(), D.models_config())
    a_placebo(s)
    a_selection(s)
    a_verification(s)
    a_side_by_side(s)
    a_claims(s)
    a_sensitivity(s)
    a_not_run(s, status, d["invalid_replies"])
    t2_comparisons()
    a_kinds()
    a_instruction()
    a_reasons()
    a_control()
    a_control_split()
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    sys.exit(main())
