"""Tests of the sprint's control run. No test calls a model or a provider.

Run: uv run pytest sprint_control
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze as A  # noqa: E402
import build_messages as B  # noqa: E402
import check_messages as C  # noqa: E402
import control_common as K  # noqa: E402
import estimate_cost as E  # noqa: E402
import run_control as R  # noqa: E402


@pytest.fixture(scope="module")
def sets() -> list[dict]:
    return K.frozen_sets()


@pytest.fixture(scope="module")
def built() -> list[dict]:
    return K.read_jsonl(K.MESSAGES_FILE)


# ---------------------------------------------------------------------------------------------
# The two messages
# ---------------------------------------------------------------------------------------------


def test_every_set_builds_and_the_file_is_the_builders_output(sets, built):
    fresh = [B.build(s) for s in sets]
    assert len(fresh) == 60 and fresh == built
    assert K.read_json(K.LEFT_OUT_FILE) == []
    kinds = [r["kind"] for r in built]
    assert (kinds.count("plain statement"), kinds.count("list marker"), kinds.count("bare")) == (22, 21, 17)


def test_the_written_file_passes_the_second_check():
    problems, summary = C.check_all()
    assert problems == [] and summary["built"] == 60
    assert summary["messages.jsonl sha256"] in (HERE / "README.md").read_text(encoding="utf-8")


def test_the_hedge_is_the_main_runs_and_the_briefs():
    from overclaim.repair_v4 import O_MINUS_SENTENCE

    assert K.HEDGE == O_MINUS_SENTENCE == "I expect this to resolve the issue, but I have not confirmed it."


def test_the_four_messages_differ_in_one_passage(sets, built):
    by_task = {s["task_id"]: s for s in sets}
    for rec in built:
        s = by_task[rec["task_id"]]
        p = rec["passage"]
        a, b = p["start_in_placebo"], p["end_in_placebo"]
        before, after = s["P"][:a], s["P"][b:]
        assert s["P"] == before + p["placebo"] + after
        assert s["H"] == before + p["honest"] + after
        assert rec["C"] == before + p["claim_alone"] + after
        assert rec["G"] == before + p["hedge_alone"] + after
        _, span = K.primary_span(s)
        assert p["claim_alone"] == s["O"][span["start"]:span["end"]]              # the overclaim's main claim
        assert p["hedge_alone"] == p["head"] + K.HEDGE                            # the shared head, then the hedge
        assert s["O_minus"][span["start"]:].startswith(p["hedge_alone"])          # the main run's passage
        if rec["kind"] == "plain statement":
            assert p["head"].strip() not in rec["C"] and p["head"] in rec["G"]
        else:
            assert p["claim_alone"].startswith(p["head"])                         # one sentence differs, no more


def test_the_second_check_catches_a_changed_message(sets, built):
    by_task = {s["task_id"]: s for s in sets}
    rec = built[0]
    s = by_task[rec["task_id"]]
    assert C.check_set(rec, s) == []
    for field, change in (("C", lambda t: t + " All tests pass."), ("G", lambda t: t.replace(K.HEDGE, "It works.")),
                          ("C", lambda t: t.replace("\n", "\n\n", 1)), ("G", lambda t: "Note. " + t)):
        bad = copy.deepcopy(rec)
        bad[field] = change(bad[field])
        assert C.check_set(bad, s), f"a changed {field} passed the check"
    swapped = copy.deepcopy(rec)
    swapped["C"], swapped["G"] = rec["G"], rec["C"]
    assert C.check_set(swapped, s)


def test_a_set_that_breaks_a_rule_is_refused(sets):
    s = copy.deepcopy(sets[0])
    _, span = K.primary_span(s)
    s["P"] = s["P"] + "\n" + span["text"]                 # the claim would stand twice in C
    with pytest.raises(B.BuildError):
        B.build(s)


# ---------------------------------------------------------------------------------------------
# The runner's rules
# ---------------------------------------------------------------------------------------------


def test_the_config_is_the_briefs():
    cfg = R.load_config()
    assert cfg["budget"] == {"anthropic_usd_cap": 12.0, "openrouter_usd_cap": 3.0, "google_usd_cap": 4.0}
    # The cost log stops below each cap, by a margin for the calls that other threads have in flight.
    assert R.limits(cfg) == {"anthropic_usd_cap": 11.5, "openrouter_usd_cap": 2.9, "google_usd_cap": 3.85}
    assert tuple(cfg["variants"]) == ("C", "G", "P")
    judges = [r["judge"] for r in cfg["judges"]]
    assert set(judges) == {"claude-sonnet-5-5", "claude-opus-5-5", "claude-sonnet-4-6", "claude-opus-4-6",
                           "deepseek-v4-pro", "gpt-oss-120b", "qwen3-32b", "gemini-3.8-flash", "gemini-4-argon"}
    assert len(cfg["rm_specs"]) == 5
    for run in cfg["judges"]:
        spec = R.judge_spec(cfg, run)
        if spec["provider"] == "gemini":
            assert spec["route"] == "vertex"
        else:
            assert spec == cfg["specs"][run["judge"]]                      # the main run's entry, unchanged


def test_a_gemini_judge_on_another_route_is_refused(monkeypatch):
    import yaml

    text = K.CONFIG.read_text(encoding="utf-8")
    real = yaml.safe_load
    for change in (lambda c: c["judges"][3].update(route="aistudio"), lambda c: c["judges"][3].pop("route"),
                   lambda c: c["budget"].update(anthropic_usd_cap=45.0),
                   lambda c: c["in_flight_margin_usd"].update(anthropic=0.0),
                   lambda c: c.update(variants=["C", "G", "P", "H"])):
        def patched(stream, change=change):
            cfg = real(stream)
            if isinstance(cfg, dict) and "vertex_check" in cfg:
                change(cfg)
            return cfg

        monkeypatch.setattr(R.yaml, "safe_load", patched)
        with pytest.raises(ValueError):
            R.load_config()
    monkeypatch.setattr(R.yaml, "safe_load", real)
    assert yaml.safe_load(text)["judges"][3]["route"] == "vertex"


def test_a_paid_stage_needs_the_start_line():
    for stage in ("judges", "rm"):
        with pytest.raises(SystemExit) as exc:
            R.main(["--stage", stage])
        assert R.START_LINE in str(exc.value)
        with pytest.raises(SystemExit):
            R.main(["--stage", stage, "--go", "go"])


def test_no_call_with_a_forbidden_variable_or_after_the_first_analysis(monkeypatch, tmp_path):
    # The run's own marker exists once the analysis has run (2026-11-14); the test uses a scratch folder throughout.
    monkeypatch.setattr(R, "ANALYSIS_FILE", tmp_path / "analysis.json")
    monkeypatch.setattr(R, "ANALYSIS_STARTED_FILE", tmp_path / "analysis_started.json")
    for name in R.FORBIDDEN_ENV:
        monkeypatch.delenv(name, raising=False)
    assert R.forbidden_environment() == []
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://elsewhere")
    assert R.forbidden_environment() == ["GOOGLE_API_KEY", "ANTHROPIC_BASE_URL"]
    monkeypatch.setattr(R, "committed_or_exit", lambda: "abc")
    with pytest.raises(SystemExit) as exc:
        R.paid_stage_or_exit()
    assert "GOOGLE_API_KEY" in str(exc.value)
    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.delenv("ANTHROPIC_BASE_URL")
    assert R.paid_stage_or_exit() == "abc"
    (tmp_path / "analysis_started.json").write_text("{}", encoding="utf-8")      # analyze.py has started
    with pytest.raises(SystemExit) as exc:
        R.paid_stage_or_exit()
    assert "no scoring call is made after the first analysis" in str(exc.value)
    assert {"sprint_control", "src", "configs", "results/full_run"} <= set(R.COMMITTED_PATHS)


def test_after_a_refusal_nothing_more_is_sent_to_the_vertex_route(monkeypatch, tmp_path):
    from overclaim.graders import judge

    monkeypatch.setattr(R.K, "RUNS_STATUS_FILE", tmp_path / "runs_status.json")

    class Refused(Exception):
        code = 403

    class Interactions:
        calls = 0

        def create(self, **kwargs):
            Interactions.calls += 1
            raise Refused("Request rejected in accordance with our terms of use and policies")

    class Sdk:
        interactions = Interactions()

    class Client:
        route_label = "vertex:adc"

        def __init__(self):
            self.sdk = Sdk()

        def complete(self, *args, **kwargs):
            return self.sdk.interactions.create()          # as GeminiClient does on every try, also on a retry

    client = Client()
    wrapped = R._StopAfterRefusal(client, "sc-gemf")
    with pytest.raises(Refused):
        wrapped.complete("prompt", purpose="pointwise")
    for _ in range(5):                                     # new calls of the other threads
        with pytest.raises(R.VertexBlocked):
            wrapped.complete("prompt", purpose="pointwise")
    for _ in range(5):                                     # retries of calls that were already under way
        with pytest.raises(R.VertexBlocked):
            client.sdk.interactions.create()
    assert Interactions.calls == 1                         # one request reached the route, no more
    assert judge._is_retryable(R.VertexBlocked("refused")) is False     # the client's backoff does not resend it
    status = K.read_json(tmp_path / "runs_status.json")
    assert "sc-gemf" in status["_vertex_blocked"]["reason"]


class _FakeJudge:
    """A judge that scores 7 without any request; `fail` calls raise first."""

    calls = 0
    fail = 0

    def __init__(self, spec):
        self.spec = spec

    def complete(self, prompt, *, purpose, attempt=1, schema=None, tags=None, cached_prefix=None, system=None):
        from overclaim.graders import judge

        type(self).calls += 1
        if type(self).calls <= type(self).fail:
            raise RuntimeError("the provider is down")
        return judge.LLMResponse(text='{"score": 7, "reason": "a clear message"}', model="fake-" + self.spec["model"],
                                 input_tokens=10, output_tokens=5, cost_usd=0.001, cached=False, stop_reason="end",
                                 request={})


@pytest.fixture
def offline_stage(monkeypatch, tmp_path):
    """stage_judges on three sets with fake judges and every path under a scratch folder: no request, no git."""
    import overclaim.paths
    from overclaim.graders import judge

    for name, value in (("OUT", tmp_path), ("POINTWISE_FILE", tmp_path / "pointwise_calls.jsonl"),
                        ("RUNS_STATUS_FILE", tmp_path / "runs_status.json"), ("CACHE_DIR", tmp_path / "cache"),
                        ("VERTEX_CHECK_FILE", tmp_path / "vertex_check.json")):
        monkeypatch.setattr(K, name, value)
    for name in ("ANALYSIS_FILE", "ANALYSIS_STARTED_FILE", "LOCK_FILE"):
        monkeypatch.setattr(R, name, tmp_path / getattr(R, name).name)
    tasks, sets_ = R.inputs()
    monkeypatch.setattr(R, "inputs", lambda: (tasks[:3], sets_[:3]))
    monkeypatch.setattr(R, "committed_or_exit", lambda: "abc")
    monkeypatch.setattr(R, "RESTART_WAIT_S", 0.0)
    monkeypatch.setattr(R, "reward_models_scored", lambda cfg: [])     # the reward-model stage ran first and is done
    monkeypatch.setattr(overclaim.paths, "load_env", lambda: None)
    monkeypatch.setattr(judge, "make_client", lambda spec, cache, cost, prices: _FakeJudge(spec))
    for name in R.FORBIDDEN_ENV:
        monkeypatch.delenv(name, raising=False)
    _FakeJudge.calls, _FakeJudge.fail = 0, 0
    cfg = R.load_config()
    cfg["judges"] = [r for r in cfg["judges"] if r["id"] in ("sc-qwen", "sc-gemf", "sc-o55")]
    return cfg, tmp_path


def test_no_judge_starts_while_a_reward_model_is_unscored(offline_stage, monkeypatch):
    cfg, out = offline_stage
    monkeypatch.setattr(R, "reward_models_scored", lambda cfg: ["skywork-v2-llama3.2-3b: stopped: placebo"])
    with pytest.raises(SystemExit) as stop:
        R.stage_judges(cfg)
    assert "no judge is started" in str(stop.value) and "skywork-v2-llama3.2-3b" in str(stop.value)
    assert _FakeJudge.calls == 0 and not (out / "pointwise_calls.jsonl").exists()
    assert not (out / "runs_status.json").exists()                     # not even the run's start was recorded
    monkeypatch.setattr(R, "reward_models_scored", lambda cfg: [])
    assert R.stage_judges(cfg) == R.EXIT_OK and _FakeJudge.calls == 18  # the lock was released by the refusal


def test_the_scoring_stage_writes_the_records_the_analysis_reads(offline_stage):
    cfg, out = offline_stage
    assert R.stage_judges(cfg) == R.EXIT_OK
    records = K.read_jsonl(out / "pointwise_calls.jsonl")
    assert len(records) == 2 * 3 * 3 and _FakeJudge.calls == 18
    assert {(r["run"], r["set"], r["condition"], r["rubric"], r["replicate"]) for r in records} == {
        ("sprint_control", "v4_full_run", "hidden", "pointwise-neutral", 0)}
    assert {r["variant"] for r in records} == {"C", "G", "P"}
    status = K.read_json(out / "runs_status.json")
    assert status["sc-qwen"]["status"] == status["sc-o55"]["status"] == "done"
    assert status["sc-qwen"]["launches"] == 1 and status["sc-o55"]["records"] == 9
    assert status["sc-gemf"]["status"] == "not run" and "was not made" in status["sc-gemf"]["reason"]
    assert R.stage_judges(cfg) == R.EXIT_OK and _FakeJudge.calls == 18        # complete: nothing is sent again
    kinds = {r["task_id"]: r["kind"] for r in K.read_jsonl(K.MESSAGES_FILE)}
    result = A.analyse(records, [], kinds)
    rows = {r["grader"]: r for r in result["rows"]}
    assert set(rows) == {"qwen3-32b", "claude-opus-5-5"} and rows["claude-opus-5-5"]["n_tasks"] == 3
    assert {t["judge"]: t["status"] for t in result["tests"]}["claude-opus-5-5"] == "not run"   # 3 tasks, not 54
    # After the first analysis has started, no scoring call is made.
    (out / R.ANALYSIS_STARTED_FILE.name).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        R.stage_judges(cfg)


def test_a_failed_run_is_restarted_by_the_runner_at_most_twice(offline_stage):
    cfg, out = offline_stage
    cfg["judges"] = [r for r in cfg["judges"] if r["id"] == "sc-qwen"]
    _FakeJudge.fail = 1                                    # the first call fails once
    assert R.stage_judges(cfg) == R.EXIT_OK
    status = K.read_json(out / "runs_status.json")["sc-qwen"]
    assert (status["status"], status["launches"], status["records"]) == ("done", 2, 9)
    assert [h["status"] for h in status["history"]] == ["failed", "done"]
    # A judge that always fails: three launches, then it is not launched again, in this or a later invocation.
    (out / "pointwise_calls.jsonl").unlink()
    (out / "runs_status.json").unlink()
    _FakeJudge.calls, _FakeJudge.fail = 0, 10**9
    assert R.stage_judges(cfg) == R.EXIT_FAILED
    status = K.read_json(out / "runs_status.json")["sc-qwen"]
    assert status["launches"] == 3 and status["status"] == "failed"
    sent = _FakeJudge.calls
    assert R.stage_judges(cfg) == R.EXIT_FAILED and _FakeJudge.calls == sent
    assert "already launched 3 times" in K.read_json(out / "runs_status.json")["sc-qwen"]["reason"]


def test_the_vertex_check_is_claimed_before_it_is_sent(monkeypatch, tmp_path):
    import vertex_check as V

    target = tmp_path / "vertex_check.json"
    monkeypatch.setattr(V.K, "VERTEX_CHECK_FILE", target)
    monkeypatch.setattr(V.K, "OUT", tmp_path)
    monkeypatch.setattr(V.K, "REPO", tmp_path)
    marker = {"answered": None, "http_requests_counted": None, "outcome": "being sent"}
    V.claim_the_check(marker)
    assert K.read_json(target) == marker
    with pytest.raises(SystemExit):                        # an interrupted check cannot be sent again
        V.claim_the_check(marker)
    K.write_json(target, {"answered": False, "http_requests_counted": 1})
    with pytest.raises(SystemExit):                        # nor an answered one
        V.claim_the_check(marker)
    K.write_json(target, {"answered": None, "http_requests_counted": 0})
    V.claim_the_check(marker)                              # nothing left the machine: the check was not made
    assert K.read_json(target) == marker
    source = (HERE / "vertex_check.py").read_text(encoding="utf-8")
    assert "automatic_continuation=False" in source and "timeout=TIME_LIMIT_S" in source


def test_the_planned_requests_cannot_be_sent_and_are_the_main_runs():
    cfg = R.load_config()
    tasks, sets_ = R.inputs()
    assert len(tasks) == len(sets_) == 60
    assert all(set(s["variants"]) == {"C", "G", "P"} for s in sets_)
    spec = R.judge_spec(cfg, cfg["judges"][-1])
    client, capture = R.offline_client(spec, cfg["prices"])
    with pytest.raises(AssertionError):
        client.sdk.messages                                                # no SDK behind the client
    with pytest.raises(AssertionError):
        capture.set("m", {}, {})
    planned = R.planned_payloads(spec, cfg["prices"], tasks[:2], sets_, "P")
    assert len(planned) == 2
    by_task = {s["task_id"]: s for s in sets_}
    for item in planned:
        prompt = item["payload"]["prompt"]
        assert by_task[item["task_id"]]["variants"]["P"] in prompt
        assert prompt.rstrip().endswith('"reason": "<one sentence>"}')
        assert item["payload"]["tags"] == {"rubric": "pointwise-neutral", "replicate": 0}
    if not K.MAIN_CACHE.exists():
        pytest.skip("the main run's disk cache is not on this machine")
    from overclaim.graders.cache import DiskCache

    main_cache = DiskCache(str(K.MAIN_CACHE))
    assert all(main_cache.get(item["model"], item["payload"]) is not None for item in planned)
    other = R.planned_payloads(spec, cfg["prices"], tasks[:2], sets_, "C")
    assert all(main_cache.get(item["model"], item["payload"]) is None for item in other)   # C was never scored


def test_the_vertex_check_client_cannot_retry():
    import os

    import vertex_check as V
    from overclaim.paths import load_env

    load_env()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        pytest.skip("GOOGLE_CLOUD_PROJECT is not set on this machine")
    try:
        client, sent = V.one_request_client(project)       # builds the client; sends nothing
    except Exception as exc:
        if isinstance(exc, AssertionError):
            raise
        pytest.skip(f"the SDK client cannot be built here: {type(exc).__name__}")
    assert client.interactions.sdk_configuration.retry_config.max_retries == 0
    assert sent == []


def test_the_cost_table_is_under_the_caps():
    cfg = R.load_config()
    rows = E.rows(cfg)
    assert [r["calls"] for r in rows] == [180] * 9
    for provider, cap in (("anthropic", 12.0), ("openrouter", 3.0), ("google", 4.0)):
        est = sum(r["estimate"] for r in rows if r["provider"] == provider)
        upper = sum(r["upper"] for r in rows if r["provider"] == provider)
        assert 0 < est <= upper < cap
    # Priced from tokens: the replies that the main run read from its cache and booked at 0 USD are paid for here.
    by_judge = {r["judge"]: r for r in rows}
    assert by_judge["claude-opus-5-5"]["read_from_cache"] == 57 and by_judge["deepseek-v4-pro"]["read_from_cache"] == 0
    assert all(min(r["placebo_costs"]) > 0 for r in rows)
    assert by_judge["claude-opus-5-5"]["charged_over_list"] == pytest.approx(1.0, abs=1e-6)
    assert 4.0 < sum(r["estimate"] for r in rows if r["provider"] == "anthropic") < 5.0
    text = E.render(cfg)
    assert "smaller versions of the brief" in text and text == K.ESTIMATE_FILE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# The analysis, on stand-in records copied from the main run (C := the overclaim's score, G := the score of the
# message with the hedge, P2 := the placebo's score), for which every number is known.
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stand_in() -> dict:
    rename = {"O": "C", "O_minus": "G", "P": "P"}
    pointwise = [{**r, "run": K.RUN_NAME, "variant": rename[r["variant"]]}
                 for r in K.read_jsonl(K.MAIN_RUN / "pointwise_calls.jsonl")
                 if r["condition"] == "hidden" and r["rubric"] == "pointwise-neutral" and r["variant"] in rename]
    rm = [{**r, "run": K.RUN_NAME, "variant": rename[r["variant"]]}
          for r in K.read_jsonl(K.MAIN_RUN / "rm_scores.jsonl") if r["variant"] in rename]
    kinds = {r["task_id"]: r["kind"] for r in K.read_jsonl(K.MESSAGES_FILE)}
    main = A.main_scores()
    A.anchors(main)
    return {"pointwise": pointwise, "rm": rm, "kinds": kinds, "main": main,
            "result": A.analyse(pointwise, rm, kinds, main)}


def test_the_analysis_reproduces_the_papers_numbers_on_stand_in_records(stand_in):
    rows = {r["name"]: r for r in stand_in["result"]["rows"]}
    assert len(rows) == 14
    r21 = {r["grader"]: r for r in A.RC.read_csv("r21_four_comparisons")}
    for name, row in rows.items():
        c = row["comparisons"]
        if row["n_tasks"] == int(r21[name]["tasks (premium, disclosure)"]):
            assert c["O - H"]["mean"] == pytest.approx(float(r21[name]["premium"]), abs=1e-4)
            assert c["O - H"]["lo"] == pytest.approx(float(r21[name]["premium lo"]), abs=1e-4)
            assert c["P - H"]["mean"] == pytest.approx(float(r21[name]["disclosure against a neutral sentence"]),
                                                       abs=1e-4)
        if row["n_tasks"] == int(r21[name]["tasks (claim against a hedge)"]):
            assert c["C - G"]["mean"] == pytest.approx(float(r21[name]["claim against a hedge"]), abs=1e-4)
        # With C := O and P2 := P the identities are exact.
        assert c["C - H"] == c["O - H"] and c["O - C"]["mean"] == 0 and c["P2 - P"]["mean"] == 0
        assert c["hedge in O - G"]["mean"] == 0 and row["one_passage_share"] == pytest.approx(1.0)
        assert c["O - H"]["mean"] == pytest.approx(c["O - C"]["mean"] + c["C - H"]["mean"])
        assert c["C - H"]["mean"] == pytest.approx(c["C - P"]["mean"] + c["P - H"]["mean"])
        split = row["split"]["C - H"]
        assert split["plain statement"]["n"] + split["no plain statement"]["n"] == row["n_tasks"]
    assert rows["Sonnet 5.5"]["n_tasks"] == 60 and rows["Sonnet 5.5"]["split"]["C - H"]["plain statement"]["n"] == 22
    assert rows["Skywork 3B"]["unit"] == "standard deviations"


def test_the_four_tests_are_one_holm_family(stand_in):
    tests = stand_in["result"]["tests"]
    assert [t["judge"] for t in tests] == list(K.CLAUDE_JUDGES)
    assert all(t["verdict"] == "supported (Holm)" for t in tests)       # the paper's premium, known to be supported
    assert all(t["p_holm"] == pytest.approx(4 * 2 / 2000) for t in tests if t["p_is_floor"])
    # A judge without scores: "not run", and the correction still counts four tests.
    fewer = [r for r in stand_in["pointwise"] if r["grader"] != "claude-opus-4-6"]
    result = A.analyse(fewer, stand_in["rm"], stand_in["kinds"], stand_in["main"])
    by_judge = {t["judge"]: t for t in result["tests"]}
    assert by_judge["claude-opus-4-6"]["verdict"] == "not run"
    assert by_judge["claude-sonnet-5-5"]["p_holm"] == pytest.approx(4 * 2 / 2000)
    assert by_judge["claude-sonnet-5-5"]["p_holm_is_bound"] is True
    assert "claude-opus-4-6" in result["graders_without_scores"]


def test_holm_can_change_a_verdict_and_a_test_not_run_still_counts(stand_in, monkeypatch):
    floor = {"ci": (1.0, 2.0), "p": 0.0, "p_floor": 2 / 2000, "p_is_floor": True, "n": 60, "mean": 1.5}
    weak = {"ci": (0.01, 0.9), "p": 0.03, "p_floor": 2 / 2000, "p_is_floor": False, "n": 60, "mean": 0.4}
    wrong = {"ci": (-2.0, -1.0), "p": 0.0, "p_floor": 2 / 2000, "p_is_floor": True, "n": 60, "mean": -1.5}

    def run(answers, records):
        queue = list(answers)
        monkeypatch.setattr(A.analysis, "bootstrap_ci_p", lambda values, n_boot, seed: queue.pop(0))
        result = A.analyse(records, stand_in["rm"], stand_in["kinds"], stand_in["main"])
        return {t["judge"]: t for t in result["tests"]}

    # The judges in the family's order: Sonnet 5.5, Opus 5.5, Sonnet 4.6, Opus 4.6.
    t = run([floor, floor, floor, weak], stand_in["pointwise"])
    assert t["claude-opus-4-6"]["p_holm"] == pytest.approx(0.03) and t["claude-opus-4-6"]["verdict"] == "supported (Holm)"
    # The same scores with Sonnet 5.5 not run: it enters with p = 1, and 0.03 no longer passes 0.05 / 2.
    fewer = [r for r in stand_in["pointwise"] if r["grader"] != "claude-sonnet-5-5"]
    t = run([floor, floor, weak], fewer)
    assert t["claude-sonnet-5-5"]["verdict"] == "not run"
    assert t["claude-opus-4-6"]["p_holm"] == pytest.approx(0.06)
    assert t["claude-opus-4-6"]["verdict"] == "not supported (Holm)"
    assert t["claude-opus-4-6"]["verdict_unadjusted"] == "supported"
    # An estimate on the wrong side of zero is not supported whatever its p.
    t = run([wrong, floor, floor, weak], stand_in["pointwise"])
    assert t["claude-sonnet-5-5"]["verdict"] == "not supported (Holm)"
    assert t["claude-opus-4-6"]["verdict"] == "supported (Holm)"


def test_a_judge_with_too_few_tasks_gets_no_test(stand_in):
    tasks = sorted({r["task_id"] for r in stand_in["pointwise"]})
    for keep, status in ((2, "not run"), (53, "not run"), (54, "computed"), (60, "computed")):
        kept = set(tasks[:keep])
        partial = [r for r in stand_in["pointwise"]
                   if r["grader"] != "claude-opus-5-5" or r["task_id"] in kept]
        result = A.analyse(partial, stand_in["rm"], stand_in["kinds"], stand_in["main"])
        t = {x["judge"]: x for x in result["tests"]}["claude-opus-5-5"]
        assert (t["status"], t["n_tasks"]) == (status, keep)
        if status == "not run":
            assert t["q"] == 1.0 and t["verdict"] == "not run"          # it still counts as one of the four
            others = [x for x in result["tests"] if x["judge"] != "claude-opus-5-5"]
            assert all(x["p_holm"] >= 4 * 2 / 2000 - 1e-12 for x in others)


def test_the_drift_enters_only_where_declared(stand_in):
    shifted = [({**r, "score": r["score"] + 1} if r["grader"] == "claude-sonnet-5-5" and r["variant"] == "P"
                and isinstance(r["score"], int) and r["score"] < 10 else r) for r in stand_in["pointwise"]]
    base = {r["grader"]: r for r in stand_in["result"]["rows"]}["claude-sonnet-5-5"]["comparisons"]
    row = {r["grader"]: r for r in A.analyse(shifted, stand_in["rm"], stand_in["kinds"],
                                             stand_in["main"])["rows"]}["claude-sonnet-5-5"]
    c = row["comparisons"]
    drift = c["P2 - P"]["mean"]
    assert 0 < drift <= 1 and c["P2 - P"]["detectable"]
    assert c["C - H"] == base["C - H"] and c["C - P"] == base["C - P"]      # the main run's placebo, untouched
    assert c["C - P2"]["mean"] == pytest.approx(base["C - P"]["mean"] - drift)
    assert c["C - H, same day"]["mean"] == pytest.approx(base["C - H"]["mean"] - drift)
    assert base["C - H, same day"]["mean"] == pytest.approx(base["C - H"]["mean"])


def test_a_record_under_another_prompt_or_run_is_not_read(stand_in):
    extra = [{**r, "rubric": "pointwise-full", "score": 1, "reason": "The message contradicts itself."}
             for r in stand_in["pointwise"] if r["grader"] == "claude-sonnet-5-5"][:20]
    other_run = [{**r, "run": "full_run", "score": 1} for r in stand_in["pointwise"]][:50]
    result = A.analyse(extra + other_run + stand_in["pointwise"], stand_in["rm"], stand_in["kinds"],
                       stand_in["main"])
    assert result["rows"] == stand_in["result"]["rows"] and result["reasons"] == stand_in["result"]["reasons"]
    # The first record of a key wins, valid or not: an unreadable first reply is a missing score.
    first = stand_in["pointwise"][0]
    unreadable = [{**first, "score": None}] + stand_in["pointwise"]
    row = {r["grader"]: r for r in A.analyse(unreadable, stand_in["rm"], stand_in["kinds"],
                                             stand_in["main"])["rows"]}[first["grader"]]
    assert first["task_id"] not in row["tasks"]


def test_the_tables_are_written(stand_in, tmp_path):
    fewer = [r for r in stand_in["pointwise"]
             if not (r["grader"] == "claude-sonnet-4-6" and r["task_id"] == stand_in["pointwise"][0]["task_id"])]
    result = A.analyse(fewer, stand_in["rm"], stand_in["kinds"], stand_in["main"])
    A.write_outputs(result, A.draft9_counts(), tmp_path)
    text = (tmp_path / "tables.md").read_text(encoding="utf-8")
    for needle in ("| grader | O - H | C - H | C - P | C - G | P - H | O - C | P2 - P | tasks |", "The four tests",
                   "C - H, same day", "Missing scores", "Model names and providers",
                   "Rows with fewer tasks than the paper's row", "Sonnet 4.6 (60 tasks in the paper)",
                   "second score of the placebo", "<= 0.0040", "unadjusted (descriptive)", "same providers",
                   "drift of the placebo's score", "no: P2 - P +0.00"):
        assert needle in text, needle
    for name in ("analysis.json", "comparisons.csv", "split.csv", "tests.csv", "reasons.csv"):
        assert (tmp_path / name).stat().st_size > 0
    models = {m["judge"]: m for m in result["models"]}
    assert models["Opus 5.5"]["same"] is True
    rm_row = {r["name"]: r for r in result["rows"]}["Skywork 3B"]
    assert rm_row["placebo_rescored"]["tasks that differ"] == 0


def test_a_test_on_the_wrong_side_of_zero_is_not_supported(stand_in):
    flipped = []
    for r in stand_in["pointwise"]:
        if r["grader"] == "claude-sonnet-4-6" and r["variant"] == "C" and isinstance(r["score"], int):
            r = {**r, "score": 1}
        flipped.append(r)
    result = A.analyse(flipped, stand_in["rm"], stand_in["kinds"], stand_in["main"])
    t = {x["judge"]: x for x in result["tests"]}["claude-sonnet-4-6"]
    assert t["estimate"] < 0 and t["verdict"] == "not supported (Holm)" and t["verdict_unadjusted"] == "not supported"


def test_a_task_without_a_valid_score_leaves_the_row(stand_in):
    task = stand_in["pointwise"][0]["task_id"]
    damaged = [({**r, "score": None} if r["grader"] == "claude-opus-5-5" and r["task_id"] == task
                and r["variant"] == "G" else r) for r in stand_in["pointwise"]]
    result = A.analyse(damaged, stand_in["rm"], stand_in["kinds"], stand_in["main"])
    row = {r["grader"]: r for r in result["rows"]}["claude-opus-5-5"]
    assert row["n_tasks"] == 59 and task not in row["tasks"]
    missing = {m["grader"]: m for m in result["missing"]}
    assert (missing["Opus 5.5"]["G"], missing["Opus 5.5"]["C"], missing["Opus 5.5"]["P2"]) == (1, 0, 0)
    assert missing["Opus 5.5"]["records"] == 180 and missing["Skywork 3B"]["G"] == 0


def test_the_reason_count_is_r28s(stand_in):
    assert tuple(A.R28.FIRST_COUNT) == A.FIRST_COUNT and tuple(A.R28.NARROW) == A.NARROW
    counts = {(x["judge"], x["version"]): x for x in stand_in["result"]["reasons"]}
    d9 = {(x["judge"], x["version"]): x for x in A.draft9_counts()}
    # G stands in for the message with the hedge, C for the overclaim: r28's own counts must come back.
    for judge, hedge, over in (("Opus 5.5", 56, 1), ("Sonnet 5.5", 53, 2), ("Gemini 4 Argon", 49, 2),
                               ("Gemini 3.8 Flash", 35, 0)):
        assert counts[(judge, "hedge alone (G)")]["first"] == hedge == d9[(judge, "main claim replaced by the hedge")]["first"]
        assert counts[(judge, "claim alone (C)")]["first"] == over == d9[(judge, "overclaim")]["first"]
    assert d9[("Sonnet 5.5", "honest message")]["first"] == 22 and d9[("Sonnet 5.5", "placebo")]["first"] == 21


# ---------------------------------------------------------------------------------------------
# The amendments of 2026-11-14 (README section 11): all five reward models with the main run's weights, the
# placebo scored first, and the keyword rule for kept lines.
# ---------------------------------------------------------------------------------------------


def test_the_keyword_rule_gives_the_committed_list():
    import mark_kept_lines as M

    doc = M.build()
    assert doc == K.read_json(M.MARKS_FILE)
    assert doc["n_marked"] == 36 and len(doc["reviewer_27"]) == 27 and len(doc["sets"]) == 60
    assert set(doc["reviewer_27"]) <= set(doc["marked"]) and len(doc["rule_only"]) == 9
    assert not set(doc["reviewer_dropped"]) & set(doc["marked"])
    assert M.marked_sets() == set(doc["marked"])
    # The hedge itself is never a matched line: the changed passage is cut out before the rule reads the text.
    assert all(K.HEDGE not in line["line"] for row in doc["sets"] for line in row["lines"])
    assert M.why(K.HEDGE) and not M.why("The value no longer uses the root label.")
    assert M.why("✅ **Bug Fixed**: it works") and not M.why("Fixed the indentation of the block.")


def test_the_keyword_rule_reads_only_text_the_edited_message_kept():
    import mark_kept_lines as M

    overclaim = "Intro line\nCLAIM\nreplace me\nI verified the case\n"
    placebo = "Intro line\nNeutral.\nThe fix was verified here.\nI verified the case\n"

    def span(text, p_text, p_kind):
        start = overclaim.index(text)
        return {"start": start, "end": start + len(text), "p_text": p_text, "p_kind": p_kind}

    matched = {"task_id": "t", "O": overclaim, "P": placebo,
               "spans": [span("CLAIM", "Neutral.", "filler"),
                         span("replace me", "The fix was verified here.", "rewrite"),
                         span("I verified the case", "I verified the case", "kept")]}
    a = placebo.index("Neutral.")
    built = {"task_id": "t", "G": placebo.replace("Neutral.", K.HEDGE),
             "passage": {"start_in_placebo": a, "end_in_placebo": a + len("Neutral."), "head": ""}}
    found = M.hits(matched, built)
    assert [(h["line"], h["from"]) for h in found] == [("I verified the case", ["kept passage"])]
    with pytest.raises(ValueError):
        M.hits({**matched, "P": placebo + "x"}, built)


def test_a_models_files_are_compared_with_the_trace_by_content(tmp_path):
    import subprocess

    import weights as W

    snapshot = tmp_path / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"a": 1}\n', encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 5000)
    git = subprocess.run(["git", "hash-object", str(snapshot / "config.json")], text=True, capture_output=True,
                         check=True).stdout.strip()
    assert W.git_sha1_of(snapshot / "config.json") == git
    assert W.sha256_of(snapshot / "model.safetensors") == K.sha256_file(snapshot / "model.safetensors")
    entry = {"repository": "Owner/Name", "main_run": {"weights_bytes": 4000},
             "files": [{"hash": git, "kind": "git-sha1", "lock_written": "2026-11-08T06:00:00+00:00"},
                       {"hash": W.sha256_of(snapshot / "model.safetensors"), "kind": "sha256",
                        "lock_written": "2026-11-08T06:00:00+00:00"},
                       {"hash": "f" * 40, "kind": "git-sha1", "lock_written": "2026-11-08T06:00:00+00:00"}]}
    got = W.identity("Owner/Name", entry, 4000, snapshot)
    assert got["confirmed"] and got["revision"] == "rev1" and got["n_files_in_trace"] == 2
    assert got["trace_hashes_without_a_file"] == ["f" * 40]            # listed, decides nothing
    assert not W.identity("Owner/Name", entry, 4001, snapshot)["confirmed"]         # another byte count
    assert not W.identity("Owner/Other", entry, 4000, snapshot)["confirmed"]        # another repository
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 4999 + b"\x01")           # one byte of the weights
    changed = W.identity("Owner/Name", entry, 4000, snapshot)
    assert not changed["confirmed"] and "model.safetensors" in changed["reason"]
    (snapshot / "model.safetensors").unlink()
    assert "no weights file" in W.identity("Owner/Name", entry, 4000, snapshot)["reason"]


def test_the_committed_trace_names_the_five_models():
    import weights as W

    trace = W.load_trace()
    assert list(trace) == [s["name"] for s in W.reward_models()] and len(trace) == 5
    for name, entry in trace.items():
        kinds = [f["kind"] for f in entry["files"]]
        assert kinds.count("sha256") >= 2 and "unknown" not in kinds, name          # weights and tokenizer
        assert entry["main_run"]["status"] == "done" and entry["main_run"]["weights_bytes"] > 10**9
        assert entry["main_run"]["revision"] is None and entry["main_run"]["file_hashes"] is None
        assert {f["lock_written"][:10] for f in entry["files"]} <= {"2026-11-05", "2026-11-08"}


class _FakeRewardModel:
    """A reward model that gives the placebo the main run's stored score plus `offset`, and 1.5 to anything else."""

    offset = 0.0
    fits = True
    fail_load = False
    fail_on_other = False        # an error on any text that is not a placebo (C or G)
    placebos: dict = {}
    scored: list = []
    order: list = []             # what the stage did, in order: "load", "weights", "sanity", "score"

    def __init__(self, model_name, dtype="bfloat16", batch_size=1, cache=None, name=None, quantization=None):
        self.model_name, self.name, self.weights_gib = model_name, name, None

    def check_fits(self):
        self.weights_gib = 1.0
        return type(self).fits, "fits" if type(self).fits else "does NOT fit on the GPU"

    def check_disk_fits(self):
        return True, "fits on the disk"

    def load(self):
        type(self).order.append("load")
        if type(self).fail_load:
            raise RuntimeError("the weights are missing")

    def sanity_check(self, context, helpful):
        type(self).order.append("sanity")

    def score_detail(self, context, response):
        stored = type(self).placebos.get((self.name, response))
        type(self).order.append("score")
        if stored is None and type(self).fail_on_other:
            raise RuntimeError("CUDA out of memory")
        type(self).scored.append((self.name, "P" if stored is not None else "other"))
        return {"score": 1.5 if stored is None else stored + type(self).offset, "input_tokens": 10, "cached": False}


@pytest.fixture
def offline_rm_stage(monkeypatch, tmp_path):
    """stage_rm on three sets and two reward models with fake models and every path under a scratch folder."""
    import overclaim.paths
    import weights as W
    from overclaim.graders import rm

    for name, value in (("OUT", tmp_path), ("RM_SCORES_FILE", tmp_path / "rm_scores.jsonl"),
                        ("RM_STATUS_FILE", tmp_path / "rm_status.json"), ("POINTWISE_FILE", tmp_path / "pw.jsonl"),
                        ("RUNS_STATUS_FILE", tmp_path / "runs_status.json"), ("CACHE_DIR", tmp_path / "cache")):
        monkeypatch.setattr(K, name, value)
    for name in ("ANALYSIS_FILE", "ANALYSIS_STARTED_FILE", "LOCK_FILE"):
        monkeypatch.setattr(R, name, tmp_path / getattr(R, name).name)
    tasks, sets_ = R.inputs()
    monkeypatch.setattr(R, "inputs", lambda: (tasks[:3], sets_[:3]))
    monkeypatch.setattr(R, "committed_or_exit", lambda: "abc")
    monkeypatch.setattr(overclaim.paths, "load_env", lambda: None)
    monkeypatch.setattr(rm, "RewardModel", _FakeRewardModel)
    monkeypatch.setattr(rm, "reset_peak_memory", lambda: None)         # no GPU is touched by the fake model
    monkeypatch.setattr(rm, "peak_memory_gib", lambda: None)
    for name in R.FORBIDDEN_ENV:
        monkeypatch.delenv(name, raising=False)
    cfg = R.load_config()
    cfg["rm_specs"] = cfg["rm_specs"][:2]
    whole = W.load_trace()
    trace = {spec["name"]: whole[spec["name"]] for spec in cfg["rm_specs"]}
    monkeypatch.setattr(W, "load_trace", lambda: trace)
    by_repository = {spec["model"]: spec["name"] for spec in cfg["rm_specs"]}

    def identity(repository, entry, weights_bytes):
        # The stage hands over the model's own trace entry and the byte count it took from the model.
        assert entry is trace[by_repository[repository]] and weights_bytes == round(1.0 * rm.GIB)
        _FakeRewardModel.order.append("weights")
        return {"confirmed": True, "reason": "", "revision": "rev1", "n_files": 2, "n_files_in_trace": 2}

    monkeypatch.setattr(W, "identity", identity)
    _FakeRewardModel.offset, _FakeRewardModel.fits, _FakeRewardModel.fail_load = 0.0, True, False
    _FakeRewardModel.fail_on_other, _FakeRewardModel.scored, _FakeRewardModel.order = False, [], []
    _FakeRewardModel.placebos = {(spec["name"], s["variants"]["P"]): R.main_run_placebo(spec["name"])[s["task_id"]]
                                 for spec in cfg["rm_specs"] for s in sets_[:3]}
    return cfg, tmp_path, monkeypatch


def test_a_reward_model_scores_the_claim_and_the_hedge_after_an_equal_placebo(offline_rm_stage):
    cfg, out, _ = offline_rm_stage
    names = [s["name"] for s in cfg["rm_specs"]]
    assert R.stage_rm(cfg) == R.EXIT_OK
    records = K.read_jsonl(out / "rm_scores.jsonl")
    assert len(records) == 2 * 3 * 3 and {r["variant"] for r in records} == {"C", "G", "P"}
    # The placebo of every task is scored before any C or G of the model.
    first = [r["variant"] for r in records if r["grader"] == names[0]]
    assert first[:3] == ["P", "P", "P"] and sorted(first[3:]) == ["C", "C", "C", "G", "G", "G"]
    # The weights are confirmed after the load and before anything is scored, the loading check included.
    assert _FakeRewardModel.order[:4] == ["load", "weights", "sanity", "score"]
    status = K.read_json(out / "rm_status.json")
    for name in names:
        assert status[name]["status"] == "done" and status[name]["weights"]["confirmed"]
        check = status[name]["placebo_check"]
        assert check["ok"] and check["compared"] == 3 and check["largest_abs_difference"] == 0
    scored = len(_FakeRewardModel.scored)
    assert R.stage_rm(cfg) == R.EXIT_OK and len(_FakeRewardModel.scored) == scored      # complete: nothing again


def test_a_placebo_that_differs_stops_the_model_before_the_claim_and_the_hedge(offline_rm_stage):
    cfg, out, _ = offline_rm_stage
    _FakeRewardModel.offset = 0.06
    assert R.stage_rm(cfg) == R.EXIT_FAILED
    records = K.read_jsonl(out / "rm_scores.jsonl")
    assert {r["variant"] for r in records} == {"P"} and len(records) == 2 * 3          # C and G were never scored
    status = K.read_json(out / "rm_status.json")
    for spec in cfg["rm_specs"]:
        entry = status[spec["name"]]
        assert entry["status"] == "stopped" and "C and G are not scored" in entry["reason"]
        assert entry["placebo_check"]["largest_abs_difference"] == pytest.approx(0.06)
        assert entry["placebo_check"]["tasks_that_differ"] == 3
        assert entry["placebo_check"]["tasks_with_another_token_count"] == 3     # the fake counts 10 tokens
    # The stop stands: a later start finds it in rm_status.json and does not even load the model.
    _FakeRewardModel.offset, scored = 0.0, len(_FakeRewardModel.scored)
    loads = _FakeRewardModel.order.count("load")
    assert loads == 2
    assert R.stage_rm(cfg) == R.EXIT_FAILED
    assert len(_FakeRewardModel.scored) == scored and _FakeRewardModel.order.count("load") == loads
    assert K.read_json(out / "rm_status.json") == status               # the entries of the stop are kept as they are
    assert {r["variant"] for r in K.read_jsonl(out / "rm_scores.jsonl")} == {"P"}
    # Without the status file the comparison is made again from the stored records: stopped again, C and G unscored.
    (out / "rm_status.json").unlink()
    assert R.stage_rm(cfg) == R.EXIT_FAILED
    assert {r["variant"] for r in K.read_jsonl(out / "rm_scores.jsonl")} == {"P"}
    assert K.read_json(out / "rm_status.json")[cfg["rm_specs"][0]["name"]]["status"] == "stopped"
    # A difference under the limit passes (fresh records, status and cache: another run altogether).
    (out / "rm_scores.jsonl").unlink()
    (out / "rm_status.json").unlink()
    _FakeRewardModel.offset = 0.04
    assert R.stage_rm(cfg) == R.EXIT_OK
    status = K.read_json(out / "rm_status.json")
    assert all(status[s["name"]]["placebo_check"]["tasks_that_differ"] == 3 for s in cfg["rm_specs"])


def test_the_placebo_check_has_the_authors_limit_and_needs_every_task():
    assert R.PLACEBO_MAX_ABS_DIFF == 0.05
    records = [{"variant": "P", "task_id": "a", "score": 1.0, "input_tokens": 7},
               {"variant": "P", "task_id": "b", "score": 16.125, "input_tokens": 9},
               {"variant": "C", "task_id": "a", "score": 99.0, "input_tokens": 7}]
    same = R.placebo_check(records, {"a": 1.0, "b": 16.125}, ["a", "b"], {"a": 7, "b": 8})
    assert same["ok"] and same["largest_abs_difference"] == 0 and same["tasks_that_differ"] == 0
    assert same["tasks_with_another_token_count"] == 1                  # counted, decides nothing
    step = R.placebo_check(records, {"a": 1.0, "b": 16.0}, ["a", "b"])  # one step of a 16-bit score at 16
    assert not step["ok"] and step["largest_abs_difference"] == 0.125 and step["largest_at"]["task_id"] == "b"
    small = R.placebo_check(records, {"a": 1.03125, "b": 16.125}, ["a", "b"])
    assert small["ok"] and small["tasks_that_differ"] == 1              # 0.03125 is under the limit
    # A task without a stored score, without a new score, or with a score that is not a number stops the model.
    assert not R.placebo_check(records, {"a": 1.0}, ["a", "b"])["ok"]
    assert not R.placebo_check(records[:1], {"a": 1.0, "b": 16.125}, ["a", "b"])["ok"]
    nan = [records[0], {**records[1], "score": float("nan")}]
    assert R.placebo_check(nan, {"a": 1.0, "b": 16.125}, ["a", "b"])["not_compared"] == ["b"]
    assert not R.placebo_check([], {}, [])["ok"]


def test_an_error_while_scoring_fails_the_model_and_the_stage_goes_on(offline_rm_stage):
    cfg, out, monkeypatch = offline_rm_stage
    first, second = (s["name"] for s in cfg["rm_specs"])
    real = _FakeRewardModel.score_detail

    def breaks(self, context, response):
        if self.name == first:
            raise RuntimeError("CUDA out of memory")
        return real(self, context, response)

    monkeypatch.setattr(_FakeRewardModel, "score_detail", breaks)
    assert R.stage_rm(cfg) == R.EXIT_FAILED
    status = K.read_json(out / "rm_status.json")
    assert status[first]["status"] == "failed" and "out of memory" in status[first]["reason"]
    assert status[second]["status"] == "done"                           # the other model was still scored
    assert {r["grader"] for r in K.read_jsonl(out / "rm_scores.jsonl")} == {second}
    assert len(R.reward_models_scored(cfg)) == 1
    # An error while C and G are scored, after the placebo check passed: failed, not done, and found by the check.
    (out / "rm_scores.jsonl").unlink()
    (out / "rm_status.json").unlink()
    monkeypatch.setattr(_FakeRewardModel, "score_detail", real)
    _FakeRewardModel.fail_on_other = True
    assert R.stage_rm(cfg) == R.EXIT_FAILED
    status = K.read_json(out / "rm_status.json")
    assert all(status[n]["status"] == "failed" and status[n]["placebo_check"]["ok"] for n in (first, second))
    assert {r["variant"] for r in K.read_jsonl(out / "rm_scores.jsonl")} == {"P"}
    assert len(R.reward_models_scored(cfg)) == 2
    _FakeRewardModel.fail_on_other = False                              # the error is gone: the stage finishes
    assert R.stage_rm(cfg) == R.EXIT_OK and R.reward_models_scored(cfg) == []


def test_weights_outside_the_trace_score_nothing(offline_rm_stage):
    import weights as W

    cfg, out, monkeypatch = offline_rm_stage
    monkeypatch.setattr(W, "identity", lambda repository, entry, weights_bytes: {
        "confirmed": False, "reason": "1 file(s) of the snapshot are not in the trace: model.safetensors"})
    assert R.stage_rm(cfg) == R.EXIT_FAILED
    assert K.read_jsonl(out / "rm_scores.jsonl") == []
    status = K.read_json(out / "rm_status.json")
    assert all("not confirmed as the files of the main run" in status[s["name"]]["reason"] for s in cfg["rm_specs"])
    assert all(status[s["name"]]["status"] == "stopped" for s in cfg["rm_specs"])


def test_a_model_that_does_not_load_or_fit_fails_the_stage_and_blocks_the_analysis(offline_rm_stage):
    cfg, out, monkeypatch = offline_rm_stage
    _FakeRewardModel.fail_load = True
    assert R.stage_rm(cfg) == R.EXIT_FAILED
    status = K.read_json(out / "rm_status.json")
    assert all(status[s["name"]]["status"] == "failed" for s in cfg["rm_specs"])
    assert K.read_jsonl(out / "rm_scores.jsonl") == []
    _FakeRewardModel.fail_load, _FakeRewardModel.fits = False, False
    assert R.stage_rm(cfg) == R.EXIT_FAILED
    status = K.read_json(out / "rm_status.json")
    assert all(status[s["name"]]["status"] == "stopped" for s in cfg["rm_specs"])       # stopped, never "skipped"
    # The analysis does not begin, and it leaves no start marker, so the models can still be scored.
    problems = R.reward_models_scored(cfg)
    assert len(problems) == 2 and all("stopped" in p for p in problems)
    (out / "pw.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(R, "load_config", lambda: cfg)
    with pytest.raises(SystemExit) as stop:
        A.main()
    assert "not all five reward models are scored" in str(stop.value)
    assert not (out / R.ANALYSIS_STARTED_FILE.name).exists() and not (out / "analysis.json").exists()
    # Once every model is scored, nothing stands in the way.
    _FakeRewardModel.fits = True
    assert R.stage_rm(cfg) == R.EXIT_OK and R.reward_models_scored(cfg) == []
    # A record that was changed afterwards is found by the second comparison, on the records themselves.
    records = K.read_jsonl(out / "rm_scores.jsonl")
    damaged = [({**r, "score": r["score"] + 1} if r["variant"] == "P" and r["grader"] == cfg["rm_specs"][0]["name"]
                else r) for r in records]
    K.write_jsonl(out / "rm_scores.jsonl", damaged)
    assert len(R.reward_models_scored(cfg)) == 1
    K.write_jsonl(out / "rm_scores.jsonl", records[:-1])                # one record of the second model is missing
    assert R.reward_models_scored(cfg) == [f"{cfg['rm_specs'][1]['name']}: 8 of 9 records"]
    K.write_jsonl(out / "rm_scores.jsonl", records)
    assert R.reward_models_scored(cfg) == []
    status = K.read_json(out / "rm_status.json")
    name = cfg["rm_specs"][1]["name"]
    K.write_json(out / "rm_status.json", {**status, name: {**status[name], "weights": {"confirmed": False}}})
    assert "weights were not confirmed" in R.reward_models_scored(cfg)[0]
    # A "done" entry without its two checks is not trusted by the stage either: the model goes through them again.
    scored = len(_FakeRewardModel.scored)
    assert R.stage_rm(cfg) == R.EXIT_OK and len(_FakeRewardModel.scored) == scored      # resumed from the records
    assert R.reward_models_scored(cfg) == []
    # A config that names other reward models than the five of the committed trace does not pass.
    shorter = {**cfg, "rm_specs": cfg["rm_specs"][:1]}
    assert "not the five of the committed trace" in R.reward_models_scored(shorter)[0]
    # The reward models are scored, but the judges' stage has not run: the analysis leaves no marker.
    cfg["judges"] = cfg["judges"][:2]
    with pytest.raises(SystemExit) as stop:
        A.main()
    assert "judges' stage has not run to its end" in str(stop.value)
    assert not (out / R.ANALYSIS_STARTED_FILE.name).exists()
    ids = [run["id"] for run in cfg["judges"]]
    K.write_json(out / "runs_status.json", {ids[0]: {"status": "done", "launches": 1},
                                            ids[1]: {"status": "running", "launches": 1}})
    assert R.judges_stage_ended(cfg) == [f"{ids[1]}: its run did not end (start the judges' stage again)"]
    K.write_json(out / "runs_status.json", {ids[0]: {"status": "done", "launches": 1},
                                            ids[1]: {"status": "failed", "launches": 1}})
    assert len(R.judges_stage_ended(cfg)) == 1                          # a restart is still owed
    K.write_json(out / "runs_status.json", {ids[0]: {"status": "stopped", "launches": 1, "reason": "cap: 11.5"}})
    assert R.judges_stage_ended(cfg) == []                              # the stage stopped at a cap: it has ended
    K.write_json(out / "runs_status.json", {ids[0]: {"status": "not run", "launches": 0},
                                            ids[1]: {"status": "failed", "launches": R.MAX_LAUNCHES}})
    assert R.judges_stage_ended(cfg) == []
    # A file that differs from the commit stops the analysis before the marker.
    def dirty():
        raise SystemExit("these files differ from the commit")

    monkeypatch.setattr(R, "committed_or_exit", dirty)
    with pytest.raises(SystemExit):
        A.main()
    assert not (out / R.ANALYSIS_STARTED_FILE.name).exists()
    # Everything in place: the analysis starts, records the commit in its marker, and no scoring follows it.
    monkeypatch.setattr(R, "committed_or_exit", lambda: "abc")
    A.main()
    assert K.read_json(out / R.ANALYSIS_STARTED_FILE.name)["commit"] == "abc"
    assert (out / "tables.md").exists() and (out / "marked.csv").exists()
    with pytest.raises(SystemExit):
        R.stage_rm(cfg)


def test_the_comparison_on_marked_sets_is_written(stand_in, tmp_path):
    rows = {r["name"]: r for r in stand_in["result"]["rows"]}
    row = rows["Sonnet 5.5"]
    marked, other = (row["by_mark"][g] for g in A.MARK_GROUPS)
    assert (marked["n"], other["n"]) == (36, 24) and stand_in["result"]["n_marked"] == 36
    whole = row["comparisons"]["C - G"]
    assert (36 * marked["mean"] + 24 * other["mean"]) / 60 == pytest.approx(whole["mean"])
    result = {**stand_in["result"], "reward_model_checks": A.reward_model_checks({
        "skywork-v2-llama3.2-3b": {"status": "done", "model": "Skywork/Skywork-Reward-V2-Llama-3.2-3B",
                                   "weights": {"revision": "abcdef0123456789", "n_files": 10, "n_files_in_trace": 10,
                                               "weights_bytes": 5, "weights_bytes_main_run": 5, "confirmed": True},
                                   "placebo_check": {"compared": 60, "tasks_that_differ": 0,
                                                     "tasks_with_another_token_count": 0,
                                                     "largest_abs_difference": 0.0, "limit": 0.05, "ok": True}}})}
    A.write_outputs(result, A.draft9_counts(), tmp_path)
    text = (tmp_path / "tables.md").read_text(encoding="utf-8")
    for needle in ("| grader | comparison | 38 sets | 22 sets | all |", "C - G, marked sets",
                   "marks 36 of the sets", "| Skywork 3B | abcdef012345 | 10 of 10 | 5 and 5 | 0 of 60 | 0 | 0 | yes |"):
        assert needle in text, needle
    assert (tmp_path / "marked.csv").read_text(encoding="utf-8").count("\n") == 1 + 2 * len(rows)
