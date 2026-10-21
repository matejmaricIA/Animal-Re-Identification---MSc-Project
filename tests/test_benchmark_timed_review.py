"""Behavioural regressions for the N3 timed-review trial: traced replays, task sampling and re-costing."""
import json
import zlib

import numpy as np
import pandas as pd
import pytest

import reid_demo.benchmark_nonmatch_pruning as nm
import reid_demo.benchmark_pruning_safeguards as ps
import reid_demo.benchmark_timed_review as tr
from reid_demo.benchmark_review_simulation import Reviewer, bundle_keys, ordered_units, simulate_catalogue
import tools.build_czechlynx_timed_review_tasks as builder
from test_benchmark_gated_matching import context
from test_benchmark_nonmatch_pruning import APPEARANCE, by_similarity, setting
from test_benchmark_pruning_safeguards import TopSimilarity


# A random 80-photo world in 2-photo bundles that exercises automation, pruning, second comparisons,
# disagreement checks, duplicate checks and merge confirmations (as in the N2 tests).
RNG = np.random.default_rng(7)
CENTRES = RNG.normal(size=(8, 6))
CODES = RNG.integers(0, 8, 80)
VECTORS = CENTRES[CODES] + RNG.normal(scale=.8, size=(80, 6))
BUNDLES = [list(range(i, i + 2)) for i in range(0, 80, 2)]
WORLD = dict(order="pmatch", eps=.01, dup_every=5, dup_r=3, double_c=2, disagree_delta=.05, tau=.99,
             merge_confirm=True)
DATES = [str(d.date()) for d in pd.date_range("2020-01-01", periods=40) for _ in range(2)]


def simulate(cls, *, expert=None, second=None, extra=None, **kw):
    ctx = context(VECTORS.copy())
    fold = {int(c): 0 for c in CODES}
    return cls(BUNDLES, np.asarray(CODES), nm.PairContext(ctx), {0: TopSimilarity()}, ["s1"], {0: by_similarity()},
               APPEARANCE, fold, setting(**{**WORLD, **kw}), expert or Reviewer(0), second or Reviewer(0),
               **(extra or {})).run()


def w1_world():
    frame = pd.DataFrame({"image_id": [f"i{i}" for i in range(80)], "source": "s",
                          "trap_id": [f"t{i // 2}" for i in range(80)], "timestamp": DATES})
    units = ordered_units(frame, CODES, bundle_keys(frame, "trap_day"))
    scores = context(VECTORS.copy()).scores
    return units, scores


def test_traced_simulation_equals_safeguard_simulation_with_noisy_reviewers():
    base = simulate(ps.SafeguardSimulation, expert=Reviewer(.1, 1), second=Reviewer(.1, 2))
    traced = simulate(tr.TracedSafeguardSimulation, expert=Reviewer(.1, 1), second=Reviewer(.1, 2))
    np.testing.assert_array_equal(base.catalogue.entry_of, traced.catalogue.entry_of)
    assert base.catalogue.founder == traced.catalogue.founder
    assert dict(base.ledger.counts) == dict(traced.ledger.counts)
    assert [r["outcome"] for r in base.placements] == [r["outcome"] for r in traced.placements]
    assert base.duplicate_log == traced.duplicate_log and base.blocked_merges == traced.blocked_merges
    assert len(base.auto_log) == len(traced.auto_log) == len(traced.autos)
    # The trace accounts for every ledger decision of the simulation itself.
    traced_kinds = [d["kind"] for t in traced.parts for d in t["decisions"]] + [q["kind"] for q in traced.pairs]
    for kind, n in traced.ledger.counts.items():
        if kind != "extra_animal":
            assert traced_kinds.count(kind) == n, kind
    assert traced.autos and traced.pairs and any(t["second"] for t in traced.parts)  # Paths exercised.


def test_kind_reviewers_replace_only_their_kind():
    always_no = type("No", (), {"same": lambda self, truth: False})()
    sim = simulate(tr.TracedSafeguardSimulation, extra={"kind_reviewers": {"second_comparison": always_no}})
    seconds = [d for t in sim.parts for d in t["decisions"] if d["kind"] == "second_comparison"]
    assert seconds and not any(d["answer"] for d in seconds)
    assert all(d["answer"] == d["truth"] for t in sim.parts for d in t["decisions"] if d["kind"] in tr.FIRST_PASS)
    with pytest.raises(ValueError):
        simulate(tr.TracedSafeguardSimulation, extra={"kind_reviewers": {"not_a_kind": always_no}})


def test_traced_w1_equals_simulate_catalogue():
    units, scores = w1_world()
    labels, effort = simulate_catalogue(units, scores, 80, Reviewer(0.0), ranked=True)
    cat, ledger, parts = tr.traced_w1(units, scores, 80)
    np.testing.assert_array_equal(labels, cat.entry_of)
    assert ledger["rank1_comparison"] + ledger["further_comparison"] == effort.counts["catalogue_comparison"]
    assert ledger["extra_animal"] == effort.counts["extra_animal"]
    assert ledger["rank1_comparison"] == sum(bool(t["order"]) for t in parts)
    noisy = Reviewer(.2, 5)
    labels, effort = simulate_catalogue(units, scores, 80, Reviewer(.2, 5), ranked=True)
    cat, ledger, _ = tr.traced_w1(units, scores, 80, reviewers={k: noisy for k in tr.FIRST_PASS})
    np.testing.assert_array_equal(labels, cat.entry_of)
    assert sum(ledger.values()) == effort.decisions


def test_catalogue_reconstructs_earlier_states_across_merges():
    cat = tr.TracedCatalogue(5)
    a = cat.create([0], 10)
    b = cat.create([1], 11)
    cat.add([2], b)
    cat.merge(a, b)
    cat.add([3], 0)
    assert cat.owners_at(1).tolist() == [0, -1, -1, -1, -1]
    assert cat.owners_at(3).tolist() == [0, 1, 1, -1, -1]
    assert cat.owners_at(4).tolist() == [0, 0, 0, -1, -1]
    assert cat.owners_at(5).tolist() == [0, 0, 0, 0, -1]
    assert cat.members_at(1, 3).tolist() == [1, 2] and cat.founder_of_uid == {0: 10, 1: 11}


def test_strata_and_counts_sum_back_to_the_ledgers():
    sim = simulate(tr.TracedSafeguardSimulation)
    units, scores = w1_world()
    _, w1_ledger, w1_parts = tr.traced_w1(units, scores, 80)
    counts = tr.decision_counts(w1_parts, w1_ledger, sim, audit_glances=7)
    n2 = tr.kind_totals(counts, "N2")
    for kind in tr.KINDS:
        want = 7 if kind == "audit_glance" else int(sim.ledger.counts.get(kind, 0))
        assert n2[kind] == want, kind
    assert tr.kind_totals(counts, "W1") == {k: int(w1_ledger.get(k, 0)) for k in tr.KINDS}
    assert {s for (a, s, _) in counts if a == "W1"} <= {"K1", "K2", "K3", "N", "X"}
    pops = tr.populations(w1_parts, sim, 7)
    assert pops[("N2", "A")] == 7
    assert pops.get(("N2", "P"), 0) == sum(q["kind"] == "pair_question" for q in sim.pairs)


def test_placement_stratum_boundaries():
    def trace(steps, outcome="matched_correct"):
        return {"outcome": outcome, "decisions": [{"kind": "rank1_comparison" if s == 1 else "further_comparison",
                                                   "step": s} for s in range(1, steps + 1)]}
    assert tr.placement_stratum("W1", trace(1)) == "K1"
    assert tr.placement_stratum("W1", trace(10)) == "K2"
    assert tr.placement_stratum("W1", trace(11)) == "K3"
    assert tr.placement_stratum("N2", trace(11)) == "K2"
    assert tr.placement_stratum("N2", trace(3, "new")) == "N"
    assert tr.placement_stratum("N2", trace(3, "rescued_correct")) == "N"


def world_units():
    sim = simulate(tr.TracedSafeguardSimulation)
    units, scores = w1_world()
    w1_cat, _, w1_parts = tr.traced_w1(units, scores, 80)
    return sim, w1_cat, scores, tr.candidate_units(w1_parts, sim, CODES, DATES)


def test_sampling_is_deterministic_and_query_photos_never_repeat():
    _, _, _, units = world_units()
    allocation = {("W1", "N"): 3, ("N2", "N"): 2, ("N2", "S"): 2, ("N2", "S+"): 1, ("W1", "K2"): 3,
                  ("N2", "A"): 2, ("N2", "A+"): 1, ("W1", "K1"): 4, ("N2", "K1"): 3, ("N2", "P"): 2, ("N2", "P+"): 2}
    first, short, used = tr.sample_units(units, allocation, np.random.default_rng(3))
    again, _, _ = tr.sample_units(units, allocation, np.random.default_rng(3))
    ids = lambda chosen: {k: [(u["type"], u["source_index"]) for u in us] for k, us in chosen.items()}
    assert ids(first) == ids(again)
    seen = set()
    for us in first.values():
        for u in us:
            query = set(u.get("query", []))
            assert not query & seen
            seen |= query
    for key, wanted in allocation.items():
        assert len(first[key]) == min(wanted, len(first[key])) <= wanted
        if len(first[key]) < wanted:
            assert short[f"{key[0]}|{key[1]}"]["sampled"] == len(first[key])
    practice, _ = tr.sample_practice(units, {("W1", "K1"): 2, ("N2", "K1"): 1}, np.random.default_rng(4), used)
    for us in practice.values():
        for u in us:
            assert not set(u["query"]) & seen


def test_injected_units_carry_the_opposite_answer():
    _, _, _, units = world_units()
    assert units[("N2", "P+")] and all(u["truth"] == [True] and u["injected"] for u in units[("N2", "P+")])
    assert all(not u["injected"] for u in units[("N2", "P")])
    assert all(any(u["truth"]) and u["injected"] and len(u["screens"]) <= WORLD["double_c"]
               for u in units[("N2", "S+")])
    assigned = {u["source_index"]: u["screens"][0] for u in units[("N2", "A")]}
    assert units[("N2", "A+")] and all(u["injected"] and u["screens"][0] != assigned[u["source_index"]]
                                       for u in units[("N2", "A+")])


def test_task_payload_hides_truth_and_ranks_exemplars_by_similarity():
    _, w1_cat, scores, units = world_units()
    unit = next(u for u in units[("W1", "K2")] if len(u["screens"]) >= 2)
    photo = lambda i: {"image": int(i), "date": DATES[i], "camera": "c"}
    payload, truth, images = tr.build_task("t001", unit, w1_cat, scores, photo)
    text = json.dumps(payload)
    for word in ("identity", "stratum", "truth", "injected", "feedback", "source_index"):
        assert word not in text
    assert [s["kind"] for s in payload["screens"]][:2] == ["rank1_comparison", "further_comparison"]
    assert truth["true_step"] == unit["truth"].index(True) + 1 and truth["stratum"] == "K2"
    members = w1_cat.members_at(unit["screens"][0], unit["seq"])
    best = max(members, key=lambda m: (scores[unit["query"], m].max(), -m))
    first = payload["screens"][0]
    assert first["exemplars"][0]["image"] == best and len(first["exemplars"]) <= tr.EXEMPLARS
    assert first["members"] == len(members) and images >= set(unit["query"])
    practice, _, _ = tr.build_task("p001", unit, w1_cat, scores, photo, practice=True)
    assert practice["feedback"] == ["same" if t else "different" for t in unit["truth"]]


def test_builder_refuses_identity_bearing_browser_text():
    builder.check_browser_text(json.dumps({"banner": "Find this lynx in the catalogue", "date": "2020-01-02",
                                           "camera": "snpa-77", "token": "ab12"}), ["lynx_065"])
    for leak in ("lynx_065", "clx_0008731f", "CzechLynx/snpa/x", "28866.jpg", "crop.png"):
        with pytest.raises(ValueError):
            builder.check_browser_text(json.dumps({"x": leak}), ["lynx_065"])
    with pytest.raises(ValueError):
        builder.check_browser_text("Tom", ["Tom"])


def png(*chunks):
    out = b"\x89PNG\r\n\x1a\n"
    for kind, body in chunks:
        out += len(body).to_bytes(4, "big") + kind + body + zlib.crc32(kind + body).to_bytes(4, "big")
    return out


def test_leak_scan_reads_image_metadata_not_compressed_pixels():
    segment = lambda marker, body: b"\xff" + marker + (len(body) + 2).to_bytes(2, "big") + body
    scan = b"\x12\x34LYNx8\x99\x00"  # A chance byte run inside entropy-coded data (as in calibration image 602).
    clean_jpeg = b"\xff\xd8" + segment(b"\xe0", b"JFIF\0") + b"\xff\xda\x00\x02" + scan + b"\xff\xd9"
    found = lambda data: bool(builder.IDENTITY_BYTES.search(builder.readable_metadata(data)))
    assert not found(clean_jpeg)
    assert found(b"\xff\xd8" + segment(b"\xfe", b"photo lynx_065") + b"\xff\xda\x00\x02\x01\xff\xd9")
    assert found(b"\xff\xd8" + segment(b"\xe1", b"Exif\0\0 28866_lynx_065.jpg") + b"\xff\xda\x00\x02\xff\xd9")
    assert found(clean_jpeg + b"appended lynx-2")  # Bytes after the end-of-image marker.
    assert not found(png((b"IHDR", b"\0" * 13), (b"IDAT", b"lynx_1 inside pixels"), (b"IEND", b"")))
    assert found(png((b"IHDR", b"\0" * 13), (b"tEXt", b"Comment\0lynx_065"), (b"IEND", b"")))
    assert found(png((b"zTXt", b"Comment\0\0" + zlib.compress(b"LYNX 7 lynx_7"))))
    assert found(png((b"iTXt", b"Comment\0\1\0en\0\0" + zlib.compress(b"lynx_9"))))
    assert found(b"GIF89a lynx_3")  # Unknown formats are scanned whole.


def test_step_times_subtract_pauses_and_attribute_undo_to_the_returned_step():
    events = [{"seq": 1, "type": "task_start", "task_id": "t", "t": 0},
              {"seq": 2, "type": "screen_shown", "step": 1, "t": 0},
              {"seq": 3, "type": "pause", "t": 1000}, {"seq": 4, "type": "resume", "t": 3000},
              {"seq": 5, "type": "answer", "step": 1, "t": 5000},
              {"seq": 6, "type": "screen_shown", "step": 2, "t": 6000},
              {"seq": 7, "type": "undo", "to_step": 1, "t": 7000},
              {"seq": 8, "type": "screen_shown", "step": 1, "t": 8000},
              {"seq": 9, "type": "answer", "step": 1, "t": 9000},
              {"seq": 10, "type": "screen_shown", "step": 2, "t": 9100},
              {"seq": 11, "type": "answer", "step": 2, "t": 10100},
              {"seq": 12, "type": "undo", "to_step": 2, "t": 10500},  # From the confirmation bar: no time.
              {"seq": 13, "type": "task_complete", "task_id": "t", "t": 11000}]
    times = tr.step_times(events)
    assert times[1]["ms"] == pytest.approx(5000) and times[1]["paused_ms"] == pytest.approx(2000)
    assert times[1]["undos"] == 1 and times[2]["ms"] == pytest.approx(1000)
    stale = [{"seq": 0, "type": "task_start", "task_id": "t", "t": 0},
             {"seq": 0.5, "type": "screen_shown", "step": 1, "t": 0}]
    assert tr.attempt_events(stale + events, "t")[0]["seq"] == 1  # The completed attempt only.


def test_a_pause_that_starts_before_the_screen_is_subtracted_from_the_start():
    session = [{"seq": 1, "type": "pause", "t": 0}, {"seq": 2, "type": "task_start", "task_id": "t", "t": 10},
               {"seq": 3, "type": "screen_shown", "step": 1, "t": 100}, {"seq": 4, "type": "resume", "t": 600},
               {"seq": 5, "type": "answer", "step": 1, "t": 1600},
               {"seq": 6, "type": "task_complete", "task_id": "t", "t": 1700}]
    attempt = tr.attempt_events(session, "t")
    assert tr.paused_before(session, attempt[0]["seq"])
    times = tr.step_times(attempt, tr.paused_before(session, attempt[0]["seq"]))
    assert times[1]["ms"] == pytest.approx(1000) and times[1]["paused_ms"] == pytest.approx(500)
    assert tr.step_times(attempt)[1]["ms"] == pytest.approx(1500)  # Without the carried pause state.


def tiny_export(w_ms=(2000, 2000), n_ms=4000, extra=()):
    tasks = {"w": {"screens": [{"kind": "rank1_comparison"}, {"kind": "further_comparison"}]},
             "n": {"screens": [{"kind": "rank1_comparison"}]}}
    truth = {"w": {"arm": "W1", "type": "placement", "stratum": "N", "injected": False, "practice": False,
                   "screens_truth": [False, False]},
             "n": {"arm": "N2", "type": "placement", "stratum": "K1", "injected": False, "practice": False,
                   "screens_truth": [True]}}
    events, seq, t = [], 0, 0.0

    def add(kind, **kw):
        nonlocal seq
        seq += 1
        events.append({"reviewer": "R1", "session_id": "s1", "seq": seq, "type": kind, "t": t, **kw})

    add("task_start", task_id="w")
    for step, ms in enumerate(w_ms, 1):
        add("screen_shown", task_id="w", step=step)
        t += ms
        add("answer", task_id="w", step=step, answer="different")
    add("task_complete", task_id="w")
    add("task_start", task_id="n")
    add("screen_shown", task_id="n", step=1)
    t += n_ms
    add("answer", task_id="n", step=1, answer="same")
    add("task_complete", task_id="n")
    answers = [{"reviewer": "R1", "task_id": "w", "step": s, "answer": "different", "decision_ms": ms}
               for s, ms in enumerate(w_ms, 1)]
    answers.append({"reviewer": "R1", "task_id": "n", "step": 1, "answer": "same", "decision_ms": n_ms})
    export = {"kind": tr.EXPORT_KIND, "events": events, "answers": answers + list(extra),
              "completions": [{"reviewer": "R1", "task_id": k, "session_id": "s1"} for k in ("w", "n")]}
    return tasks, truth, export


def test_stratified_estimator_recovers_known_times():
    table = tr.decision_table(*tiny_export())
    assert table.seconds_raw.tolist() == [2.0, 2.0, 4.0] and set(table.timing_source) == {"events"}
    counts = {("W1", "N", "rank1_comparison"): 10, ("W1", "N", "further_comparison"): 90,
              ("W1", "X", "extra_animal"): 2, ("N2", "K1", "rank1_comparison"): 25, ("N2", "X", "extra_animal"): 1}
    result = tr.stratified_saving(table, counts)
    assert result["totals_seconds"] == {"W1": pytest.approx(210.0), "N2": pytest.approx(105.0)}
    assert result["saving"] == pytest.approx(0.5)
    low, high, _ = tr.bootstrap_saving(table, counts, resamples=50)
    assert low == pytest.approx(0.5) and high == pytest.approx(0.5)


def test_cap_and_fallback_cells():
    table = tr.decision_table(*tiny_export(w_ms=(300000, 2000)))
    counts = {("W1", "N", "rank1_comparison"): 1, ("W1", "N", "further_comparison"): 1,
              ("N2", "K1", "rank1_comparison"): 1, ("N2", "K2", "rank1_comparison"): 1}
    cells = tr.cell_costs(table, counts).set_index(["arm", "stratum", "kind"])
    assert cells.loc[("W1", "N", "rank1_comparison"), "seconds_per_decision"] == pytest.approx(120.0)
    assert cells.loc[("N2", "K2", "rank1_comparison"), "source"] == "arm_kind_fallback"
    uncapped = tr.cell_costs(table, counts, cap=None).set_index(["arm", "stratum", "kind"])
    assert uncapped.loc[("W1", "N", "rank1_comparison"), "seconds_per_decision"] == pytest.approx(300.0)


def test_decision_rule_boundaries_and_labels():
    ok = {"R1": .5, "R2": .5}
    assert tr.decision_rule(.40, (.41, .5), ok, True)["verdict"] == "SUCCESS"
    assert tr.decision_rule(.3999, (.3, .5), ok, True)["verdict"] == "PARTIAL"
    assert tr.decision_rule(.25, (.2, .3), ok, True)["verdict"] == "PARTIAL"
    assert tr.decision_rule(.2499, (.2, .3), ok, True)["verdict"] == "REFRAME"
    assert tr.decision_rule(.45, (.4, .5), ok, False)["verdict"] == "INCOMPLETE"
    labelled = tr.decision_rule(.45, (.35, .55), {"R1": .5, "R2": .38}, True, reduced=True, reviewers=1)
    assert labelled["labels"] == ["imprecise", "reviewer-dependent", "reduced", "single reviewer"]


def test_error_rates_exclude_unsure_and_report_the_sensitivity():
    answers = ["same"] * 2 + ["unsure"] + ["different"] * 7
    table = pd.DataFrame({"practice": False, "arm": "N2", "kind": "pair_question", "truth": False, "answer": answers,
                          "task_id": [f"t{i}" for i in range(10)]})
    row = tr.error_rates(table).set_index("error").loc["false_same"]
    assert row.decisions == 10 and row.errors == 2 and row.unsure == 1
    assert row.rate == pytest.approx(2 / 9) and row.unsure_rate == pytest.approx(.1)
    assert row.rate_unsure_as_error == pytest.approx(.3) and row.low < row.rate < row.high
    rates = tr.measured_rates(table.assign(stratum="P"))
    assert rates[("N2", "pair_question")]["false_same"] == pytest.approx(.2)
    assert rates[("N2", "merge_confirmation")]["false_same"] == pytest.approx(.2)  # Borrowed from pair questions.


def test_measured_reviewer_hits_its_rates():
    reviewer = tr.MeasuredReviewer(false_same=.1, false_different=.3, seed=1)
    different = np.mean([reviewer.same(False) for _ in range(20000)])
    same = np.mean([not reviewer.same(True) for _ in range(20000)])
    assert different == pytest.approx(.1, abs=.01) and same == pytest.approx(.3, abs=.015)
    perfect = tr.MeasuredReviewer(0, 0)
    assert perfect.same(True) and not perfect.same(False)
    with pytest.raises(ValueError):
        tr.MeasuredReviewer(1.2, 0)


def test_blocks_are_balanced_and_reviewer_orders_follow_abba():
    records = [{"task_id": f"w{i}", "arm": "W1", "stratum_key": "K1" if i < 60 else "N"} for i in range(150)]
    records += [{"task_id": f"n{i}", "arm": "N2", "stratum_key": "K1" if i < 30 else "P"} for i in range(150)]
    blocks = tr.make_blocks(records, np.random.default_rng(0))
    assert [len(b) for b in blocks["W1"]] == [25] * 6 and [len(b) for b in blocks["N2"]] == [25] * 6
    assert all(sum(t.startswith("w") and int(t[1:]) < 60 for t in b) == 10 for b in blocks["W1"])
    orders = tr.reviewer_orders(blocks, ["p1", "p2"], 42)
    assert [b["arm"] for b in orders["R1"]["blocks"]][:4] == ["W1", "N2", "N2", "W1"]
    assert [b["arm"] for b in orders["R2"]["blocks"]][:4] == ["N2", "W1", "W1", "N2"]
    for slot in orders:
        assert sorted(t for b in orders[slot]["blocks"] for t in b["tasks"]) == sorted(r["task_id"] for r in records)
        first = orders[slot]["blocks"][:tr.RULE["early_stop_blocks"]]
        assert arm_counts(first) == {"W1": 4, "N2": 4}
    r1 = orders["R1"]["blocks"]
    every = [t for b in r1 for t in b["tasks"]]
    reduced = [t for b in r1[:8] for t in b["tasks"]]
    assert tr.completion_status(orders, "R1", every + ["p1"]) == "complete"
    assert tr.completion_status(orders, "R1", reduced) == "reduced"
    assert tr.completion_status(orders, "R1", reduced[:-1]) == "incomplete"


def arm_counts(blocks):
    out = {}
    for b in blocks:
        out[b["arm"]] = out.get(b["arm"], 0) + 1
    return out
