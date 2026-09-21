"""Objective, calibration and generator checks. All offline, all fast."""

import argparse
import math
import random

import pytest
import torch

from maatlm.confidence import top1
from maatlm.losses import batch_loss, decision_loss, js_divergence, ranked_probability_score
from maatlm.tiny import tiny_model


def _fit(target, rps_weight, steps=400):
    logits = torch.zeros(len(target), requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    t = torch.tensor(target)
    for _ in range(steps):
        opt.zero_grad()
        decision_loss(logits, t, "score", "log", rps_weight).backward()
        opt.step()
    return torch.softmax(logits.detach(), -1)


def test_log_plus_rps_is_minimised_at_the_true_distribution():
    """Both terms are proper: the optimum of their sum is still the target."""
    target = [0.1, 0.6, 0.3]
    p = _fit(target, rps_weight=1.0)
    assert torch.allclose(p, torch.tensor(target), atol=2e-2), p


def test_rps_is_ordinal_aware():
    t = torch.tensor([1.0, 0.0, 0.0, 0.0])
    near = torch.tensor([0.0, 1.0, 0.0, 0.0])
    far = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert ranked_probability_score(near, t) < ranked_probability_score(far, t)
    # log / Brier cannot tell these apart; RPS can
    assert math.isclose(((near - t) ** 2).sum().item(), ((far - t) ** 2).sum().item())


def test_consistency_term_is_zero_for_identical_and_positive_otherwise():
    p = torch.tensor([0.2, 0.8])
    assert js_divergence(p, p).item() < 1e-6
    assert js_divergence(p, torch.tensor([0.9, 0.1])).item() > 0.1


def test_batch_loss_uses_pairs():
    m = tiny_model(attn_implementation="eager")
    q = {"d": {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None}}}
    l1 = m.layout("The package is late.", q)
    l2 = m.layout("My delivery has not arrived yet.", q)
    batch = m.collate([l1, l2])
    raw = m(batch)
    tg = [[torch.tensor([1.0, 0.0])], [torch.tensor([1.0, 0.0])]]
    base, _ = batch_loss(raw, tg, batch.layouts, "log")
    with_pairs, per = batch_loss(raw, tg, batch.layouts, "log", pairs=[(0, 1)], consistency_weight=1.0)
    assert per["consistency"] is not None and with_pairs.item() >= base.item()


def test_temperature_lookup_prefers_option_count():
    m = tiny_model()
    m.set_temperature("choice", 2.0)
    m.set_temperature("choice:3", 0.5)
    assert math.isclose(m.temperature("choice", 3).item(), 0.5)
    assert math.isclose(m.temperature("choice", 4).item(), 2.0)  # falls back to the type-level value
    assert math.isclose(m.temperature("noul").item(), 1.0)
    assert set(m.temperatures()) >= {"choice", "score", "noul", "choice:3"}


def test_temperatures_survive_save_load(tmp_path):
    m = tiny_model()
    m.set_temperature("score:5", 1.7)
    m.save(str(tmp_path))
    from maatlm.model import SystemOneModel

    m2 = SystemOneModel.from_pretrained(str(tmp_path))
    assert math.isclose(m2.temperature("score", 5).item(), 1.7, rel_tol=1e-5)


def test_confidence_matches_typesafe_formula():
    # (n * p_max - 1) / (n - 1), from docs.typesafe.ai/confidence
    assert math.isclose(top1([0.85, 0.15, 0.0]), 0.775)
    assert math.isclose(top1([0.9, 0.06, 0.04]), 0.85)
    assert math.isclose(top1([1 / 3] * 3), 0.0, abs_tol=1e-9)


def test_paraphrase_dataset_yields_pairs():
    from maatlm.data import Example, SystemOneDataset, collate_train

    m = tiny_model()
    ex = Example(
        state="Charged twice.",
        questions={"n": {"type": "noul", "instructions": "Double charge?"}},
        targets={"n": {"p": 1.0}},
        paraphrases=["I was billed two times."],
    )
    ds = SystemOneDataset(m, [ex], paraphrases=True)
    batch, targets = collate_train(m)([ds[0]])
    assert batch.pairs == [(0, 1)] and len(targets) == 2


# ------------------------------------------------------------------ generator


def _gen(n=240, seed=0, **kw):
    from maatlm.datasets.generator import generate

    args = argparse.Namespace(
        n=n, seed=seed, families=None, contrast=0.5, paraphrases=2, trap=0.2, json_state=0.4, max_options=8
    )
    args.__dict__.update(kw)
    return generate(args)


def test_generator_targets_are_valid_distributions():
    from maatlm.data import target_tensor
    from maatlm.schema import SystemOneRequest

    exs = _gen()
    fams = {e.meta["family"] for e in exs}
    assert fams == {"policy", "routing", "multi_hop", "temporal_numeric", "severity", "judge"}
    for e in exs:
        SystemOneRequest(state=e.state, questions=e.questions)
        for qid, q in e.questions.items():
            t = target_tensor(q, e.targets[qid])
            if q["type"] == "noul":
                assert 0.0 <= float(t) <= 1.0
            else:
                assert (t >= 0).all() and abs(float(t.sum()) - 1.0) < 1e-6


def test_generator_has_constructed_uncertainty():
    """Some targets must be genuinely soft (that is the point of the generator)."""
    exs = _gen()
    soft = 0
    for e in exs:
        for qid, q in e.questions.items():
            t = e.targets[qid]
            vals = [t["p"]] if "p" in t else t.get("probabilities", [1.0])
            if any(0.05 < v < 0.95 for v in vals):
                soft += 1
    assert soft > 20


def test_generator_contrastive_twins_flip_an_answer():
    exs = _gen(contrast=1.0)
    by = {}
    for e in exs:
        by.setdefault(e.meta["group"], {})[e.meta["variant"]] = e
    pairs = [v for v in by.values() if "a" in v and "b" in v]
    assert len(pairs) > 50
    for v in pairs:
        a, b = v["a"], v["b"]
        flipped = False
        for qid in a.questions:
            ta, tb = a.targets[qid], b.targets[qid]
            if "p" in ta and (ta["p"] >= 0.5) != (tb["p"] >= 0.5):
                flipped = True
            if "probabilities" in ta and ta["probabilities"].index(max(ta["probabilities"])) != tb["probabilities"].index(max(tb["probabilities"])):
                flipped = True
            if "label" in ta and ta["label"] != tb["label"]:
                flipped = True
        assert flipped, (a.meta, a.targets, b.targets)


def test_generator_paraphrases_share_targets_and_traps_do_not_move_them():
    exs = _gen(trap=1.0, paraphrases=2)
    assert all(e.meta["trap"] for e in exs)
    with_p = [e for e in exs if e.paraphrases]
    assert len(with_p) > len(exs) // 2
    # the same seed reproduces the same data (determinism matters for held-out sets)
    again = _gen(trap=1.0, paraphrases=2)
    assert [e.targets for e in exs] == [e.targets for e in again]
    assert [e.state for e in exs] == [e.state for e in again]


def test_generator_split_keeps_groups_together():
    from maatlm.datasets.generator import split_by_group

    exs = _gen(contrast=1.0)
    train, calib, val = split_by_group(exs, random.Random(1), 0.1, 0.1)
    g = lambda xs: {e.meta["group"] for e in xs}
    assert not (g(train) & g(calib)) and not (g(train) & g(val)) and not (g(calib) & g(val))
    assert len(train) + len(calib) + len(val) == len(exs)


# ------------------------------------------------------------------ paid-data tooling (offline parts)


def test_rewrite_acceptance_keeps_facts():
    from maatlm.datasets.rewrite import _accept

    src = "Confirmed: a receipt is on file. Purchased on 2026-03-02. Order total: $480. It's doubtful that the item is unused."
    good = "Hi team, quick one: we've confirmed the receipt is on file, the purchase was on 2026-03-02 and the order came to $480. It's doubtful the item is unused though."
    dropped_number = "Confirmed: receipt on file, purchased 2026-03-02. It's doubtful the item is unused."
    softened_hedge = "Confirmed: a receipt is on file. Purchased on 2026-03-02. Order total: $480. The item is unused."
    assert _accept(src, good)
    assert not _accept(src, dropped_number)
    assert not _accept(src, softened_hedge)
    assert not _accept(src, "")


def test_distill_prompt_and_parse_roundtrip_carry_meta():
    from maatlm.data import Example
    from maatlm.distill import build_prompt, parse_teacher

    ex = Example("Charged twice.", {"q": {"type": "choice", "instructions": "Team?", "criteria": {"billing": None, "other": None}}, "n": {"type": "noul", "instructions": "Refund?"}}, {}, ["Billed two times."], {"source": "x"})
    prompt = build_prompt(ex)
    assert "billing" in prompt and "Refund?" in prompt
    t = parse_teacher(ex, '{"q": {"probabilities": {"billing": 0.9, "other": 0.1}}, "n": {"p_true": 0.2}}')
    assert t == {"q": {"probabilities": [0.9, 0.1]}, "n": {"p": 0.2}}


def test_unlabeled_pack_is_schema_valid():
    from maatlm.datasets.unlabeled import PACK
    from maatlm.schema import SystemOneRequest

    SystemOneRequest(state={"subject": "s", "message": "m"}, questions=PACK)
    assert len(PACK) >= 10 and {q["type"] for q in PACK.values()} == {"choice", "score", "noul"}


# ------------------------------------------------------------------ backbone safety


class _Cfg:
    def __init__(self, layer_types, sliding_window=None):
        self.layer_types = layer_types
        self.sliding_window = sliding_window


def test_full_attention_backbone_is_accepted():
    from maatlm.model import check_backbone_attention

    info = check_backbone_attention(_Cfg(["full_attention"] * 36))
    assert info["incompatible"] == {} and info["layer_types"] == {"full_attention": 36}


def test_sliding_window_backbone_is_accepted():
    """Gemma 4: sliding attention still honours an explicit mask (measured drift 0.0000)."""
    from maatlm.model import check_backbone_attention

    info = check_backbone_attention(_Cfg(["sliding_attention"] * 28 + ["full_attention"] * 7, 512))
    assert info["incompatible"] == {} and info["sliding_window"] == 512


def test_linear_attention_backbone_is_refused():
    """Qwen3.5: linear-attention layers ignore the tree mask, so the invariants are void."""
    from maatlm.model import check_backbone_attention

    cfg = _Cfg(["linear_attention"] * 24 + ["full_attention"] * 8)
    with pytest.raises(RuntimeError, match="isolation and option equivariance"):
        check_backbone_attention(cfg)
    info = check_backbone_attention(cfg, strict=False)  # non-strict warns instead
    assert info["incompatible"] == {"linear_attention": 24}


def test_unknown_layer_types_do_not_block():
    from maatlm.model import check_backbone_attention

    assert check_backbone_attention(_Cfg([]))["incompatible"] == {}
