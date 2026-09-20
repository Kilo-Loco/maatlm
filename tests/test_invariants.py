"""Structural guarantees of the parallel layout.

These hold for ANY weights (they're properties of the attention mask and
position ids), so they're checked on a random tiny backbone.
"""

import math

import pytest
import torch

from maatlm.tiny import tiny_model

STATE = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."
Q_DEPT = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"returns": "Exchanges, refunds", "shipping": "Delivery delays", "billing": "Charges"},
}
Q_SEV = {"type": "score", "instructions": "How severe?", "criteria": ["Cosmetic", "Degraded", "Blocking"]}
Q_REFUND = {"type": "noul", "instructions": "The customer asks for a refund."}


@pytest.fixture(scope="module")
def model():
    m = tiny_model(attn_implementation="eager")  # eager = exact float math on CPU
    m.eval()
    return m


def _probs(resp, qid):
    a = resp.answers[qid]
    if a.type == "noul":
        return {"noul": a.noul}
    return a.probabilities


def test_questions_are_isolated(model):
    """Answer to `department` is identical whether or not other questions are present."""
    alone = model.predict(STATE, {"department": Q_DEPT})
    together = model.predict(STATE, {"severity": Q_SEV, "refund": Q_REFUND, "department": Q_DEPT})
    a, b = _probs(alone, "department"), _probs(together, "department")
    for k in a:
        assert math.isclose(a[k], b[k], abs_tol=1e-3), (k, a[k], b[k])


def test_question_order_does_not_matter(model):
    r1 = model.predict(STATE, {"department": Q_DEPT, "severity": Q_SEV, "refund": Q_REFUND})
    r2 = model.predict(STATE, {"refund": Q_REFUND, "severity": Q_SEV, "department": Q_DEPT})
    for qid in ("department", "severity", "refund"):
        a, b = _probs(r1, qid), _probs(r2, qid)
        for k in a:
            assert math.isclose(a[k], b[k], abs_tol=1e-3), (qid, k, a[k], b[k])


def test_option_order_is_equivariant(model):
    """Permuting options permutes the distribution exactly; no positional bias."""
    q = dict(Q_DEPT)
    q_rev = dict(Q_DEPT, criteria=dict(reversed(list(Q_DEPT["criteria"].items()))))
    a = _probs(model.predict(STATE, {"d": q}), "d")
    b = _probs(model.predict(STATE, {"d": q_rev}), "d")
    for k in a:
        assert math.isclose(a[k], b[k], abs_tol=1e-3), (k, a[k], b[k])


def test_batching_matches_single(model):
    single = [model.predict(STATE, {"d": Q_DEPT, "s": Q_SEV}), model.predict("Nothing works.", {"r": Q_REFUND})]
    batched = model.predict_batch([(STATE, {"d": Q_DEPT, "s": Q_SEV}), ("Nothing works.", {"r": Q_REFUND})])
    for s, b in zip(single, batched):
        for qid in s.answers:
            pa, pb = _probs(s, qid), _probs(b, qid)
            for k in pa:
                assert math.isclose(pa[k], pb[k], abs_tol=1e-3)


def test_type_safety(model):
    r = model.predict(STATE, {"d": Q_DEPT, "s": Q_SEV, "r": Q_REFUND})
    d, s, n = r.answers["d"], r.answers["s"], r.answers["r"]
    assert set(d.probabilities) == set(Q_DEPT["criteria"])
    assert d.choice in Q_DEPT["criteria"]
    assert math.isclose(sum(d.probabilities.values()), 1.0, abs_tol=2e-3)
    assert set(s.probabilities) == {"0", "1", "2"} and 0.0 <= s.score <= 2.0
    assert 0.0 <= n.noul <= 1.0
    assert r.usage.output_tokens == 3 + 3 + 1  # one readout slot per option / level / statement


def test_schema_limits():
    from pydantic import ValidationError
    from maatlm.schema import ChoiceQuestion, ScoreQuestion

    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="x", criteria={f"o{i}": None for i in range(256)})
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="x", criteria=["only one"])
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="x", criteria=[str(i) for i in range(11)])


def test_mask_is_a_tree(model):
    lay = model.layout(STATE, {"d": Q_DEPT, "r": Q_REFUND})
    m = lay.attn_mask
    # option end tokens never attend to each other
    p = lay.heads[0].positions
    assert not m[p[1], p[0]] and not m[p[0], p[1]]
    # noul end token never sees the choice question, but sees the state
    n = lay.heads[1].positions[0]
    assert not m[n, p[0]] and m[n, 0]
    # causal within the flat sequence
    assert not torch.triu(m, diagonal=1).any()
