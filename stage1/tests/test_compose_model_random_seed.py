"""Unit tests for compose_model random_donor determinism — spec §10.1.

These tests verify the RNG behavior inside compose_model without loading
actual model weights. We monkey-patch compose_model to exercise only the
source_start sampling logic.
"""

import copy
import random
import unittest.mock as mock

import pytest

from stage1.models.composer import compose_model, compute_random_donor_seed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_layer():
    """A minimal object with state_dict / load_state_dict for mocking."""
    class FakeLayer:
        def __init__(self):
            self._state = {}

        def state_dict(self):
            return dict(self._state)

        def load_state_dict(self, sd):
            self._state.update(sd)

    return FakeLayer()


def _fake_model(n_layers: int = 28):
    """Minimal model stub accepted by compose_model."""
    class FakeConfig:
        num_hidden_layers = n_layers
        hidden_size = 8
        num_attention_heads = 2

    class FakeLayerList:
        def __init__(self, layers):
            self._layers = layers

        def __getitem__(self, idx):
            return self._layers[idx]

        def __len__(self):
            return len(self._layers)

    class FakeModelInner:
        def __init__(self, layers):
            self.layers = layers

    class FakeModel:
        def __init__(self):
            self.config = FakeConfig()
            layers = [_fake_layer() for _ in range(n_layers)]
            self.model = FakeModelInner(FakeLayerList(layers))

        def __deepcopy__(self, memo):
            new_obj = FakeModel.__new__(FakeModel)
            new_obj.config = self.config
            new_obj_layers = [_fake_layer() for _ in range(self.config.num_hidden_layers)]
            new_obj.model = FakeModelInner(FakeLayerList(new_obj_layers))
            return new_obj

    return FakeModel()


def _get_source_start_from_compose(recipient, donor, b: int, t: int, seed: int) -> int:
    """Run compose_model and extract source_start from returned metadata."""
    _, meta = compose_model(recipient, donor, b=b, t=t, condition="random_donor", seed=seed)
    return meta["source_start"]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_same_seed_deterministic_across_two_calls():
    """compose_model(random_donor, seed=42812) yields the same source_start twice."""
    seed = compute_random_donor_seed(42, 8, 12)  # 42812
    b, t = 8, 12

    recipient1 = _fake_model()
    donor1 = _fake_model()
    recipient2 = _fake_model()
    donor2 = _fake_model()

    ss1 = _get_source_start_from_compose(recipient1, donor1, b, t, seed)
    ss2 = _get_source_start_from_compose(recipient2, donor2, b, t, seed)

    assert ss1 == ss2, (
        f"Same seed {seed} must produce same source_start; got {ss1} vs {ss2}"
    )


def test_different_seeds_produce_different_source_starts():
    """Different seeds must produce different source_starts; uses precomputed constants.

    This test verifies that the caller's seed is used verbatim (not re-encoded)
    by checking against analytically derived expected source_start values.
    Expected values: random.Random(42812).randint(0, 24) and random.Random(43216).randint(0, 24).
    """
    b, t = 8, 12  # block_width = 4, max_start = 24
    seed_a = compute_random_donor_seed(42, 8, 12)   # 42812
    seed_b = compute_random_donor_seed(42, 12, 16)  # 43216

    # Precomputed expected values — ensures verbatim seed use
    import random as _rand
    expected_ss_a = _rand.Random(seed_a).randint(0, 24)
    expected_ss_b = _rand.Random(seed_b).randint(0, 24)

    r_a = _fake_model()
    d_a = _fake_model()
    ss_a = _get_source_start_from_compose(r_a, d_a, b, t, seed_a)

    r_b = _fake_model()
    d_b = _fake_model()
    ss_b = _get_source_start_from_compose(r_b, d_b, b, t, seed_b)

    assert ss_a == expected_ss_a, (
        f"seed {seed_a} must yield source_start {expected_ss_a}; got {ss_a}"
    )
    assert ss_b == expected_ss_b, (
        f"seed {seed_b} must yield source_start {expected_ss_b}; got {ss_b}"
    )
    assert {ss_a, ss_b} != {ss_a} or ss_a != ss_b, (
        "seeds 42812 and 43216 happen to produce same source_start — verify formula"
    )
    # Primary assertion: different seeds produce different source_starts
    assert len({ss_a, ss_b}) > 1, (
        f"seeds {seed_a} and {seed_b} must produce different source_starts; "
        f"both yielded {ss_a}"
    )


def test_metadata_contains_expected_keys():
    """compose_model for random_donor must return seed, source_start, b, t in metadata."""
    seed = 42812
    b, t = 8, 12
    recipient = _fake_model()
    donor = _fake_model()

    _, meta = compose_model(recipient, donor, b=b, t=t, condition="random_donor", seed=seed)

    assert "seed" in meta, "metadata must contain 'seed'"
    assert "source_start" in meta, "metadata must contain 'source_start'"
    assert "b" in meta, "metadata must contain 'b'"
    assert "t" in meta, "metadata must contain 't'"
    assert meta["seed"] == seed
    assert meta["b"] == b
    assert meta["t"] == t


def test_source_start_in_valid_range():
    """source_start must satisfy 0 <= source_start <= num_layers - block_width."""
    b, t = 8, 12  # block_width = 4
    n_layers = 28
    max_start = n_layers - (t - b)  # 24

    for seed in range(100, 120):
        recipient = _fake_model()
        donor = _fake_model()
        ss = _get_source_start_from_compose(recipient, donor, b, t, seed)
        assert 0 <= ss <= max_start, (
            f"source_start={ss} out of range [0, {max_start}] for seed={seed}"
        )
