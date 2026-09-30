"""Tests for the sanity helper: one-to-one (Hungarian) matched accuracy."""
import numpy as np
import pytest

import sanity


def test_hungarian_perfect_prediction():
    true = np.array([1, 1, 2, 2, 3, 3, 0])
    pred = np.array([7, 7, 5, 5, 9, 9, 0])       # relabelled, plus noise==noise
    res = sanity.hungarian_match_accuracy(pred, true, np.ones(7, bool))
    assert res["accuracy"] == 1.0
    assert res["n_predicted_labels"] == 4
    assert res["n_true_labels"] == 4


def test_hungarian_penalises_split():
    # one true cell split across two predicted labels: at most the majority
    # mass of the true cell can be recovered, never both halves
    true = np.array([1] * 10 + [2] * 10)
    pred = np.array([4] * 6 + [5] * 4 + [6] * 10)
    # best injective map: 4->1 (6 mol), 6->2 (10 mol); label 5 unmatched
    res = sanity.hungarian_match_accuracy(pred, true, np.ones(20, bool))
    assert res["accuracy"] == pytest.approx(16 / 20)


def test_hungarian_never_exceeds_many_to_one_upper_bound():
    # merging two true cells into one predicted label: at most one is credited
    true = np.array([1] * 10 + [2] * 10)
    pred = np.array([4] * 20)
    res = sanity.hungarian_match_accuracy(pred, true, np.ones(20, bool))
    assert res["accuracy"] == pytest.approx(10 / 20)


def test_hungarian_noise_semantics():
    # a cell prediction on background molecules is never correct, and noise
    # predictions are only correct on true background
    true = np.array([1] * 4 + [0] * 4)
    pred = np.array([3] * 4 + [3] * 4)      # cell 3 covers cell 1 and bg
    res = sanity.hungarian_match_accuracy(pred, true, np.ones(8, bool))
    assert res["accuracy"] == pytest.approx(4 / 8)


def test_hungarian_respects_mask_and_rejects_empty():
    true = np.array([1, 1, 2, 2])
    pred = np.array([1, 1, 2, 2])
    mask = np.array([True, False, True, False])
    res = sanity.hungarian_match_accuracy(pred, true, mask)
    assert res["accuracy"] == 1.0 and res["n_evaluated"] == 2
    with pytest.raises(ValueError, match="empty"):
        sanity.hungarian_match_accuracy(pred, true, np.zeros(4, bool))


# ---------------------------------------------------------------------------
# NCV colour skipping (--skip-ncv-color by default)
# ---------------------------------------------------------------------------

def _fake_binary(path, advertises_flag: bool = True):
    """Executable stub whose `run --help` does/does not list --skip-ncv-color."""
    from pathlib import Path

    path.mkdir(parents=True, exist_ok=True)
    body = '#!/usr/bin/env bash\nif [ "$1" = run ] && [ "$2" = --help ]; then\n'
    if advertises_flag:
        body += ('  echo "  --skip-ncv-color  Skip neighborhood composition '
                 'color embedding to speed up development runs"\n')
    else:
        body += '  echo "Usage: baysor run"\n'
    body += "  exit 0\nfi\nexit 0\n"
    bin_path = path / "baysor_stub"
    bin_path.write_text(body)
    bin_path.chmod(0o755)
    return str(bin_path)


_SANE_CFG = {"scale_um": 5.0, "min_molecules_per_cell": 10,
             "prior": "none", "prior_confidence": 0.5, "extra_args": []}


def test_supports_skip_ncv_color(tmp_path):
    yes = _fake_binary(tmp_path / "yes", True)
    no = _fake_binary(tmp_path / "no", False)
    assert sanity.supports_skip_ncv_color(yes) is True
    assert sanity.supports_skip_ncv_color(no) is False
    # unrunnable binary counts as unsupported (older builds keep colours)
    assert sanity.supports_skip_ncv_color(str(tmp_path / "missing")) is False


def test_build_baysor_command_skip_ncv_color(tmp_path):
    yes = _fake_binary(tmp_path / "yes", True)
    mol = tmp_path / "molecules.parquet"
    out = tmp_path / "out"
    # default: --skip-ncv-color when the binary supports it
    cmd = sanity.build_baysor_command(yes, mol, _SANE_CFG, out,
                                      prior="none", has_z=False)
    assert cmd.count("--skip-ncv-color") == 1
    # --ncv-color opt-out (skip_ncv_color=False)
    cmd = sanity.build_baysor_command(yes, mol, _SANE_CFG, out,
                                      prior="none", has_z=False,
                                      skip_ncv_color=False)
    assert "--skip-ncv-color" not in cmd
    # binary without the flag: never emitted
    no = _fake_binary(tmp_path / "no", False)
    cmd = sanity.build_baysor_command(no, mol, _SANE_CFG, out,
                                      prior="none", has_z=False)
    assert "--skip-ncv-color" not in cmd
    # meta extra_args carries it: exactly one occurrence (CLI11 dedup)
    cfg = dict(_SANE_CFG, extra_args=["--skip-ncv-color"])
    cmd = sanity.build_baysor_command(yes, mol, cfg, out,
                                      prior="none", has_z=False)
    assert cmd.count("--skip-ncv-color") == 1
