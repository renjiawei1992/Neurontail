import numpy as np

from a5_neurotail.identify.lape import lape_scores, select_lape


def test_lape_scores_prefers_one_language():
    # neuron 0 fires only in language 0; neuron 1 fires equally in both
    p = np.array([[0.9, 0.5], [0.01, 0.5]])
    s = lape_scores(p, smoothing=1e-6)
    assert s[0] < s[1]
    sel = select_lape(s, p, target_idx=0, top_frac=0.5, min_prob=0.05)
    assert 0 in sel.tolist()


def test_select_lape_respects_min_prob():
    p = np.array([[0.001, 0.8], [0.001, 0.8]])
    s = np.array([0.0, 0.0])
    sel = select_lape(s, p, target_idx=0, top_frac=1.0, min_prob=0.05)
    assert len(sel) >= 1  # fallback when no neuron passes
