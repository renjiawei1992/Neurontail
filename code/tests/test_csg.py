from a5_neurotail.eval.csg import causal_specificity_gap


def test_csg_uses_strongest_control():
    assert causal_specificity_gap(10.0, 2.0, 3.0) == 7.0
    assert causal_specificity_gap(10.0, 8.0, 3.0) == 2.0
    assert causal_specificity_gap(1.0, 5.0, 3.0) == -4.0
