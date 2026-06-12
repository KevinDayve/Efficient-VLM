"""Invariant + limit-behavior tests for the Pareto-adaptive budgeting (paper 2.4).

Run on a machine with torch:  python tests/test_pareto_budget.py
"""
import torch
from efficient_vlm.utils import pareto_budget, hill_tail_index

T, N = 8, 128
n_video = T * N


def _check(scores, K, k_min=1, temp=1.0, label=""):
    b = pareto_budget(scores, n_frames=T, K=K, k_min=k_min, temp=temp)
    assert b.sum().item() == min(K, n_video), f"{label}: sum {int(b.sum())} != {K}"
    assert (b >= 0).all() and (b <= N).all(), f"{label}: out of [0, N]"
    if k_min <= K // T:
        assert (b >= k_min).all(), f"{label}: a bin fell below k_min"
    return b


def main():
    torch.manual_seed(0)
    heavy = torch.distributions.Pareto(torch.tensor(1.5), torch.tensor(1.0)).sample((n_video,)).log()
    flat = torch.zeros(n_video)

    for K in (T, 256, 512, 768, n_video):
        _check(heavy, K, label=f"heavy K={K}")
        _check(flat, K, label=f"flat K={K}")

    b_uniform = pareto_budget(heavy, T, 512, k_min=1, temp=0.0)
    b_concent = pareto_budget(heavy, T, 512, k_min=1, temp=100.0)
    print("gamma(heavy)            =", round(hill_tail_index(torch.softmax(heavy, 0)), 3))
    print("temp=0   budgets        =", b_uniform.tolist())
    print("temp=100 budgets        =", b_concent.tolist())
    assert (b_uniform.max() - b_uniform.min()) <= (b_concent.max() - b_concent.min()), \
        "temp=0 (uniform) should be flatter than temp=100 (concentrated)"

    b_low = pareto_budget(heavy, T, 5, k_min=1)   # K < T: can't floor every bin
    assert b_low.sum().item() == 5
    print("K=5 (<T) budgets        =", b_low.tolist())
    print("ALL PARETO_BUDGET TESTS PASSED")


if __name__ == "__main__":
    main()
