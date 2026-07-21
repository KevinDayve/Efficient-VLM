"""Validate the layer-overlap diagnostic on synthetic recv maps with known answers.

No model, no data, no GPU -- a few seconds on CPU. Run this before trusting a
layer_overlap table: every claim the diagnostic makes rests on chance = c/(2M-c),
and case 3 below is the one that actually tests that formula (independent random
layers must land at normJ ~ 0). The imports come from lcds_ai2d.py, whose diagnostic
block is backbone-agnostic (qwen and llava share it via --backbone) -- so this covers both.

    python check_layer_overlap.py
"""

import numpy as np
import torch

from lcds_ai2d import layer_overlap_stats, gap_profile

LAYERS = [4, 8, 12, 16, 20, 24, 28]          # the stride-4 plan on a 32-layer LLM
torch.manual_seed(0)
ok = True


def check(tag, got, want, tol=0.02):
    global ok
    hit = abs(got - want) <= tol
    ok &= hit
    print(f"  {'PASS' if hit else 'FAIL'}  {tag:<44} got {got:+.4f}  want {want:+.4f}")


# --- case 1: every layer identical -> J=1, pool == cand -----------------------
print("\n[1] identical layers: union is a no-op")
one = torch.rand(576)
st = layer_overlap_stats({L: one.clone() for L in LAYERS}, LAYERS, 576, 116)
check("mean_jaccard", st["mean_jaccard"], 1.0)
check("pool_inflation", st["pool_inflation"], 1.0)

# --- case 2: disjoint layers -> J=0, pool == n*cand ---------------------------
print("\n[2] disjoint layers: union is maximal")
recv = {}
for i, L in enumerate(LAYERS):
    a = torch.zeros(576)
    a[i * 58:(i + 1) * 58] = 1.0 + torch.rand(58)      # its own exclusive block
    recv[L] = a
st = layer_overlap_stats(recv, LAYERS, 576, 58)
check("mean_jaccard", st["mean_jaccard"], 0.0)
check("pool_inflation", st["pool_inflation"], float(len(LAYERS)))

# --- case 3: independent random layers -> normJ ~ 0 ---------------------------
# THE load-bearing case: validates chance = c/(2M-c), which every normJ depends on.
print("\n[3] random layers: normJ must be ~0 (validates the chance formula)")
for M in (576, 1024):                                   # fixed-grid and a dynamic-res M
    for rho in (0.05, 0.10, 0.25):
        cand = 2 * max(1, round(rho * M))
        nj, ch, mj = [], [], []
        for _ in range(40):
            st = layer_overlap_stats({L: torch.rand(M) for L in LAYERS}, LAYERS, M, cand)
            nj.append(st["norm_jaccard"]); ch.append(st["chance_jaccard"]); mj.append(st["mean_jaccard"])
        check(f"M={M} rho={rho} (meanJ {np.mean(mj):.3f} vs chance {np.mean(ch):.3f})",
              float(np.mean(nj)), 0.0)

# --- case 4: the sink confounder ----------------------------------------------
# Random layers + 20 shared always-hot sinks: raw J is inflated, but excluding the
# sinks must collapse normJ back to ~0. This is what --sink_exclude buys.
print("\n[4] shared attention sinks: raw J lies, sink-exclusion recovers the truth")
recv = {}
for L in LAYERS:
    a = torch.rand(576)
    a[:20] += 10.0
    recv[L] = a
raw = layer_overlap_stats(recv, LAYERS, 576, 116, 0)
ex = layer_overlap_stats(recv, LAYERS, 576, 116, 20)
print(f"       sink=0  meanJ={raw['mean_jaccard']:.3f} normJ={raw['norm_jaccard']:+.3f}")
check("sink=20 normJ collapses to chance", ex["norm_jaccard"], 0.0)

# --- case 5: vacuous configs return None --------------------------------------
print("\n[5] vacuous configs -> None (not a meaningless 1.0)")
for tag, args in (("cand >= M_eff", (LAYERS, 16, 16, 0)),
                  ("sink eats the grid", (LAYERS, 576, 116, 576)),
                  ("single layer", ([4], 576, 116, 0))):
    sl, M, cand, sk = args
    got = layer_overlap_stats({L: torch.rand(M) for L in sl}, sl, M, cand, sk)
    print(f"  {'PASS' if got is None else 'FAIL'}  {tag:<44} -> {got if got is None else 'dict'}")
    ok &= got is None

# --- case 6: gap profile decays with layer distance ---------------------------
print("\n[6] drifting layers: gap profile should decay with distance")
base = torch.rand(576)
recv = {L: base + 0.35 * i * torch.rand(576) for i, L in enumerate(LAYERS)}
prof = gap_profile(layer_overlap_stats(recv, LAYERS, 576, 116)["jaccard"], LAYERS)
print("       " + "  ".join(f"{g}:{v:.3f}" for g, v in prof.items()))
vals = list(prof.values())
mono = all(vals[i] >= vals[i + 1] - 0.05 for i in range(len(vals) - 1))
print(f"  {'PASS' if mono else 'FAIL'}  decays with layer distance")
ok &= mono

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
raise SystemExit(0 if ok else 1)
