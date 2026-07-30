# Status / handoff — tail-index study of prunable structure in Qwen2.5-VL

**Read this first in a new session.** Self-contained snapshot of the experiment arc,
results, the current pending fork, and next steps. Full narrative writeup:
[`experiments_writeup.tex`](experiments_writeup.tex).

## ►► ACTIVE THREAD (new chat starts here): port the arc to EgoSchema
**Why:** every negative so far (Exp 4 token-prune ≈ random; Exp 5 cheap frame-select ≈
random, only `conf` weakly works) is on **8-frame MVBench**. EgoSchema is **long-form**
(~3-min egocentric clips, 5 options, aggregate reasoning) — the regime where pruning
matters most and where ReDiPrune (`2603.24680`) showed real gains. Goal: **are our
negatives an 8-frame-MVBench artifact, or do they hold at long context?**

**Setup DONE (2 files to sync):** `make_egoschema_json.py` (added) + `inference.py`
(DATA_LIST["EgoSchema"] line). Data on box at `~/Experiments/EgoSchema`: Subset split =
500 Qs *with* answers (`Subset/test-*.parquet`), full 5031 videos at `videos/videos/`.
Parquet schema: question / video_idx(uuid) / option[] (prefixed "A. ") / answer (0-based
index STRING). Converter strips the "X. " prefix, sets answer=correct option text,
video=`<uuid>.mp4`.

**Run to activate (on box):**
```
python additional_expts/make_egoschema_json.py \
  --parquet ~/Experiments/EgoSchema/Subset/test-00000-of-00001.parquet \
  --out_root ~/Experiments/EgoSchema
ln -s ~/Experiments/EgoSchema/videos/videos ~/Experiments/EgoSchema/video
```
Then any script runs with `--data_root ~/Experiments/EgoSchema --tasks EgoSchema`.

**Plan:**
1. **Exp 3 first** (`frame_sufficiency_mvbench.py --tasks EgoSchema --num_frames 16
   --max_pixels 200704 --blind`). Hypothesis: on long-form, `single_best ≈ or < all`
   (needs accumulation, NOT selection) — the OPPOSITE of MVBench (where single_best ≫ all).
   If so → the frame-selection lever is a short-clip artifact. If single_best ≫ all here
   too → surprising, more interesting.
2. Then port **Exp 4** (`prune_accuracy_mvbench.py`) + **Exp 5** (`frame_tailindex_select_mvbench.py`
   incl `--with_decoder`) to EgoSchema (same `--tasks EgoSchema`). Does decoder/token
   pruning or frame-conf behave differently at long context?
3. Frame budget: 16 to start (vs 8 MVBench), bump to 32; Exp 4/5 use eager attention →
   memory scales seq², watch it on long clips.

### EgoSchema results log (running; port each script, then record here)
**Ports:** Exp 1/2 scripts had HARDCODED clip lists (unlike Exp 3/4/5 which are `--tasks`-generic).
Added `--data_root/--tasks/--max_clips` to `encoder_redundancy_tailindex.py` (Exp 1) and
`encoder_decoder_tail_crossprobe.py` (Exp 2) — they now build clips from `egoschema.json`
(de-duped, sampled) or fall back to the hardcoded MVBench probes. `--max_pixels` also added to Exp 1
(default 200704; long-form full-res OOMs).

**Exp 1 — DONE (26 EgoSchema clips, `--max_frames 32 --max_pixels 200704`). Core claim HOLDS at
long context: the encoder redundancy negative is NOT an 8-frame MVBench artifact.**
Pooled temporal red 0.485 / Gini 0.200 / γ_keep **−0.232** / γ_copy −0.418; spatial 0.557 / 0.151 /
γ_keep **−0.086** / γ_copy +0.220; multiframe static-to-centroid 0.696, eff-frames/T **0.588**
(γ_keep_t −0.370). Reading: **per-clip every γ_keep ≤ 0** (temporal −0.14..−0.44; spatial ≈0, only
4 clips barely + at noise level) → no separable must-keep elite, same as MVBench.
CAVEATS: (a) pooled spatial γ_copy **+0.220 is a between-clip MIXTURE artifact** — every individual
clip is ≤0 (−0.07..−0.49); don't cite as "EgoSchema has mergeable spatial copies". (b) temporal red
LOWER (0.485 vs MVBench 0.611) = genuine sparse sampling (32 frames over ~180s ⇒ ~11s between merged
positions). NOTE `T=16` from `--max_frames 32` is Qwen's **2:1 temporal-patch merge** (TEMPORAL_PATCH_SIZE
=2, post-merge T=frames/2), NOT a frame-count cap — all 32 frames were decoded (confirmed in Exp 2:
`--max_frames 8` ⇒ encT=4, decM=884=4×13×17). (c) eff-frames/T HIGHER (0.588 vs 0.486, ~1.7× vs 2×
headroom) — sparse sampling already harvested easy temporal redundancy, and what's left is still UNIFORM
(γ_keep_t ≤0). Per-clip frame-budget lever alive + WIDER spread than MVBench (eff-frames 2.8/16
near-static → ~10.7/16).

**Exp 2 — DONE (26 EgoSchema clips, `--max_frames 8 --max_pixels 200704 --band 2-8`). Replicates
MVBench almost point-for-point: the decoder-concentration picture is NOT an 8-frame artifact.**
Pooled mean γ / frac>0 / Spearman-vs-decoder:
- (1) pre-proj ViT attn @E* [qfree]: +0.403 / 26/26 / +0.058
- (2) post-proj geometry [qfree]: −0.298 / 0/26 / −0.115
- (3) post-proj ×query [qguid]: −0.262 / 6/26 / −0.050
- (4) in-LLM decoder @L* [qguid]: **+0.832 / 26/26** (L* mostly layer 3, some 4/6)
Token-level enc.query→decoder Spearman **+0.382 (>0.1 on 26/26)** ≈ MVBench +0.363 — cheap coarse
proxy SURVIVES long-form. Norm control ‖v‖→dec **−0.114 (0/26)** — MORE negative than MVBench's −0.007,
so the norm-confound is ruled out even harder (enc.query corr is genuine, not high-norm-token picking).
Encoder stages still don't predict decoder at clip level (ρ≈0). Caveat: encoder side is only encT=4
(8 frames merged) so stage (2) redundancy is very sparse — but (2) is the flat stage anyway and the
decoder-concentration headline (4) doesn't depend on it.
Exp 2 EXTRA — DONE (`--real_query`, same 26 EgoSchema clips, paired A/B vs generic). Real official
question per clip. Stages 1/2 γ IDENTICAL (query-free — mechanism check passes). Decoder γ* **0.832→0.694**
(still 26/26; L* drifts deeper 4–5) — generic "describe" prompt slightly OVER-states concentration
(pools on generic sinks). **Cheap proxy STRENGTHENS with real q:** token enc.q→dec **+0.382→+0.430**
(26/26); stage-3 clip-level ρ-vs-decoder **~0→+0.539**. Norm control still −0.137 (no confound).
⇒ REVISES writeup's "stage-3 query score fails at clip level" — that ρ≈0 was partly a GENERIC-PROMPT
artifact; a real question makes input-space visual·query track the decoder even clip-level. Strengthens
the CORRELATIONAL proxy story only — Exp 4 still shows proxy ≯ random as a SELECTOR. Writeup Exp 2 +
this STATUS updated.

**Exp 3 — DONE (N=16), REFUTES the "long-form needs accumulation" hypothesis. Frame selection is
NOT a short-clip artifact — the lever is if anything LARGER at long context.** 500-q EgoSchema Subset,
`--num_frames 16 --max_pixels 200704 --blind`: all 58.0 / single_mean 51.4 / single_best 77.4 /
single_worst 22.8 / blind 30.8. **single_best − all = +19.4** (MVBench +12.7): a chosen frame beats
all 16 frames by MORE, on a benchmark built for aggregate reasoning. Predicted single_best ≤ all
(accumulation) — got the OPPOSITE.
DRAW-ROBUST evidence (the load-bearing part, independent of the oracle): all − single_mean = **+6.6**
(16 frames barely beat one typical frame; even smaller than MVBench +8.3) and all − blind = **+27.2**,
blind 30.8 (chance 20%) — vision matters MORE than MVBench (blind 43.8 vs chance 25%), EgoSchema is a
cleaner vision test. CAVEAT: single_best is an oracle over N draws → N=16 mechanically inflates it vs
MVBench N=8 (chance-indep null oracle ~97% @16/5-opt vs ~90% @8/4-opt); observed 77.4 sits BELOW that
floor (frames strongly correlated) but cross-N single_best isn't clean. **TODO: run N=8 control**
(`--num_frames 8`, ~20min) for apples-to-apples single_best vs MVBench. Result: `frame_sufficiency_mvbench.json`.

**Exp 3 EXTRA — DONE. N=8 control + positional analysis.** (a) N=8 control RESOLVES the draw-artifact
worry: at matched N=8, single_best−all = **+21.0** (MVBench +12.7) — selection lever genuinely LARGER
long-form, not oracle inflation; all−single_mean only +1.1 at N=8 (8 frames ≈ 1 typical frame).
(b) POSITIONAL: added `pos_correct`/`best_pos_hist` to frame_sufficiency. Per-position single-frame
acc is FLAT — N=16 spans 49.8–54.8%, N=8 48.8–55.0% (~2 SE spread), no early/late trend across the two;
`middle` frame = 51.2% = random floor. ⇒ temporal position is NOT a usable selector; extends Exp 5's
negative to raw position. (best_pos_hist pos-0 spike = argmax tie artifact, ignore.) Files:
`frame_sufficiency_egoschema_n16.json` / `_n8.json`. Writeup Exp 3 updated (N=8 row + positional para).

**Exp 4 — DONE (N=500). REPLICATES the MVBench closer; N=200 divergence was underpowered noise.**
decoder−random = **−6.0 / −1.0 / +3.4 / +1.6** @ ρ={0.01,0.05,0.10,0.25} (N=200 was −1.5/+2.0/+4.5/+3.0).
Most negative at TIGHTEST budget = the non-causal signature (MVBench −3.3/−3.2/−3.3/+1.1). random is TOP
at ρ=0.01 (38.4). Residual: decoder edges floors at loose ρ=0.10/0.25 (53.4 vs 50.0; 57.0 vs 55.4) — mild
real effect but where pruning saves least (full 60.2). Token selection stays CLOSED for aggressive
pruning long-form. `prune_accuracy_egoschema.json`. Writeup Exp 4 para added.

**Exp 5 — DONE (N=500). REPLICATES; conf is STRONGER long-form.** Anchors single_mean 51.3 / single_best
76.8 / all 59.0 (n_mixed=273). All cheap visual/attn scorers at floor (within-AUC 0.44–0.51); `middle`
51.2 = floor (confirms positional flatness). **conf 57.4 (+6.1, +24% headroom; pooled AUC 0.682, within
0.636)** vs MVBench (+3.5, +17%, 0.650/0.570) — answer frame is FUNCTIONAL not visual, sharper long-form.
NUANCE: dec_xi NEUTRAL here (0.489/0.488) not ANTI-correlated as on MVBench (0.435) — that backward-arrow
was dataset-specific; core "decoder concentration ≠ frame picker" holds. `frame_tailindex_select_egoschema.json`.
Writeup Exp 5 para added.

## ►► Exp 6 — attention CHANGE-POINT frame selector: DONE both datasets (NEGATIVE; complementarity REFUTED)
**Idea (user):** every Exp-5 scorer was a per-frame *level* stat and all failed (AUC ~0.5); the
one untested axis is the *temporal derivative* — a **change-point** in the aggregated
query-guided patch attention marks the event ("what does he do AFTER he stands up" → attention
shifts off the bench). Score positions by `|a[t]−a[t−1]|`, not by `a[t]`.
**Script:** `frame_attn_changepoint_mvbench.py` (new). One full-clip eager forward → debiased
query-guided importance (`debiased_scores_by_layer`) @L* → aggregate per temporal position via
`token_grid` → trajectory `a[T]`. Scorers on it: `cp_post` (change→later "after" pos), `cp_pre`
(→earlier pos), `attn_level` (a[t] itself, the level baseline = level-vs-change contrast), `conf`
(isolated per-position margin, the Exp-5 winner to beat). Anchors + within-clip AUC = Exp-5 harness.
**Native granularity:** Qwen temporal-pairs, so candidates are T=N/2 *positions* (no per-frame
attention); each position scored isolated on its real pair (frames[2t],frames[2t+1]). Coarse ⇒
run at higher N / on EgoSchema. Change-point logic unit-tested (cp_post picks "after", cp_pre "before").
**STILL CORRELATIONAL** (reads attention magnitude — Exp 4 caveat); causal = per-position patch
ABLATION→Δlogit change-point, follow-up only if this survives.
**THREATS (read results against):** (a) query-invariance (Exp 2b) ⇒ most trajectory motion is
scene/motion change (`motion` already failed) — a WIN is most plausible on temporal/action tasks;
(b) attn≠causal ⇒ don't call it causal; (c) if cp_* AND attn_level ~0.5, the temporal-derivative
axis is closed too. **Load-bearing read = PER-TASK within-clip AUC: does cp_* beat `conf` on the
temporal tasks (Action Localization, Moving Direction) where conf lost?**
**EgoSchema RESULT (N=16, T=8 positions, 500q, mixed n=181) — CHANGE-POINT FAILS.** within-clip AUC:
**cp_post 0.480 (BELOW chance)** / cp_pre 0.521 (≈chance) / **attn_level 0.540** / conf 0.578. anchors
single_mean 46.8 / single_best 65.4 / all 59.0 (position-granular, so single_best < Exp-7 frame 77.4).
Reads: (1) the temporal DERIVATIVE does NOT pick the answer position — cp_post below chance ("after the
change" is if anything LESS answerable), cp_pre only a whisker above. (2) the plain attention LEVEL
(attn_level 0.540) *beats both change-points* ⇒ threat (c) fired: level ≥ change, both weak — the
temporal-derivative axis is closed, not just level. (3) conf (0.578) still the only real signal (arc
replicates). All of cp/level within ~1 SE of 0.5 (SE≈0.037 @n=181); conf ~2 SE. attn_level's whisker
(in-CONTEXT query-guided mass) edges Exp-5's isolated dec_mass (0.512) but not meaningfully. Caveat:
T=8 coarse (7 edges), but the flatness says resolution isn't the bottleneck. NOTE overwrote
`frame_attn_changepoint_mvbench.json` with EgoSchema — save separately before the MVBench run.
**MVBench DONE (N=16, T=8, 950q, mixed n=413) — CONFIRMS negative + REFUTES complementarity.** within-clip
AUC cp_post 0.489 / cp_pre 0.481 / attn_level 0.481 (ALL below chance) / conf 0.570. anchors single_mean
55.2 / single_best 76.5 / all 61.8. **The specific bet fails:** on the temporal tasks where conf loses,
cp does NOT pick up slack — Moving Direction all below chance (cp_post 0.37, conf 0.21 anti-corr); Action
Localization conf WINS (0.61), cp at chance. Scattered per-task cells >0.6 (cp_pre 0.645 Object Shuffle,
attn_level 0.594 Fine-grained Action, cp_post 0.571 Moving Count) are INCONSISTENT across the 3 cp/level
variants + within noise given ~57 task×scorer cells (n=14–25, SE~0.05) — no coherent pocket. Result
`../frame_attn_changepoint_mvbench.json` (repo root). **Exp 6 CLOSED both datasets; writeup Exp 6 + short
+ Synthesis pt 4/5 updated. Memory: `changepoint-frame-select-fails`.**
**NEXT (post Exp 6/7):** frame selection is a real lever with NO cheap key (level/change/contrastive all
fail; only conf, weak+expensive). Deployable direction = uniform/recoverable compression (`mod-pivot-recoverable-mlp-skip`)
or task-gated contrastive selector on selection-shaped tasks only (Exp 7 narrow pocket). 7B replication still open.
**Run (on box):**
```
python additional_expts/frame_attn_changepoint_mvbench.py --data_root ../MVBench/ \
  --num_frames 16 --max_samples 40 --max_pixels 200704
# long-form (more temporal resolution): --data_root ~/Experiments/EgoSchema --tasks EgoSchema
```

## ►► Exp 7 — contrastive (SigLIP) frame selector: DONE (MVBench, N=16, 950q). Space was
## PART of Exp 5's failure but NOT a fix — answerability is FUNCTIONAL, not retrieval-shaped.
SigLIP scores raw PIL frames (no Qwen merge) vs question; Qwen supplies per-frame labels + anchors.
Anchors: single_mean 53.6 / single_best 78.4 / all 61.8. **Pooled within-clip AUC (n_mixed=483):**
q_match 0.515 (floor) / **qopt_margin 0.525 (deployable ≈ CHANCE)** / **qgt_oracle 0.582 (ceiling, WITH
label, weak)**. Pick-acc vs floor: q_match +0.4 / qopt_margin +1.8 / qgt_oracle +3.4 (+14% headroom).
- **THE SHARP READ:** qgt_oracle 0.582 (handed the answer text) only **TIES Exp-5 `conf` 0.570** (no
  label). The LLM's functional confidence captures as much frame-answerability as contrastive matching
  does even WITH the answer ⇒ answerability is a FUNCTIONAL/reasoning property, aligned space only
  *proxies* it. Resolves the space-vs-reasoning fork mostly toward **reasoning**.
- **vs Exp 5:** Qwen-internal pre-LLM scorers were 0.44–0.53 (≤chance); the aligned space CLEARS chance
  on selection-shaped tasks ⇒ Exp 5's fail was PARTLY the non-contrastive space — but not fixed
  (deployable qopt_margin still ≈chance pooled, BELOW conf).
- **A-priori bucket split (selection = single decisive frame vs accumulation = existence/count/order),
  NOT fit to AUCs:** qgt_oracle **selection 0.602 vs accumulation 0.529 (Δ+0.073)** — hypothesis holds
  at the CEILING; qopt_margin 0.530 vs 0.511 (both ≈chance) — the deployable proxy does NOT harvest it.
  Naive "object/scene vs temporal" REFUTED: **Object Existence AUC 0.50** (poster-child retrieval task,
  at chance) is the tell — it's accumulation (Exp 3's clean multi-frame task), no single frame to find.
- **Narrow pocket (oracle):** Object Interaction 0.682, Scene Transition 0.670, Action Prediction 0.646,
  Action Antonym/Sequence/Character Order ~0.62 — a coherent selection-shaped cluster, not one lucky task.
- **Trap:** qopt_margin pooled AUC 0.568 > within-clip 0.525 ⇒ it tracks which CLIPS are easy, not which
  FRAME — near-chance as an actual frame selector. Ignore Moving Direction (n_mix=3) & Egocentric Nav (n=8).
- **VERDICT:** lever real but NARROW + not cheaply exploitable (replicates the arc). Report adds a
  selection-vs-accumulation bucket line + `task_bucket_auc` in JSON. Result: `frame_clip_select.json`.
- **EgoSchema DONE (N=16, 500q, mixed n=273): same shape, gap to conf WIDENS.** within-clip AUC
  q_match 0.509 / qopt_margin 0.536 (≈chance) / **qgt_oracle 0.592**. anchors single_mean 51.4 /
  single_best 77.4 / all 58.0. **The diagnostic flips:** on MVBench oracle TIED conf (0.582 vs 0.570);
  long-form it FALLS BEHIND (0.592 vs Exp-5 conf **0.636**). Even handed the answer, contrastive
  matching sees the frame WORSE than the model's own confidence does without it — "functional not
  retrieval-shaped" STRENGTHENS long-form (matches Exp 5's conf-strengthens-long-form). Oracle 0.592
  ≈ MVBench selection bucket 0.602 ⇒ a decisive frame often exists even long-form (Exp 3). `frame_clip_select.json` (overwrote — save separately if keeping MVBench).
- **NEXT:** (1) quantify the narrow pocket's ACCURACY gain (task-gated selector on the selection cluster,
  free from saved JSON); (2) skeptical a bigger CLIP (SigLIP2) helps — ceiling ties/loses to conf even
  WITH the label; (3) writeup Exp 7 + EgoSchema replication para + Synthesis pt 4/5 DONE.

### (setup notes) contrastive frame selector — how to run
**Idea (user):** the one untested PRE-LLM lever — an externally **aligned** image-text space
(Exp 5's `V·q_hat` failed partly because Qwen's projection is NON-contrastive). Score each raw
PIL frame vs the question with a contrastive model, pick argmax. Sidesteps the [f,f] merge
entirely (CLIP sees the full-res frame) ⇒ also the clean control for the "is it the merge?" q.
**Script:** `frame_clip_select.py` (new, self-contained). Qwen (sdpa) only supplies per-frame
answerability LABELS (same [f,f] single-frame protocol) + anchors; the contrastive model scores.
`--clip_model {siglip(default),clip,xclip}`. Scorers: `q_match` (frame·question), `qopt_margin`
(frame·"q+option_j", top1−top2 across options = decisiveness), **`qgt_oracle` (frame·"q+CORRECT
option") — USES THE LABEL, the CEILING diagnostic**. Eval = Exp-5 harness (anchors + within-clip
AUC + PER-TASK). Margin/AUC/pick math unit-tested (numpy).
**RUN THE CEILING FIRST (arc discipline — establish headroom before building a selector):**
`qgt_oracle` within-clip AUC ≫0.5 ⇒ the answerable frame IS visible in aligned space, engineering
`qopt_margin` is worthwhile; `qgt_oracle` ≈0.5 ⇒ contrastive matching fundamentally can't see it ⇒
kills the direction for ~1h CLIP compute. Separates the two hypotheses this thread circled: is
pre-LLM selection failing on **space** (fixable) or on **reasoning/answerability** (not)?
**Expected (from arc):** task split — object/scene tasks (Scene Transition, Object Existence,
Counterfactual, where `conf` won) beat temporal/counting (Action Localization, Moving Direction,
CLIP's weak spots). X-CLIP's video embedding fights per-frame scoring; SigLIP likely the better
frame scorer, X-CLIP tests if video-text training helps. `qopt_margin` match≠answerability =
necessary-not-sufficient (Exp-4 echo). See `oracle-not-feature-predictable`, `oracle-ceiling-and-pareto`.
**Run (on box):**
```
python additional_expts/frame_clip_select.py --data_root ../MVBench/ --num_frames 8 \
  --max_samples 40 --max_pixels 200704 --clip_model siglip
# long-form: --data_root ~/Experiments/EgoSchema --tasks EgoSchema ; video-text: --clip_model xclip
```

**ARC STATUS: all 5 experiments replicate on long-form EgoSchema — DONE + written up.** Top-level framing
reconciled: title subtitle, abstract, Common-setup scope, and a Synthesis point (#5) all now note the
EgoSchema replication (encoder uniform, decoder concentrated but not causally exploitable, frame selection
the real lever + stronger long-form, only conf picks the frame, dec_xi anti-corr is MVBench-specific).

## The question
Video VLMs make ~10^4 visual tokens/clip. Pruning helps only if importance is
**concentrated** (a small separable subset to rank on). We ask: *is it concentrated
enough to rank, and at which pipeline stage (encoder geometry / encoder+query /
decoder attention) does that concentration first appear — and is it exploitable?*

## Setup (constant across experiments)
- Model **Qwen2.5-VL-3B-Instruct**, bf16, eager attention.
- **26 MVBench clips** (tail-index probes) / MVBench tasks (accuracy probes).
- Tail index = **Dekkers–Einmahl–de Haan moment estimator** (`efficient_vlm/utils.py::einmahlHaan`),
  `k_frac=0.10`. γ>0 = heavy tail (rankable elite); γ≤0 = light/bounded (nothing to rank).
- Data root on the box: `/home/ubuntu/Experiments/MVBench` (a.k.a. `../MVBench`).
- Caveat carried throughout: cosine scores pull γ≤0, softmax attention pulls γ>0 —
  trust **relative** and **correlational** results, not bare signs.

## Scripts
| script | what it does |
|---|---|
| `encoder_redundancy_tailindex.py` | Exp 1: temporal/spatial redundancy tails (lag-1 + multi-frame) |
| `encoder_decoder_tail_crossprobe.py` | Exp 2 (2×2 stages) + Exp 2b (`--query_overlap`; token-level corr always on) |
| `frame_sufficiency_mvbench.py` | Exp 3: 1-frame vs all-frames accuracy |
| `prune_accuracy_mvbench.py` | Exp 4 **THE CLOSER** (DONE): γ→accuracy, keep top-ρ by scorer, measure MVBench acc |
| `frame_tailindex_select_mvbench.py` | Exp 5 (DONE): can a per-frame tail index PICK the right frame? selection acc + within-clip AUC discrimination |
| `frame_attn_changepoint_mvbench.py` | Exp 6 (SET UP, pending): does a temporal CHANGE-POINT in query-guided patch attention pick the position? cp_post/cp_pre vs attn_level/conf |
| `frame_clip_select.py` | Exp 7 (SET UP, pending): can a CONTRASTIVE model (SigLIP/CLIP/X-CLIP) pick the frame PRE-LLM? q_match/qopt_margin + qgt_oracle CEILING; Qwen supplies labels only |

## Results so far

**Exp 1 — encoder redundancy is UNIFORM, not rankable.** Pooled temporal red 0.611 /
Gini 0.339 / γ_keep −0.311 / γ_copy −1.083; spatial 0.530 / 0.145 / −0.224 / −0.223.
Multi-frame: static-to-centroid 0.767 (γ_copy −1.029), eff-frames/T 0.486 (γ −0.968),
lag-decay ~flat. All tails ≤0 on every clip. **Tail index contradicts Gini** (higher
Gini, more-negative γ): redundancy is a *graded field*, global + uniform (~2× headroom),
no separable subset at any granularity. → uniform temporal compression only; per-clip
frame budget is the only adaptive knob (eff/T varies 0.32–0.70 across clips).

**Exp 2 — concentration is in the DECODER (2×2, 8 frames, max_pixels=200704).**
mean γ / frac>0 / Spearman-vs-decoder:
- (1) pre-proj ViT attn @E* [qfree]: **+0.466 / 26/26 / −0.015**
- (2) post-proj geometry [qfree]: −0.354 / 0/26 / +0.004
- (3) post-proj ×query [qguid]: −0.117 / 11/26 / −0.018
- (4) in-LLM decoder @L* [qguid]: **+0.802 / 26/26 / —** (L* mostly layer 3)

**Exp 2b — REVISION (this overturned two earlier claims; see writeup §revision).**
- **Token-level** enc.query→decoder Spearman = **+0.363, 26/26 clips >0.1** (up to +0.60).
  The clip-level ρ≈0 was the *wrong granularity*. A cheap input-space visual·query score
  **does** moderately predict decoder importance (~13% rank variance). No norm confound
  (||v||→dec ≈ −0.007 pooled).
- **Query-dependence** (8 clips, 3 divergent queries): top-K Jaccard **0.795 vs random
  0.053**, cross-query score Spearman **+0.959**. The decoder's kept-set is ~**FIXED /
  query-INVARIANT** — a visual-saliency prior, *not* question-adaptive selection.
- Net: decoder importance is **stable + heavy-tailed + cheaply (coarsely) predictable**
  → good for a training-free pre-LLM pruner; little upside from query-adaptivity.

**Exp 3 — frame SELECTION > accumulation (950 questions, N=8).** all 62.2% /
single_mean 53.9% / single_best (oracle) 74.9% / single_worst 31.8% / blind 43.8%.
Multi-frame adds only +8.3 over a typical frame, but a *chosen* frame beats all-frames
by +12.7. Null model (independent frames) predicts oracle ≈99.8% vs observed 74.9% ⇒
~25% of questions genuinely need multi-frame. Object Existence is the clean multi-frame
task; Action Localization: video *hurts* (all 30% < blind 50%).

**Exp 4 — THE CLOSER: decoder concentration is NOT causally exploitable (570q, N=8,
4 budgets).** Keep top-ρ visual tokens by each scorer at matched budget, prune at input
(mRoPE positions preserved), read MVBench answer. Same prune mechanism all scorers → gap
is the scorer's. Overall (full 62.5%), decoder / encquery / random / uniform:
- ρ=0.01: 46.0 / 50.0 / 49.3 / 46.3   (decoder−random **−3.3**)
- ρ=0.05: 50.9 / 54.2 / 54.0 / 55.3   (decoder−random **−3.2**)
- ρ=0.10: 54.4 / 54.9 / 57.7 / 56.1   (decoder−random **−3.3**)
- ρ=0.25: 58.9 / 56.3 / 57.9 / 59.8   (decoder−random +1.1)

**Monotone tell (strengthens the negative):** an exploitable scorer should help MORE as
budget tightens; decoder does the OPPOSITE — ~3pts BELOW random at 1–10%, parity only at
25%. Signature of a non-causal signal; closes the "works at aggressive budgets" escape
hatch. SE≈2.1 at N=570 (each gap ~1.5 SE; the *consistency across 4 budgets* is the
result). Content-free floors (random/uniform) are the **top** pruned setting at EVERY
budget — *spread* beats *importance* (Exp 1's uniform field, now in accuracy). encquery−
random = +0.7/+0.2/−2.8/−1.6 (level then below). Within a budget all 4 scorers within
~3pts, all below full (gap ~4pts@0.25 → ~16pts@0.01). Caveat: decoder score from
full-context forward applied as INPUT prune (slightly conservative vs true in-LLM FastV)
— but input prune is the only prefill-saving variant, and encquery (pure input scorer,
no confound) also loses to random. **Both deployable variants fail.**

## RESOLVED: the fork landed in the BOTTOM branch → PIVOT
| outcome | meaning | verdict |
|---|---|---|
| decoder≫random & encquery≈decoder | exploitable + cheap proxy works | ✗ not this |
| decoder≫random & encquery≈random | exploitable only in-LLM | ✗ not this |
| **decoder≈random** | **γ>0 necessary NOT sufficient (attention≠causal)** | **✓ THIS** |

**γ>0 was necessary but not sufficient: attention concentration is not causal
importance.** The importance-ranked token-selection thread is CLOSED — no training-free
pre-LLM *selection* pruner to be had here. Remaining levers: (a) **frame selection**
(Exp 3, `frame-selection-is-the-lever`) — largest effect measured; (b) **uniform /
recoverable** compression (Exp 1 + `mod-pivot-recoverable-mlp-skip`) — blind ~2×
temporal reduction + per-layer skip where dropped tokens stay keys. Selection at the
TOKEN level is not the lever; FRAME-level selection + cheap uniform compression are.
Result saved: `prune_accuracy_mvbench.json`. Writeup §Exp 4 + revised Synthesis.

**Exp 5 — CHEAP frame selection also fails: tail index does NOT pick the frame (570q,
N=8).** Score each frame on its own encoder tokens ([f,f] clip), pick argmax, measure acc
vs anchors + per-frame discrimination AUC. Overall: all 62.8 / single_best 75.6 (oracle) /
single_mean 54.6 (random floor). Scorers: xi_encq 53.1 / xi_sal 54.9 / mass_encq 56.0 /
motion 54.2 — all within ±1.6 of random, capturing −8%..+6% of the 21pt oracle headroom.
Discrimination (n_mixed=248): pooled AUC 0.50–0.51; **within-clip AUC** (controls
difficulty) xi_encq **0.462** / xi_sal 0.488 / mass_encq 0.526 / motion 0.487 — all ≈0.5,
xi_encq below. **High tail index does NOT mark the answerable frame.** mass ≥ concentration
again (as Exp 4 at token level); tail index is the worst, mildly anti-selective. Frame
selection is a real lever (Exp 3) but NOT cheaply exploitable — needs an LLM-in-the-loop
signal (per-frame forward → no saving). Result: `frame_tailindex_select_mvbench.json`.
Writeup §Exp 5 + Synthesis pt 4. Memory: `frame-tailindex-not-selective`.

### NEXT STEPS (post-pivot)
1. **[DONE]** Strong in-LLM per-frame references (Exp 5 `--with_decoder`, 570q, eager).
   Overall single_mean 54.6 / single_best 75.4 / all 62.3. **dec_xi 52.8, dec_mass 53.2 →
   decoder concentration FAILS to pick the frame; dec_xi is ANTI-correlated (pooled AUC
   0.435, Δmean −0.038, within 0.468) — a more concentrated frame is LESS answerable.**
   **conf 58.1 (+3.5 vs random, +17% headroom) — the ONLY working signal: pooled AUC 0.650,
   within-clip 0.570, top1-on-mixed 59.3%.** Answer to "how is the best frame different":
   it's the one the LLM can answer confidently (functional/semantic), NOT visually special.
   But conf is modest (1/6 of headroom) + expensive (per-frame forward → no saving). Task
   split: conf wins on Scene Transition (+14: 90.0 vs 75.8), Counterfactual +15, Character
   Order +10, Action Antonym +9; loses on Action Localization −7, Moving Direction −1
   (temporal/counting). Writeup §Exp 5 rewritten. → next: cheap PROXY for conf?
2. **[PROPOSED — diagnostic first, no GPU]** Characterize `gt_frames` from the saved
   `frame_tailindex_select_mvbench.json`: within the ~44% mixed clips, is the good frame a
   NEEDLE (1 of 8) or ABUNDANT (5–6 of 8)? + temporal position of correct frames + per-task.
   Decides whether a better frame selector can even exist / has headroom. Aggregates imply
   ~32% all-frames-correct, ~24% none, ~44% mixed — the diagnostic resolves the mixed shape.
3. **[PROPOSED — cheap-proxy-for-conf]** External contrastive CLIP/SigLIP image×question
   match as a frame scorer (the one signal Qwen's non-contrastive space can't give; Exp 5's
   xi_encq/mass_encq were crippled by using Qwen's non-aligned space). Test whether it
   recovers conf's win, specifically on the object/scene tasks where conf works (Scene
   Transition, Object Existence, Counterfactual). CLIP is weak on verbs/counting/temporal, so
   expect a task split. Frame-level = CLIP's retrieval sweet spot (normal similarity, NOT
   LiteLVLM's token-level reversal). In scope (frame pruning). Add as a scorer to Exp 5.
4. **[PROPOSED — importance×diversity DPP]** Add `diverse` (feature-space FPS, importance-off)
   + `impdiv`/`dpp` (joint) scorers to `prune_accuracy_mvbench.py` (uses anchor_layer_prune
   greedy / CDPruner). The one selection variant Exp 4 didn't test: does content-aware
   diversity beat uniform's positional spread? impdiv > uniform → selection lives; ≈ uniform
   → prune loss is intrinsic. In scope (pruning). [user flagged this gap]
5. MoD recoverable MLP-skip (`mod-pivot-recoverable-mlp-skip`): skipped tokens stay keys,
   attention dense; measure accuracy vs FLOPs. Deployable direction if selection stays closed.
6. (dropped) anchor_layer_prune greedy diversity as a TOKEN selector by IMPORTANCE — Exp 4
   shows spread ≈ ceiling; superseded by #4 which tests diversity properly.
7. (dropped-cheap) pre-LLM frame selector by concentration/query/motion — Exp 5 refutes it.

## Gini question (answered — do NOT rewrite in Gini)
γ is strictly better for the ranking question and *caught the false positive Gini gave*
(Exp 1). Writeup already reports both; the Gini-vs-γ contrast is itself a result.
"Importance + redundancy both matter for selection" is about a joint importance×diversity
objective (CDPruner DPP / `anchor_layer_prune` greedy) — neither Gini nor γ measures
diversity, so the right metric there is **downstream accuracy** (= the running test),
not Gini.

## Hardening backlog (cheap, do regardless of fork)
1. **[DONE — PASSED]** k_frac sensitivity (γ at 0.05/0.10/0.20): signs STABLE. Load-bearing
   stages identical sign+frac>0 at every k_frac — (1) ViT +0.41/+0.47/+0.53 (26/26 all),
   (2) geometry −0.30/−0.35/−0.48 (≤1/26), (4) decoder +0.75/+0.80/+0.88 (26/26 all).
   Magnitudes drift monotonically with k_frac (expected), no flips. Only (3) ×query has a
   k_frac-sensitive MAGNITUDE (mean −0.66/−0.12/−0.06) but stable sign+frac>0 (~10/26); no
   conclusion depended on it. Headline concentration claim is not a knob artifact. Cmd:
   `encoder_decoder_tail_crossprobe.py --k_frac_sweep 0.05 0.10 0.20 --max_pixels 200704`.
2. **[DONE — PASSED, STRENGTHENS]** Real-query overlap (54/60 clips; 6 skipped for data
   layout: nturgbd .avi absent, tvqa = frame dirs — not a code bug). 3 real MVBench
   questions/clip (own + 2 mismatched-TASK, i.e. an irrelevant question). **top-K Jaccard
   0.658 vs random 0.053 (~12×), score Spearman +0.900.** Query-invariance CONFIRMED with
   real, task-divergent questions (vs original 8-clip/3-generic: Jaccard 0.795 / Spearman
   0.959). Real questions move the set slightly MORE (0.658<0.795) = a modest adaptive
   residual, concentrated on CLEVRER multi-object tasks (Moving Attr/Dir/Count, Object
   Existence lowest ~0.42–0.48) where the question must disambiguate; real-world action
   clips most invariant (~0.92–0.95). Dominantly a fixed saliency prior. Cmd:
   `encoder_decoder_tail_crossprobe.py --real_queries --data_root ../MVBench/ --max_per_task 3 --n_real_queries 3 --max_pixels 200704`.
3. 7B replication of the cross-probe headline.
4. Null calibration for the bounded-score γ priors.

## Persistent memory (auto-loaded each session) — related notes
`temporal-redundancy-not-heavy-tailed`, `frame-selection-is-the-lever`,
`mod-pivot-recoverable-mlp-skip`, `oracle-ceiling-and-pareto`,
`oracle-not-feature-predictable`, `budget-uses-moment-estimator`, `fastv-baseline`,
`cdpruner-conditional-diversity`.
