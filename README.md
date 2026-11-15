# SmolDataEnvs on 8 GB: SFT distillation vs. GRPO for a 0.6B data-analysis agent

Repository: <https://github.com/kustcl/smoldataenvs-qwen3-0.6b-rtx4060> ·
SwanLab: <https://swanlab.cn/@cl666/SmolData-GRPO/v1/jwt6a9/overview>

Everything here runs on one consumer laptop (Windows 11, RTX 4060 Laptop, 8 GB VRAM) with
**Qwen3-0.6B**. It contains a local agent sandbox, an evaluation harness, the SFT and GRPO
training scripts, and the raw results of the experiments below.

**Headline result: on this hardware, SFT (distillation from recorded trajectories) is by far the
biggest win, and GRPO made the model worse.** The repository is published mainly as an honest,
reproducible negative result for small-model RL in this regime, together with the tooling that
produced it.

**This is a solo project on one laptop — help is welcome.** If you have ideas or hardware to spare,
see [*We are asking for help*](#we-are-asking-for-help) below; the most useful contributions are a
parallel tool loop, a batched screening scheduler, and a 2B/4B reproduction with `num_generations=8`.

## Results (same 90 held-out tasks, same decoding)

Evaluation: `eval` split, easy + medium (16 + 74 tasks), agent mode, greedy decoding, ≤12 model
calls per task, 1024 tokens per call, Docker sandbox.

| Stage | v1 (lenient grader) | v2 (strict checker) | Submits an answer | Hits the 12-turn cap |
|---|---|---|---|---|
| Qwen3-0.6B, untrained | 0.0% | 0.0% | 0% | — |
| **SFT (1 epoch, 411 steps)** | **25.6%** (23/90) | **21.1%** (19/90) | **63%** | 29 |
| GRPO from SFT, 17 steps | 20.0% | 16.7% | 53% | 35 |
| GRPO from SFT, 138 steps | 17.8% | 15.6% | 53% | 37 |
| GRPO checkpoint-50 | ≈ below SFT | ≈ below SFT | — | — |

By tier (v1 / v2):

| Model | easy (16) | medium (74) | unseen Kaggle dataset (21) |
|---|---|---|---|
| SFT | 56.3% / 43.8% | 18.9% / 16.2% | 4.8% / 4.8% |
| GRPO-138 | 50.0% / 43.8% | **10.8% / 9.5%** | 9.5% / 9.5% (2/21, noise) |

Paired per-task comparison (both runs, identical settings):

- overall v1: 15 tasks both correct, **8 SFT-only**, **1 GRPO-only**, 66 neither → McNemar **p = 0.039**
- medium v1: 8 both, **6 SFT-only, 0 GRPO-only** → **p = 0.031**
- overall v2: 13 both, 6 SFT-only, 1 GRPO-only → p = 0.125

Training reward went **up** (mean strict reward 0.40 → 0.56 over the 138 steps) while held-out
accuracy went **down** — the classic "optimising the sampler, not the model" signature.

**Training curves (SwanLab):** <https://swanlab.cn/@cl666/SmolData-GRPO/v1/jwt6a9/overview> —
the GRPO runs' reward, submission rate, zero-variance fraction and step timings. Raw summaries and
the 138-step training log are also in [`reports/evals/`](reports/evals/).

## What is in this repository

| File | Purpose |
|---|---|
| `smolenv_local.py` | Local sandbox (Docker / bwrap / macOS sandbox-exec), task staging, tool execution, grading (v1 + v2) |
| `checker_v2.py` | Strict answer checker used as the RL reward |
| `eval_harness.py` | Agent-mode evaluation for local HF models or any OpenAI-compatible endpoint; batched local generation |
| `train_sft_local.py` | LoRA SFT on the recorded agent trajectories |
| `train_grpo_agent.py` | GRPO with TRL's `environment_factory` (multi-turn bash tool, answer via `/workdir/answer.txt`) |
| `train_common.py` | Profile loading, task-pool filtering, SwanLab wiring |
| `train_plan.yaml` | All hyperparameters and the plan of record |
| `scripts/prescreen_until.py` | Screens train tasks until N "mixed" (non-zero-variance) tasks are found |
| `scripts/prescreen_until.py`, `scripts/stage_task_files.py`, `scripts/merge_lora.py` | Data staging / pool building / adapter merging |
| `reports/rl_degradation_analysis_138.md` | Detailed post-mortem of the 138-step GRPO run (Chinese) |
| `grpo_pool_150/` | The 138-task pool (`grpo_pool_150.json`), screening report and raw rollouts |

## Upstream and provenance

This is **not** a from-scratch implementation. It is the **Windows / 8 GB port** of the
SmolDataEnvs training protocol, and the lineage is:

| Source | What it provides |
|---|---|
| [`adithya-s-k/FineEnvs`](https://github.com/adithya-s-k/FineEnvs) @ `08a5622`, `04-smoldataenvs/scripts/` | The reference scripts (`train_grpo.py`, `train_sft.py`, `eval_pass1.py`, `rollout.py`); every default in `train_plan.yaml` comes from here, and every deviation is listed under `deviations:` in that file |
| [`FineEnvs/SmolDataEnvs`](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs) @ `b2bf3564` | 5,394 verifiable data-analysis tasks and the official (lenient) grader |
| [`FineEnvs/SmolDataEnvs-sft`](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-sft) @ `a9fa95ab` | 4,677 recorded multi-turn bash trajectories used for SFT |
| `README.dgx-spark-4b.md` (this repo) | The DGX Spark / Qwen3.5-4B variant of the same protocol, from which this port was adapted |
| `vendor/fineenvs_rollout.py` (Apache-2.0), `vendor/smoldataenvs_grader.py` (MIT) | Vendored reference code; see [`NOTICE`](NOTICE) |

What this port changes: a local sandbox (`smolenv_local.py`) instead of Hugging Face's hosted
sandboxes, a strict checker (`checker_v2.py`) used as the RL reward, a batched local-HF evaluator,
Windows-specific plumbing, and Qwen3-0.6B instead of 2B–4B models.

## Hardware and software

| | |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop, 8 GB GDDR6, 24 SMs (Ada, CC 8.9) |
| CPU / RAM | i7-12650H, 16 GB |
| OS | Windows 11 |
| Sandbox | Docker Desktop (WSL2), one throwaway container per tool call |
| Stack | PyTorch 2.13 + CUDA 12.6, transformers 5.17, TRL 1.14, PEFT, datasets |

Measured on this machine: Qwen3-0.6B generates ~19 tok/s for a single sequence, ~85 tok/s
aggregated at batch 4 (2.65 GiB), ~55 tok/s at batch 8 (4.17 GiB).

## SFT

**Data.** `FineEnvs/SmolDataEnvs-sft`, 4,677 multi-turn bash trajectories
(easy 1,402 / medium 2,640 / hard 635). Two filters, both applied before training:

1. drop rows with two consecutive `assistant` turns (511 rows — an artefact of the recorder);
2. drop rows longer than **2,048 tokens** instead of truncating them (truncation cuts the
   `answer.txt` write). This keeps ~3,300 rows: easy ~1,300, medium ~1,850, hard ~260.

**Config** (`train_plan.yaml → sft.agent_format`):

| | |
|---|---|
| LoRA | r = 16, alpha = 32, dropout 0.05, `all-linear` |
| LR | 1e-4 (cosine), 1 epoch = **411 steps** |
| Batch | per-device 2 × grad-accum 4 = 8 |
| Max length | 2,048 (long trajectories dropped, never truncated) |
| Precision | bf16 + gradient checkpointing |
| Assistant-only loss | yes (TRL `assistant_only_loss`), Qwen3 training chat template |

The untrained 0.6B never writes an answer file at all (0/16 easy). After one epoch of SFT it
finishes the agent loop and reaches 25.6% / 21.1% on the 90-task held-out set — the single
largest jump in this project.

## GRPO

**Task pool.** GRPO needs prompts the policy solves *sometimes*; all-correct and all-wrong groups
carry zero advantage. `scripts/prescreen_until.py` samples each candidate 4× at T = 0.8 with the
SFT policy and keeps tasks with 1–3 correct answers:

```
4,151 eligible train tasks (easy + medium, complete inputs, ≤ 100 MB)
  → 370 screened → 138 "mixed" (easy 63 / medium 75), 183 never solved, 49 always solved
  → 138-task pool  (≈ 2.6 h at ~3 rollouts/min with batched generation)
```

**Training config** (`train_plan.yaml → profiles.agent_qwen3_06b_lora`):

| | |
|---|---|
| Algorithm | GRPO, `loss_type="dapo"`, `epsilon=0.2`, `scale_rewards="group"`, `beta=0` (no reference model / KL) |
| Rollouts per update | **1 prompt × 4 generations** (`per_device=1`, `steps_per_generation=4`, `num_generations=4`) |
| Reward | strict checker (v2) on `/workdir/answer.txt`, weight 1 + `0.05 × submitted` |
| LoRA | r = 16, alpha = 32 |
| LR | 3e-6 |
| Budget | `max_completion_length=3072` (includes tool output), 2048 per turn, ≤ 6 tool iterations |
| Steps | 138 = one pass over the pool; 3.8 h; 7.9/8.0 GiB peak VRAM |

**Why it regressed (analysis, not guesswork):**

1. **4 rollouts per update** is below what GRPO needs for a stable group advantage (TRL/GRPO
   practice: ≥ 8). The 8 GB card cannot hold more while keeping 3k-token multi-turn transcripts.
2. **No KL anchor** (`beta=0`): nothing keeps the policy near SFT, and the tiny-batch gradient
   noise dominates. Degradation was already complete by step 50.
3. **Sampled training vs. greedy evaluation**: the training reward rose while held-out greedy
   accuracy fell — the policy learned "keep trying" (which sometimes pays off under T = 0.8) and
   lost the SFT habit of stopping and writing an answer. Submissions dropped 63% → 53%, and the
   number of episodes that hit the 12-turn cap rose 29 → 37.
4. **The loops predate RL**: 25/90 SFT transcripts already repeat the *same failing command* up to
   11 times. GRPO raised that to 31/90 but did not create it; the fix is in the SFT data
   (error-recovery trajectories), not in more RL.
5. **Narrow pool**: 138 tasks, each seen once per epoch; the training-reward curve is measured on a
   different task subset every step, so it cannot be read as generalisation.

Full post-mortem: [`reports/rl_degradation_analysis_138.md`](reports/rl_degradation_analysis_138.md) (Chinese).

## Evaluation protocol

- `eval` split, `easy,medium` = 90 tasks; answers graded by the dataset's lenient grader (v1) and
  by the strict checker (v2). v2 is a subset of v1; the gap is formatting/tolerance.
- Agent mode: the model works in a bash container, ≤ 12 model calls, 1024 tokens per call,
  greedy, "thinking" disabled.
- Local HF generation is **batched** (`--hf-batch-size 4`, concurrency 8): ~3× faster than the
  original one-sequence path. Batching changes floating-point reduction order, so **numbers are
  only comparable within the same batching setting** — all results above were produced with
  batch = 4, concurrency = 8.

## Reproducing

```bash
# 0) stage the task files (needs the Hugging Face bucket listing; see scripts/stage_task_files.py)
python scripts/stage_task_files.py

# 1) SFT (LoRA on the recorded trajectories)
python train_sft_local.py --model models/Qwen3-0.6B --out runs/sft-0.6b-2048-e1 --max-length 2048

# 2) evaluate SFT on easy+medium
python eval_harness.py run --hf-model runs/sft-0.6b-2048-e1/merged \
  --split eval --tiers easy,medium --mode agent --sandbox win-docker \
  --concurrency 8 --hf-batch-size 4 --out runs/eval-sft-easy-medium

# 3) screen a mixed-task pool for GRPO (stops at --target or --max-hours)
python scripts/prescreen_until.py --target 150 --tiers easy,medium --n-samples 4 \
  --concurrency 16 --hf-batch-size 6 --max-hours 8 --out grpo_pool_150 \
  --pool-out grpo_pool_150/grpo_pool_150.json

# 4) GRPO from the SFT model
python train_grpo_agent.py --profile agent_qwen3_06b_lora \
  --init-from runs/sft-0.6b-2048-e1/merged --swanlab --run-dir runs/grpo-agent-plus150 \
  --set task_pool_file=grpo_pool_150/grpo_pool_150.json --set max_steps=138 \
  --set save_steps=25 --set learning_rate=3e-6 --set reward_submitted_weight=0.05
```

No-GPU smoke tests: `python eval_harness.py selftest` (mock model, real sandbox) and
`python train_grpo_agent.py --profile agent_qwen3_06b_lora --dry-run`.

## What we would do differently

- **Distil instead of RL at this scale.** SFT-style distillation from a stronger model's
  transcripts is the reliable lever for a 0.6B model; GRPO needs a model with enough headroom to
  explore, more samples per prompt, and a KL anchor.
- Train SFT with **4,096-token context** (2,048 drops 25% of medium and 59% of hard trajectories —
  exactly the long error-recovery and finishing behaviour the model is missing).
- Add trajectories that show **recovering from an error and submitting within N turns**; the
  dominant failure mode is repeating the same failing command until the turn cap.
- If RL is retried here: `beta≈0.02` (KL), `lr=1e-6`, ≤ 50 steps, and evaluate sampled
  (T > 0) as well as greedy.

## Would other small models behave the same?

We only ran **Qwen3-0.6B**. It would be wrong to read this as "RL does not work for small
models" — but it is also wrong to read it as "only this model fails". Our reading, split by what
we think is a property of the *setup* versus a property of the *model*:

**Likely independent of the model (these are properties of the training regime):**

- **4 generations per optimizer step** is below what GRPO needs for a stable group advantage.
  Any model gets a noisy gradient at this batch size; the direction of the noise is what changes.
- **`beta = 0`** leaves nothing anchoring the policy to SFT. Any LoRA fine-tune will drift; drift
  is not a Qwen property.
- **Training at T = 0.8 and evaluating greedily** rewards behaviour that only pays off under
  sampling. Every model that learns "keep trying" during sampling will look worse at T = 0.
- **A 138-task pool, each task seen once**, cannot support generalisation; the training-reward
  curve is measured on a different subset every step.

**Model-dependent (why other ~0.5–1B models would differ in degree, not in kind):**

- **Base pass rate.** Screening produces "mixed" tasks only where the model sometimes succeeds.
  Qwen3-0.6B solves ~25% of easy+medium, giving a ~1/3 mixed rate; a stronger 1B model yields more
  usable tasks (easier pool), a weaker 360M model fewer (harder pool, more zero-variance groups).
- **Tool-loop competence.** The failure we saw — repeating the *same failing command* up to 11
  times — is a capacity/repetition phenomenon. It is common in sub-1B models and rarer as models
  grow; it is also visible in the *SFT* numbers (25/90 transcripts), so it predates RL.
- **Chat-template quirks.** Qwen3's `<think></think>` markers and `<|im_end|>` template caused a
  silent train/eval mismatch here; Llama-3.2, SmolLM2 or Gemma would fail differently, not less.
- **Sensitivity to format drift under RL** (dropping spaces, appending `%`) is tokenizer- and
  SFT-data-dependent.

**Testable predictions — we would rather have them tested than asserted** (≈1 day per model on
this laptop; only `--model` changes):

- **P1.** Qwen2.5-0.5B or Llama-3.2-1B, same protocol, shows the same sign (RL ≤ SFT), with a
  magnitude roughly tracking the base pass rate.
- **P2.** At ≥2B with `num_generations = 8` and `beta ≈ 0.02`, RL becomes neutral-to-positive —
  i.e. what fails here is the *regime* (batch size, no KL, tiny pool), not "small models" as a class.
- **P3.** SFT data with error-recovery trajectories reduces the loop rate at every model size —
  the data fix is model-agnostic.

If you have a GPU and a day, P1 is the most useful experiment anyone could run against this
repo — see *Reproducing*, the model path is the only change.

## We are asking for help

This is a solo effort on a single 8 GB laptop, and **we would genuinely welcome help from the
community** — both on the engineering and on the science. Concrete things that would move this
forward:

1. **Parallel / asynchronous tool execution.** TRL 1.14's `_tool_call_loop` executes tool calls
   serially across rollouts (only multiple calls inside one rollout are gathered). An async or
   thread-pool variant would cut 10–25% of step time. We tried the obvious `async def` route; it
   does not help because TRL gathers per rollout.
2. **A batched-generation scheduler for screening.** `eval_harness.py` now batches 4–6 local HF
   requests, which made screening ~3× faster; a scheduler that merges requests across tasks
   (continuous batching) would cut the 2.6-hour pool build further.
3. **GRPO with a KL anchor on 8 GB.** `beta > 0` needs a reference model (~1.2 GiB). We could not
   fit it alongside 3k-token multi-turn transcripts; a cheaper anchor (adapter snapshots, periodic
   refresh, early stopping with validation) would be valuable.
4. **Linux / bwrap parity.** Re-run the numbers on Linux and tell us whether the regression
   reproduces; we suspect it does, but Windows-specific paging and Docker latencies could matter.
5. **Better SFT data.** Trajectories that show *recovering from a tool error and submitting within
   N turns* are the model's biggest gap. If you have them, we will test them.
6. **A 2B / 4B reproduction.** If you have more VRAM, running the same protocol with
   `num_generations=8` and `beta>0` would separate "small-model limit" from "tiny-batch RL".

If you want to help, open an issue or a pull request, or just tell us which experiment you would
like to run — the exact commands are in *Reproducing* above, and we are happy to adapt them.

## Pitfalls we hit (so you don't have to)

Everything below cost us real time; each one is either fixed in the code or spelled out in the
commands above.

**Environment**

- **Docker Desktop silently off.** Without the engine, `detect_backend()` returns `none`, tool
  calls run on the host and die with `ModuleNotFoundError: No module named 'resource'` (a Unix
  module), the model retries until the turn cap and **every reward is 0** — with no error in the
  training log. Check `smolenv_local.detect_backend()` before a long run.
- **HF downloads time out** from this network; `HF_ENDPOINT=https://hf-mirror.com` plus
  `HF_HUB_DISABLE_XET=1` fixes it.
- **0-byte staged files.** An aborted download leaves an empty CSV that grades silently; the
  sandbox now treats a 0-byte input as a missing file (task becomes ungraded, never scored).
- **SwanLab 0.10.5** removed `swanlab.package.has_api_key`, raises from `get_run()` before init,
  reads the project from `SWANLAB_PROJ_NAME`, and crashes the GBK console on emoji output
  (`sys.stdout.reconfigure(encoding="utf-8")` fixes the last one).
- **`python -I` drops the cwd** from `sys.path`, so a repo-local import fails; add `sys.path`.

**Data and grading**

- **Consecutive assistant turns.** 511 of 4,677 SFT rows contain two assistant turns in a row
  (a recorder artefact); the model learns to answer itself. Drop them.
- **Never right-truncate agent trajectories** — truncation removes the final `answer.txt` write.
  Filter by length, drop whole rows.
- **Length filtering must use TRL's training template.** The stock Qwen3 template drops the empty
  `<think></think>` block from historical turns and under-counts by 8–20 tokens; and eval prompts
  must be rendered with the same template or turn ≥2 diverges from training.
- **`exclude_modules="visual.*"` excludes nothing** in PEFT (patterns are full-matched); use
  `.*visual.*`.

**TRL semantics (the ones that bit us)**

- **`grad_accum` does not add prompts.** With `per_device=1`, `grad_accum=4`,
  `num_generations=4`: `steps_per_generation=4`, `generation_batch_size=4` *sequences* = **1
  prompt × 4 generations per optimizer step**. Proof: `epoch` rises by `1/pool_size` and
  `num_tokens` by ~4k per step. Grad accumulation only splits the group into 4 micro-steps.
- **Tool calls are serial across rollouts.** `async def` tools are gathered *within* one rollout
  only; making `bash` async does not parallelise anything here.
- **`reward_weights` covers only your functions.** TRL appends the environment's `get_reward`
  with weight 1, so `[0, 0.05, 0]` means "strict reward + 0.05 × submitted".
- **YAML 1.1 parses `3e-6` as a string**, which surfaces as
  `TypeError: '<=' not supported between instances of 'float' and 'str'` inside AdamW. Write
  `3.0e-6`, or let `--set` coerce numeric strings (this repo does).
- **A missing `task_pool_file` used to fall back to random train tasks silently**; it now raises.

**Evaluation**

- **A GRPO adapter's base is the merged SFT model**, not `models/Qwen3-0.6B`. Loading it on the
  base model produces a model that never writes an answer and finishes 90 tasks in 80 seconds —
  it looks like "the base model", not like an error.
- **Batching changes greedy results.** bf16 batched matmul rounds differently; on a 0.6B model
  many next-token distributions are near 50/50 ties, so trajectories flip. Numbers are comparable
  **only within the same batching setting** — re-baseline when you change it (8 of 16 easy tasks
  flipped between batch 1 and batch 4).
- **VRAM is the wall.** Batch 4 at 4k-token prompts is ~2.7 GiB in isolation, but inside the eval
  harness (long transcripts, 8 concurrent tasks) the observed peak was **7.9/8.0 GiB**; the
  screening configuration (batch 6, ≤7 turns) peaked at 6.5 GiB. Batch 8 OOMs. Training peaks at
  7.95/8.0 GiB while generating only 4 sequences.
- **`--concurrency` parallelises sandboxes, not generation** unless batching is enabled; the local
  HF client used to serialise everything behind one lock (~19 tok/s).

## Limitations

- One model (Qwen3-0.6B), one seed, one 90-task held-out set (95% CI ≈ ±9 pp per run); every
  number above is a single run.
- The GRPO pool (138 tasks) was screened with the SFT policy at T = 0.8; the pool is therefore
  tuned to that policy, which favours RL rounds 1 but biases later rounds.
- The Windows Docker backend is what the numbers were produced with; Linux/bwrap results may
  differ slightly (different tool latencies, no WDDM paging).

## License and credits

Code in this port is under the repository LICENSE. Task data comes from
[FineEnvs/SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs) and
[FineEnvs/SmolDataEnvs-sft](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-sft); the sandbox
design follows Hugging Face FineEnvs' environment protocol. See [`NOTICE`](NOTICE).
