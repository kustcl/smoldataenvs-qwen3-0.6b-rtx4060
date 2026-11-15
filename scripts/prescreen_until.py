"""Screen train tasks until TARGET "mixed" tasks are found, then write a GRPO pool.

A task is mixed when its strict reward (`reward_v2`) is neither all-0 nor all-1 across
`--n-samples` rollouts: 0 < sum < N. All-0 and all-1 groups have zero GRPO advantage, so
they are thrown away. Sampling uses the same agent loop / sandbox / settings as the GRPO
rollout (temperature 0.8, 6 tool calls + final turn), so the pool matches training.

Batches are screened one after another and the script stops as soon as TARGET mixed tasks
exist. Rerunning the same command resumes: `eval_harness.py --resume` (default) skips
(task, sample) pairs already written to the shared output directory.

    # 50 mixed tasks from train easy+medium, rollouts from the SFT checkpoint
    python scripts/prescreen_until.py --target 50 --tiers easy,medium

Outputs (kept out of runs/, which holds eval results and model weights):
    grpo_train_pool/                    rollouts + transcripts (single shared run dir)
    grpo_train_pool/batches/            task-id lists, one per batch (reproducible)
    grpo_train_pool/report.json         screening stats
    grpo_train_pool/grpo_pool_qwen3_06b_agent.json  the pool (task ids, exactly TARGET)
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PARQUET = ROOT / "data/SmolDataEnvs/data/train-00000-of-00001.parquet"


def load_listing():
    listing = json.loads((ROOT / "data/bucket_listing.json").read_text(encoding="utf-8"))
    sizes: dict[str, dict[str, int]] = {}
    whole: dict[str, int] = {}
    for key, files in listing.items():
        _bucket, prefix = key.split("::", 1)
        sizes.setdefault(prefix, {})
        for f in files or []:
            sizes[prefix][f["path"].split("/")[-1]] = f["size"]
            whole[prefix] = whole.get(prefix, 0) + f["size"]
    return sizes, whole


def candidates(tiers: set[str], max_input_mb: float) -> tuple[list[str], dict[str, str]]:
    """Train task ids whose inputs are complete on disk (no ungraded samples) and small."""
    sizes, whole = load_listing()
    dest = ROOT / "data/task_files"
    df = pd.read_parquet(TRAIN_PARQUET)
    df = df[df.difficulty_tier.isin(tiers)]
    ids, tier_of, skipped_files, skipped_size = [], {}, 0, 0
    for r in df.itertuples():
        names = list(r.files)
        if names:
            need = 0
            ok = True
            for f in names:
                p = dest / r.bucket_prefix / f
                exp = sizes.get(r.bucket_prefix, {}).get(f)
                if not p.is_file() or p.stat().st_size == 0 or (exp is not None and p.stat().st_size != exp):
                    ok = False
                    break
                need += exp or p.stat().st_size
            if not ok:
                skipped_files += 1
                continue
        else:
            pdir = dest / r.bucket_prefix
            if not pdir.is_dir() or not any(p.is_file() for p in pdir.iterdir()):
                skipped_files += 1
                continue
            need = whole.get(r.bucket_prefix, 0)
        if max_input_mb and need > max_input_mb * 1e6:
            skipped_size += 1
            continue
        ids.append(r.task_id)
        tier_of[r.task_id] = r.difficulty_tier
    print(f"candidates: {len(ids)} tasks ({dict(collections.Counter(tier_of.values()))}); "
          f"skipped {skipped_files} with incomplete inputs, {skipped_size} over {max_input_mb:g} MB")
    return ids, tier_of


def read_progress(out_dir: Path, reward: str):
    """task_id -> {sample: reward} from the shared results.jsonl (resume-aware)."""
    by: dict[str, dict[int, float | None]] = collections.defaultdict(dict)
    order: list[str] = []
    p = out_dir / "results.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tid = r["task_id"]
            if tid not in by:
                order.append(tid)
            by[tid][r.get("sample", 0)] = r.get(reward)
    return by, order


def summarise(by, n_samples: int) -> tuple[dict[str, list], set[str], set[str], set[str]]:
    full = {t: v for t, v in by.items() if len(v) >= n_samples and all(x is not None for x in v.values())}
    sums = {t: sum(v.values()) for t, v in full.items()}
    mixed = {t for t, s in sums.items() if 0 < s < n_samples}
    never = {t for t, s in sums.items() if s == 0}
    always = {t for t, s in sums.items() if s == n_samples}
    return full, mixed, never, always


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=50, help="mixed tasks to collect")
    ap.add_argument("--tiers", default="easy,medium")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=25, help="candidates per eval run (stop granularity)")
    ap.add_argument("--reward", default="reward_v2", help="reward that defines mixed (GRPO optimises this)")
    ap.add_argument("--hf-model", default="models/Qwen3-0.6B")
    ap.add_argument("--hf-adapter", default="runs/sft-0.6b-2048-e1/final")
    ap.add_argument("--sandbox", default="win-docker")
    ap.add_argument("--concurrency", type=int, default=8, help="tasks in flight inside one eval run")
    ap.add_argument("--hf-batch-size", type=int, default=4,
                    help="coalesced generate() batch for the local HF client (1 = serial)")
    ap.add_argument("--hf-batch-window-ms", type=float, default=20.0)
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop after this many hours of screening (0 = no limit); the pool is still written")
    ap.add_argument("--max-turns", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    # eval-easy per-call prompts: p50 1155, p95 3743, max 7165. 4096 truncates only 4.2%
    # of calls (1/16 tasks); 3072 would truncate 18.3%. Left truncation drops system+tools.
    ap.add_argument("--hf-max-prompt-tokens", type=int, default=4096)
    ap.add_argument("--max-input-mb", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="grpo_train_pool")
    ap.add_argument("--pool-out", default="grpo_train_pool/grpo_pool_qwen3_06b_agent.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tiers = {t.strip() for t in a.tiers.split(",") if t.strip()}
    ids, tier_of = candidates(tiers, a.max_input_mb)
    random.Random(a.seed).shuffle(ids)

    out_dir = ROOT / a.out
    batches_dir = out_dir / "batches"
    out_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(exist_ok=True)

    by, _ = read_progress(out_dir, a.reward)
    full, mixed, never, always = summarise(by, a.n_samples)
    print(f"resume: {len(full)} tasks already screened, {len(mixed)} mixed so far")

    # a task is done once it has n_samples graded rewards; partial ones are re-run (resume fills them)
    remaining = [t for t in ids if t not in full]
    selected = [t for t in ids if t in mixed][: a.target]
    batch_i = len(list(batches_dir.glob("batch-*.json")))
    started = time.time()
    full0 = len(full)

    while len(selected) < a.target and remaining:
        elapsed_h = (time.time() - started) / 3600
        if a.max_hours and elapsed_h >= a.max_hours:
            print(f"time budget {a.max_hours:g} h reached after {elapsed_h:.2f} h; stopping")
            break
        batch = remaining[: a.batch_size]
        remaining = remaining[a.batch_size:]
        batch_i += 1
        bfile = batches_dir / f"batch-{batch_i:03d}.json"
        bfile.write_text(json.dumps(batch))
        cmd = [sys.executable, str(ROOT / "eval_harness.py"), "run",
               "--hf-model", a.hf_model, "--split", "train", "--task-ids", str(bfile),
               "--n-samples", str(a.n_samples), "--temperature", str(a.temperature),
               "--top-p", str(a.top_p), "--max-turns", str(a.max_turns),
               "--max-tokens", str(a.max_tokens),
               "--hf-max-prompt-tokens", str(a.hf_max_prompt_tokens),
               "--hf-batch-size", str(a.hf_batch_size),
               "--hf-batch-window-ms", str(a.hf_batch_window_ms),
               "--repetition-penalty", str(a.repetition_penalty),
               "--concurrency", str(a.concurrency), "--mode", "agent",
               "--sandbox", a.sandbox, "--require-data", "--out", str(out_dir)]
        if a.hf_adapter:
            cmd[cmd.index("--split"):cmd.index("--split")] = ["--hf-adapter", a.hf_adapter]
        print(f"\n=== batch {batch_i}: {len(batch)} candidates -> {bfile} ===")
        print(" ".join(cmd), flush=True)
        if a.dry_run:
            print("dry run: stopping before the first eval")
            return
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            print(f"eval_harness failed with exit {rc}; rerun the same command to resume")
            break

        by, _ = read_progress(out_dir, a.reward)
        full, mixed, never, always = summarise(by, a.n_samples)
        for t in batch:  # deterministic pool order: screening order
            if t in mixed and t not in selected:
                selected.append(t)
        elapsed = time.time() - started
        done_now = max(len(full) - full0, 1)          # tasks finished in THIS invocation (resume-safe)
        per_task_h = (elapsed / 3600) / done_now
        rate = len(mixed) / max(len(full), 1)
        need = max(a.target - len(selected), 0)
        eta = need / rate * per_task_h if rate else float("inf")
        print(f"\nscreened {len(full)} tasks: {len(mixed)} mixed, {len(never)} never, {len(always)} always "
              f"(mixed rate {rate:.0%}) -> pool {min(len(selected), a.target)}/{a.target}")
        print(f"elapsed {elapsed/3600:.2f} h ({per_task_h*3600:.0f} s/task this run), ETA {eta:.1f} h", flush=True)

    pool = selected[: a.target]
    Path(ROOT / a.pool_out).write_text(json.dumps(pool))
    report = {
        "target": a.target, "tiers": sorted(tiers), "n_samples": a.n_samples, "reward": a.reward,
        "candidates": len(ids), "screened": len(full), "mixed": len(mixed),
        "never_solved": len(never), "always_solved": len(always),
        "pool": pool, "pool_tiers": dict(collections.Counter(tier_of.get(t, "?") for t in pool)),
        "batches": [p.name for p in sorted(batches_dir.glob("batch-*.json"))],
        "elapsed_h": round((time.time() - started) / 3600, 2),
        "settings": {k: getattr(a, k) for k in ("hf_model", "hf_adapter", "max_turns", "max_tokens",
                                                "temperature", "concurrency", "hf_batch_size", "sandbox")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\npool: {len(pool)} mixed tasks -> {a.pool_out} {report['pool_tiers']}")
    if len(pool) < a.target:
        print(f"WARNING: only {len(pool)} mixed tasks found; candidates exhausted. Add more candidates/"
              f"tiers or screen another seed.")
    print(json.dumps({k: v for k, v in report.items() if k != "pool"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
