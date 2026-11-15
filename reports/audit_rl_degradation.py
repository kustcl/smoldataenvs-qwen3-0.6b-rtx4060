"""Recompute the RL audit from saved results and completions; never loads model weights."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def exact_p(lost, gained):
    n = lost + gained
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(lost, gained) + 1)) / 2**n)


def main():
    output = {"scope": "Existing Windows Qwen3-0.6B runs; no model inference or retraining"}
    runs = {}
    evals = {}
    for name in ("eval-sft-b4-easy-medium", "eval-grpo150-easy-medium"):
        rows = read_jsonl(ROOT / "runs" / name / "results.jsonl")
        evals[name] = {r["task_id"]: r for r in rows}
        repeats = []
        errors = 0
        for r in rows:
            messages = read_json(ROOT / "runs" / name / "transcripts" / f"{r['task_id']}-s0.json")
            commands = []
            for m in messages:
                for tc in m.get("tool_calls", []):
                    args = tc["function"]["arguments"]
                    commands.append(json.loads(args)["command"] if isinstance(args, str) else args["command"])
                if m["role"] == "tool" and re.search(r"Traceback|SyntaxError|NameError|KeyError|TypeError|not found", m["content"]):
                    errors += 1
            repeats.append(len(commands) - len(set(commands)))
        submitted = sum(r["submitted"] for r in rows)
        correct = sum(r["reward_v2"] for r in rows)
        runs[name] = dict(n=len(rows), correct=correct, submitted=submitted,
                          strict_accuracy=correct / len(rows), correct_if_submitted=correct / submitted,
                          no_answer=len(rows) - submitted, hit_max_turns=sum(r["hit_max_turns"] for r in rows),
                          mean_calls=mean(r["n_tool_calls"] for r in rows),
                          mean_tokens=mean(r["completion_tokens"] for r in rows),
                          tasks_with_repeated_commands=sum(n > 0 for n in repeats),
                          duplicate_commands=sum(repeats), tool_error_messages_heuristic=errors,
                          config=read_json(ROOT / "runs" / name / "config.json"))
    output["evals"] = runs
    baseline = evals["eval-sft-b4-easy-medium"]
    output["paired"] = {}
    for name, rows in evals.items():
        if name == "eval-sft-b4-easy-medium":
            continue
        ids = sorted(set(baseline) & set(rows))
        lost = [t for t in ids if baseline[t]["reward_v2"] > rows[t]["reward_v2"]]
        gained = [t for t in ids if baseline[t]["reward_v2"] < rows[t]["reward_v2"]]
        output["paired"][name] = dict(common_tasks=len(ids), lost=lost, gained=gained,
                                    mcnemar_exact_p=exact_p(len(lost), len(gained)),
                                    lost_reasons=dict(Counter(rows[t]["reason_v2"] for t in lost)))
    logs = [r for r in read_jsonl(ROOT / "runs/grpo-agent-plus150/train_log.jsonl") if "reward" in r]
    output["training"] = {}
    keys = ["rewards/SmolDataEnv/mean", "rewards/reward_submitted/mean", "frac_reward_zero_std",
            "completions/clipped_ratio", "entropy", "tools/call_frequency", "completions/mean_length"]
    for lo, hi in ((1, 46), (47, 92), (93, 138), (1, 138)):
        seg = [r for r in logs if lo <= r["step"] <= hi]
        output["training"][f"{lo}-{hi}"] = {k: mean(r[k] for r in seg) for k in keys}
    output["training"]["submission_only_signal_steps"] = [r["step"] for r in logs
        if r["rewards/SmolDataEnv/std"] == 0 and r["reward_std"] > 0]
    output["training"]["zero_gradient_steps"] = sum(r["grad_norm"] == 0 for r in logs)
    output["training"]["nonzero_clip_ratio_steps"] = sum(r["clip_ratio/region_mean"] != 0 for r in logs)

    # pyarrow reads the exact logged trajectories without pandas or any GPU work.
    import pyarrow.parquet as pq
    paths = sorted((ROOT / "runs/grpo-agent-plus150/completions").glob("*.parquet"))
    first = pq.read_table(paths[0]).to_pylist()
    output["completion_schema"] = {k: type(v).__name__ for k, v in first[0].items()}
    output["completion_example"] = {k: str(v)[:200] for k, v in first[0].items()}
    output["completion_files"] = len(paths)
    completion_rows = [r for p in paths for r in pq.read_table(p).to_pylist()]
    output["completion_reward_audit"] = {
        "rollouts": len(completion_rows),
        "wrong_positive_advantage": sum(r["SmolDataEnv"] == 0 and r["advantage"] > 0 for r in completion_rows),
        "step9_rewards_and_advantages": [{k: r[k] for k in ("SmolDataEnv", "reward_submitted", "advantage")}
                                         for r in completion_rows if r["step"] == 9],
        "step9_positive_completion_excerpt": [r["completion"][:4000] for r in completion_rows
                                               if r["step"] == 9 and r["advantage"] > 0],
    }

    # Measure context budgets with the same local tokenizer/training template as eval.
    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template, has_generation_markers
    tok = AutoTokenizer.from_pretrained(ROOT / "runs/sft-0.6b-2048-e1/merged", local_files_only=True)
    template = None if has_generation_markers(tok.chat_template) else get_training_chat_template(tok)
    import sys
    sys.path.insert(0, str(ROOT))
    from smolenv_local import BASH_TOOL
    budgets = {}
    for name in ("eval-sft-b4-easy-medium", "eval-grpo150-easy-medium"):
        over_3072, over_8192, success_over = 0, 0, []
        success_calls = []
        for tid, rec in evals[name].items():
            msgs = read_json(ROOT / "runs" / name / "transcripts" / f"{tid}-s0.json")
            def encode(ms, generation):
                return tok.apply_chat_template(ms, tools=[BASH_TOOL], tokenize=True,
                    chat_template=template, return_dict=False,
                    add_generation_prompt=generation, enable_thinking=False)
            prefix_len = len(encode(msgs[:2], True))
            total = len(encode(msgs, False)) - prefix_len
            over_3072 += total > 3072
            for i, msg in enumerate(msgs):
                if msg["role"] == "assistant":
                    over_8192 += len(encode(msgs[:i], True)) > 8192
            if rec["reward_v2"]:
                success_calls.append(rec["n_tool_calls"])
                if total > 3072:
                    success_over.append(tid)
        budgets[name] = dict(transcripts_over_3072=over_3072,
                             eval_prompts_over_8192=over_8192,
                             successful_transcripts_over_3072=success_over,
                             successful_calls_over_6=sum(c > 6 for c in success_calls))
    output["token_budget_audit"] = budgets
    dest = ROOT / "reports/rl_degradation_evidence.json"
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
