"""Multi-turn SFT warm start for SmolDataEnvs (agent format).

Trains on the 4,677 recorded bash-tool trajectories (SmolDataEnvs-sft): system + user
-> assistant tool calls -> tool responses -> ... -> answer.txt. Loss is on assistant
turns only; tool output and user messages are context.

Hyperparameters come from the `sft.agent_format` block of train_plan.yaml.

    python train_sft_local.py --dry-run                          # no GPU: data + length checks
    python train_sft_local.py --model models/Qwen3-0.6B
    python train_sft_local.py --max-steps 1 --no-eval            # smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

# SwanLab's console output contains emoji; the Windows GBK console raises on them and
# would turn a finished run into a traceback at swanlab.finish().
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent

from train_common import setup_swanlab  # noqa: E402


def _plain(obj):
    """The SFT parquet stores tool schemas with numpy arrays (e.g. parameters.required);
    Jinja's tojson cannot serialise those, so convert to plain Python first."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return [_plain(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_rows() -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(ROOT / "data/SmolDataEnvs-sft/data/train-00000-of-00001.parquet")
    return [{"messages": [json.loads(m) for m in r.messages], "tools": _plain(list(r.tools))}
            for r in df.itertuples()]


def drop_consecutive_assistant(rows: list[dict]) -> list[dict]:
    """Drop rows where an assistant turn is followed by another assistant turn.

    Those are recording artifacts of the source agents: the model says something in one
    message and calls the tool in the next. The eval harness ends an episode as soon as a
    reply has no tool_calls (eval_harness.py: `if not resp["tool_calls"]: break`), so
    training on them would teach the model to stop mid-episode without submitting.
    """
    kept = []
    for r in rows:
        roles = [m["role"] for m in r["messages"]]
        if not any(a == b == "assistant" for a, b in zip(roles, roles[1:])):
            kept.append(r)
    return kept


def drop_over_length(rows: list[dict], model: str, max_length: int) -> list[dict]:
    """Keep complete trajectories only: TRL truncates on the right, which would cut the
    final tool call that writes answer.txt and teach the model to end without submitting.

    The length must be measured exactly the way SFTTrainer(assistant_only_loss=True)
    tokenizes: training chat template (with {% generation %} markers, which add a few
    tokens) + return_assistant_tokens_mask. Measuring with the plain template undercounts
    by up to ~20 tokens and lets borderline rows through to be truncated."""
    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template, has_generation_markers

    mp = Path(model)
    if not mp.exists() and (ROOT / model).exists():
        mp = ROOT / model
    tok = AutoTokenizer.from_pretrained(mp)
    template = None if has_generation_markers(tok.chat_template) else get_training_chat_template(tok)
    kept = []
    for i, r in enumerate(rows, 1):
        out = tok.apply_chat_template(r["messages"], tools=r["tools"], tokenize=True, return_dict=True,
                                      add_generation_prompt=False, return_assistant_tokens_mask=True,
                                      chat_template=template,
                                      chat_template_kwargs=r.get("chat_template_kwargs") or {})
        if len(out["input_ids"]) <= max_length:
            kept.append(r)
        if i % 1000 == 0:
            print(f"  length check {i}/{len(rows)} ...", flush=True)
    return kept


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=("agent",), default="agent", help="kept for command compatibility")
    ap.add_argument("--model", default="models/Qwen3-0.6B")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1, help="override epochs (smoke runs)")
    ap.add_argument("--max-length", type=int, default=None, help="override the config's max sequence length")
    ap.add_argument("--no-eval", action="store_true", help="skip the holdout evaluation (saves memory/time)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--swanlab", action="store_true", help="log training curves to SwanLab")
    ap.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"),
                    default=os.environ.get("SWANLAB_MODE", "cloud"))
    ap.add_argument("--swanlab-project", default=os.environ.get("SWANLAB_PROJECT", "SmolData-GRPO"))
    ap.add_argument("--swanlab-name", default=None, help="experiment name (default: output dir name)")
    a = ap.parse_args()

    plan = yaml.safe_load((ROOT / "train_plan.yaml").read_text())
    cfg = plan["sft"]["agent_format"]
    rows = load_rows()
    # trl 1.14's SFTConfig has no chat_template_kwargs field on Windows; per-example kwargs
    # are read by the tokenizer (Qwen3: thinking off, same as GRPO and the eval harness).
    for r in rows:
        r["chat_template_kwargs"] = {"enable_thinking": False}

    cap = a.max_length or cfg["max_length"]
    n0 = len(rows)
    rows = drop_consecutive_assistant(rows)
    print(f"agent: dropped {n0 - len(rows)}/{n0} trajectories with consecutive assistant turns")
    n1 = len(rows)
    rows = drop_over_length(rows, a.model, cap)
    print(f"agent: dropped {n1 - len(rows)}/{n1} trajectories longer than max_length={cap}; {len(rows)} kept")

    chars = sorted(sum(len(json.dumps(m)) for m in r["messages"]) for r in rows)
    print(f"agent: {len(rows)} examples; chars p50={chars[len(chars)//2]} p99={chars[int(len(chars)*0.99)]} "
          f"(~{chars[int(len(chars)*0.99)]//3.5:.0f} tokens p99 vs max_length {cap})")
    if a.dry_run:
        print("dry run ok")
        return

    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    if a.no_eval:  # no holdout: train on every row (the eval dataset would just waste 5%)
        train_ds, eval_ds = Dataset.from_list(rows), None
    else:
        ds = Dataset.from_list(rows).train_test_split(test_size=cfg.get("eval_holdout", 0.05), seed=42)
        train_ds, eval_ds = ds["train"], ds["test"]
    out = a.out or str(ROOT / "runs" / f"sft-agent-{a.model.replace(chr(92), '/').split('/')[-1]}")
    # A fresh suffix per run: reruns reuse the same output dir, so the default experiment
    # name would otherwise collide in SwanLab and resume the previous run.
    run_name = f"{a.swanlab_name or Path(out).name}-{secrets.token_hex(2)}"
    report_to = setup_swanlab(a.swanlab, a.swanlab_mode, a.swanlab_project, run_name)
    if a.swanlab and a.swanlab_mode != "disabled":
        print(f"swanlab experiment: {run_name}")
    L = cfg["lora"]
    trainer = SFTTrainer(
        model=a.model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=LoraConfig(r=L["r"], lora_alpha=L["alpha"], lora_dropout=L["dropout"],
                               target_modules=L["target_modules"], exclude_modules=L["exclude_modules"]),
        args=SFTConfig(
            output_dir=out,
            max_steps=a.max_steps,
            num_train_epochs=cfg["epochs"],
            max_length=cap,
            per_device_train_batch_size=cfg["per_device_batch"],
            per_device_eval_batch_size=1,   # agent transcripts are long; keep eval memory flat
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["learning_rate"],
            gradient_checkpointing=True,
            bf16=True,
            warmup_steps=20,
            logging_steps=10,
            eval_strategy="no" if a.no_eval else "steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            # Loss on assistant turns only (tool responses and user messages are context).
            # TRL swaps in a chat template with {% generation %} markers when the model's
            # own template lacks them.
            assistant_only_loss=True,
            report_to=report_to,
            run_name=run_name,
        ),
    )
    trainer.train()
    trainer.save_model(out + "/final")
    print(f"saved -> {out}/final  (evaluate with eval_harness.py --hf-adapter; merge before GRPO init)")


if __name__ == "__main__":
    main()
