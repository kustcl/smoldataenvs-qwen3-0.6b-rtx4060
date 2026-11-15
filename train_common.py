"""Shared training plumbing: profile loading, row sampling, run metadata.

Split out of the former train_grpo_local.py so the agent-only project keeps the
helpers the agent trainer and the pre-screen scripts need.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_profile(name: str) -> dict:
    import yaml

    plan = yaml.safe_load((ROOT / "train_plan.yaml").read_text())
    prof = dict(plan["grpo"]["common"])
    prof.update(plan["grpo"]["profiles"][name])
    prof["_plan"] = plan
    return prof


def build_train_rows(prof: dict, n_tasks: int | None = None) -> list[dict]:
    from eval_harness import load_tasks

    rows = load_tasks("train", tiers=prof.get("difficulty", ""), limit=0)
    pool = prof.get("task_pool_file")
    if pool:  # pre-screened: tasks the starting model solves only sometimes
        path = ROOT / pool
        if not path.exists():  # never silently fall back to the whole train split
            raise SystemExit(f"task_pool_file not found: {pool} (pass --set task_pool_file=... or create it)")
        keep = set(json.loads(path.read_text()))
        rows = [r for r in rows if r["task_id"] in keep]
    if prof.get("max_input_mb"):
        listing = json.loads((ROOT / "data/bucket_listing.json").read_text())
        size = {(k.split("::")[1], f["path"].split("/")[-1]): f["size"] for k, v in listing.items() for f in (v or [])}
        whole = {k.split("::")[1]: sum(f["size"] for f in (v or [])) for k, v in listing.items()}

        def need(r):  # empty `files` -> the whole prefix is staged
            return sum(size.get((r["bucket_prefix"], f), 0) for f in r["files"]) if r["files"] else whole.get(r["bucket_prefix"], 0)

        rows = [r for r in rows if need(r) <= prof["max_input_mb"] * 1e6]
    random.Random(prof.get("seed", 42)).shuffle(rows)
    return rows[: (n_tasks or prof["num_tasks"])]


def _swanlab_has_key() -> bool:
    """True if SwanLab has credentials. swanlab < 0.10 ships `swanlab.package.has_api_key`;
    0.10+ stores the key in ~/.swanlab/.netrc instead (the module was removed)."""
    try:
        from swanlab.package import has_api_key  # swanlab < 0.10
        return has_api_key()
    except Exception:
        pass
    if os.environ.get("SWANLAB_API_KEY"):
        return True
    nrc = Path.home() / ".swanlab" / ".netrc"
    try:
        return nrc.is_file() and "password" in nrc.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def setup_swanlab(enabled: bool, mode: str, project: str, experiment: str | None = None) -> str:
    """Configure SwanLab for Trainer(report_to=...) and return the report target.

    cloud needs `swanlab login` (or SWANLAB_API_KEY); local writes to ./swanlog
    without a login. Raises before training starts if cloud is requested but the
    user is not logged in, instead of hanging on an interactive login prompt.
    """
    if not enabled or mode == "disabled":
        return "none"
    try:
        import swanlab

        # transformers 5.x's SwanLab callback calls swanlab.get_run() at on_train_begin and
        # expects None before init; swanlab >= 0.6 raises "No active Run" instead, which
        # aborts the whole training run. Make it return None, as the integration expects.
        if not getattr(swanlab, "_smolenv_get_run_patched", False):
            _orig_get_run = swanlab.get_run

            def _get_run_or_none():
                try:
                    return _orig_get_run()
                except RuntimeError:
                    return None

            swanlab.get_run = _get_run_or_none
            swanlab._smolenv_get_run_patched = True
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"SwanLab is not available ({e}); pip install swanlab or drop --swanlab")
    if mode == "cloud" and not _swanlab_has_key():
        raise SystemExit("SwanLab cloud mode requested but not logged in: run `swanlab login` "
                         "(or set SWANLAB_API_KEY), or use --swanlab-mode local")
    os.environ["SWANLAB_MODE"] = mode
    # swanlab >= 0.7 moved the project name to SWANLAB_PROJ_NAME; a plain SWANLAB_PROJECT
    # value is parsed as the nested `project` settings object and raises SettingsError.
    try:
        import swanlab as _sl
        _ver = tuple(int(x) for x in _sl.__version__.split(".")[:2])
    except Exception:
        _ver = (0, 0)
    if _ver >= (0, 7):
        os.environ.pop("SWANLAB_PROJECT", None)
        os.environ["SWANLAB_PROJ_NAME"] = project
    else:
        os.environ["SWANLAB_PROJECT"] = project
    if experiment:
        os.environ["SWANLAB_EXP_NAME"] = experiment
    if mode == "local":
        logdir = ROOT / "swanlog"
        logdir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("SWANLAB_LOG_DIR", str(logdir))
    return "swanlab"


def write_meta(run_dir: Path, prof: dict, n_tasks: int, mode: str, init_from=None) -> None:
    """run_dir/meta.json: what the live dashboard needs to show progress and an ETA."""
    keep = ("model", "lora", "optimise", "max_steps", "num_generations", "per_device_train_batch_size",
            "gradient_accumulation_steps", "learning_rate", "temperature", "max_completion_length", "eval_every",
            "save_steps", "task_pool_file", "max_tool_calling_iterations")
    meta = {k: prof.get(k) for k in keep if k in prof}
    meta.update(mode=mode, n_tasks=n_tasks, init_from=init_from, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
