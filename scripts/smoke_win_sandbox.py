"""Smoke test for the Windows agent sandbox (Docker Desktop Linux containers).

Runs a synthetic task through the same code path the agent loop uses:

  bash tool -> sandbox stdout/stderr -> /workdir/answer.txt -> v1/v2 reward

and checks the isolation properties: no network, read-only rootfs, timeout kill.

    python scripts/smoke_win_sandbox.py
    python scripts/smoke_win_sandbox.py --backend win-docker
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import smolenv_local as env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    a = ap.parse_args()

    backend = env.detect_backend(a.backend)
    print(f"backend: {backend}")
    if backend == "none":
        raise SystemExit("no sandbox backend available (is Docker Desktop running, and the image built?)")

    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="smolenv-smoke-", ignore_cleanup_errors=True) as td:
        td = Path(td)
        prefix = td / "data" / "bkt" / "prefix01"
        prefix.mkdir(parents=True)
        (prefix / "sales.csv").write_text("region,revenue\nWest,10\nEast,5\nWest,7.5\n")
        row = {
            "task_id": "smoke1", "answer": "17.5", "reward_mode": "numeric",
            "atol": 1e-6, "rtol": 1e-6,
            "bucket_prefix": "bkt/prefix01", "files": ["sales.csv"],
            "instruction": "What is the total revenue for the West region?",
        }
        cfg = env.SandboxConfig(backend=backend, timeout_s=60, mem_mb=4096, cpus=2)
        sb = env.LocalSandbox(row, td / "data", td / "runs", cfg)
        if sb.missing_files:
            raise SystemExit(f"input staging failed: {sb.missing_files}")

        # 1. bash tool: pandas inside the container reads the read-only mounted CSV
        res = sb.run_bash(
            "python3 -c \"import pandas as pd; "
            "df = pd.read_csv('/home/user/input/sales.csv'); "
            "print(round(df[df.region=='West'].revenue.sum(), 2))\"")
        checks["bash_pandas"] = res.returncode == 0 and "17.5" in res.stdout
        print(f"--- pandas   rc={res.returncode} out={res.stdout.strip()!r} err={res.stderr.strip()[:200]!r}")

        # 2. submit through the bash tool, then read the file on the host side
        res = sb.run_bash("echo -n 17.5 > /workdir/answer.txt")
        checks["answer_txt"] = res.returncode == 0 and (sb.answer_txt() or "").strip() == "17.5"
        print(f"--- submit   rc={res.returncode} answer={sb.answer_txt()!r}")

        # 3. the reward path: answer.txt -> v1 + v2
        graded = env.agent_final(sb, row)
        checks["agent_reward"] = graded["reward_v1"] == 1.0 and graded["reward_v2"] == 1.0
        print(f"--- reward   {graded}")

        # 4. no network (container has --network none; the guard also refuses sockets)
        res = sb.run_bash(
            "python3 -c \"import socket\n"
            "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=3); print('OPEN')\n"
            "except Exception as e:\n    print('BLOCKED:', type(e).__name__)\"")
        checks["network_blocked"] = "BLOCKED" in res.stdout
        print(f"--- network  {res.stdout.strip()[:200]!r}")

        # 5. read-only rootfs: writing outside /workdir and /tmp must fail
        res = sb.run_bash("touch /etc/evil 2>&1; echo rc=$?")
        checks["rootfs_readonly"] = "rc=1" in res.stdout
        print(f"--- rootfs   {res.stdout.strip()[:200]!r}")

        # 6. timeout: sleep past a 3 s limit; the container must die and the call return
        t0 = time.time()
        cfg_t = env.SandboxConfig(backend=backend, timeout_s=3, mem_mb=2048, cpus=2)
        sb_t = env.LocalSandbox(row, td / "data", td / "runs_timeout", cfg_t)
        res = sb_t.run_bash("sleep 100")
        checks["timeout_kills"] = res.timed_out and time.time() - t0 < 30
        print(f"--- timeout  timed_out={res.timed_out} in {time.time() - t0:.1f}s")

    print()
    for k, v in checks.items():
        print(("PASS  " if v else "FAIL  ") + k)
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
