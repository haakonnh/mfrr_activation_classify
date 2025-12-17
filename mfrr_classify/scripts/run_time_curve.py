import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run time-curve experiments by invoking mfrr_classify/src/train/train.py "
            "with varying --time_limit values and a fixed random seed (passed via --extra-args)."
        )
    )
    p.add_argument(
        "--budgets",
        type=str,
        default="30,60,120,200",
        help="Comma-separated list of time limits (seconds), e.g. '30,60,120,200'",
    )
    p.add_argument(
        "--model-preset",
        type=str,
        default="rf_xt_priority",
        help="Value for --model_preset passed to train.py (default: rf_xt_priority)",
    )
    p.add_argument(
        "--tune-up-bias",
        action="store_true",
        help="If set, includes --tune_up_bias when calling train.py",
    )
    p.add_argument(
        "--single-persistence",
        action="store_true",
        help="If set, includes --single_persistence when calling train.py",
    )
    p.add_argument(
        "--extra-args",
        type=str,
        default="",
        help=(
            "Extra args appended as-is to train.py command. "
            "Note: train.py currently has no seed flag; leave empty unless you add one (e.g., --seed 42)."
        ),
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use when invoking train.py (default: current interpreter)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    p.add_argument(
        "--logs-dir",
        type=str,
        default=str(Path("mfrr_classify") / "models" / "time_curve_logs"),
        help="Directory to write per-budget logs (stdout/stderr).",
    )
    p.add_argument(
        "--runs-dir",
        type=str,
        default=str(Path("mfrr_classify") / "models" / "time_curve_runs"),
        help="Base directory where per-budget output_dir folders will be created.",
    )
    return p


def main():
    args = build_parser().parse_args()

    # Resolve paths
    repo_root = Path(__file__).resolve().parents[2]
    train_py = repo_root / "mfrr_classify" / "src" / "train" / "train.py"
    if not train_py.exists():
        print(f"ERROR: Could not find train.py at: {train_py}")
        sys.exit(1)

    # Prepare budgets
    try:
        budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    except ValueError:
        print("ERROR: --budgets must be comma-separated integers, e.g., '30,60,120,200'")
        sys.exit(2)

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("Repo root:", repo_root)
    print("Train script:", train_py)
    print("Budgets:", budgets)
    print("Logs dir:", logs_dir)

    results = []

    aggregate_rows = []

    for sec in budgets:
        # Build command
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"t{sec}s_{timestamp}"
        run_output_dir = runs_dir / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            str(train_py),
            "--task", "multiclass",
            "--time_limit", str(sec),
            "--model_preset", args.model_preset,
            "--output_dir", str(run_output_dir),
        ]
        if args.tune_up_bias:
            cmd.append("--tune_up_bias")
        if args.single_persistence:
            cmd.append("--single_persistence")
        if args.extra_args:
            cmd.extend(shlex.split(args.extra_args))

        # Logging
        log_base = logs_dir / run_name
        out_log = log_base.with_suffix(".out.log")
        err_log = log_base.with_suffix(".err.log")

        print("\n=== Running:")
        print(" ", " ".join(shlex.quote(x) for x in cmd))
        print("  stdout:", out_log)
        print("  stderr:", err_log)

        if args.dry_run:
            results.append({"time_limit_s": sec, "returncode": None, "stdout": str(out_log), "stderr": str(err_log)})
            continue

        with open(out_log, "w", encoding="utf-8") as f_out, open(err_log, "w", encoding="utf-8") as f_err:
            proc = subprocess.run(cmd, cwd=str(repo_root), stdout=f_out, stderr=f_err)

        result = {
            "time_limit_s": sec,
            "returncode": proc.returncode,
            "stdout": str(out_log),
            "stderr": str(err_log),
            "output_dir": str(run_output_dir),
            "run_name": run_name,
        }
        results.append(result)

        # Try to parse metrics for aggregation
        metrics = None
        metrics_csv = run_output_dir / "metrics.csv"
        metrics_json = run_output_dir / "metrics.json"
        try:
            if metrics_csv.exists():
                # Expect a single-row CSV with header
                import pandas as pd  # local import to avoid hard dependency if not installed
                dfm = pd.read_csv(metrics_csv)
                if len(dfm) > 0:
                    metrics = dfm.iloc[0].to_dict()
            elif metrics_json.exists():
                with open(metrics_json, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            else:
                # Fallback: parse from stdout log (looks for a JSON blob after 'Metrics:')
                with open(out_log, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("Metrics:"):
                            try:
                                json_str = line.split("Metrics:", 1)[1]
                                metrics = json.loads(json_str)
                            except Exception:
                                pass
        except Exception:
            pass

        row = {"time_limit_s": sec, "run_name": run_name, "output_dir": str(run_output_dir), "returncode": proc.returncode}
        if isinstance(metrics, dict):
            # Normalize common keys if present
            for k in [
                "val_f1_macro", "val_accuracy", "test_f1_macro", "test_accuracy",
                "val_f1", "test_f1"
            ]:
                if k in metrics:
                    row[k] = metrics[k]
        aggregate_rows.append(row)

        if proc.returncode != 0:
            print(f"WARNING: Command for {sec}s exited with code {proc.returncode}. See logs above.")

    # Print summary
    print("\nSummary:")
    for r in sorted(results, key=lambda x: x["time_limit_s"]):
        print(f"  {r['time_limit_s']:>4d}s -> returncode={r['returncode']} | out={r['stdout']} | err={r['stderr']}")

    # Write aggregate CSV
    if aggregate_rows:
        agg_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        agg_csv = runs_dir / f"results_{agg_ts}.csv"
        # Determine fieldnames
        fieldnames = sorted({k for row in aggregate_rows for k in row.keys()})
        with open(agg_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(aggregate_rows, key=lambda x: x["time_limit_s"]):
                writer.writerow(row)
        print(f"\nAggregated metrics written to: {agg_csv}")


if __name__ == "__main__":
    main()
