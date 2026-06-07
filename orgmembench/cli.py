"""OrgMemBench CLI.

Dry-run by default — `run` only spends tokens / hits systems with `--execute`
(which also requires the relevant API keys + services to be up). Examples:

    orgmembench stats --tier small
    orgmembench list
    orgmembench run --system reference --tier small          # free, dry-run
    orgmembench run --system mem0 --tier small --execute     # real (needs keys)
    orgmembench leaderboard
    orgmembench smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="orgmembench", description="OrgMemBench harness.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("stats", help="Corpus + question stats for a tier.")
    ps.add_argument("--tier", required=True); ps.add_argument("--company", default="helix")

    sub.add_parser("list", help="List available adapters.")

    pr = sub.add_parser("run", help="Run a system on a tier.")
    pr.add_argument("--system", required=True); pr.add_argument("--tier", required=True)
    pr.add_argument("--company", default="helix"); pr.add_argument("--limit", type=int, default=None)
    pr.add_argument("--parallel", type=int, default=1,
                    help="Number of questions to process concurrently after ingest. Default: 1.")
    pr.add_argument("--execute", action="store_true",
                    help="Actually run (token spend; needs keys/services). Default: dry-run.")

    pj = sub.add_parser("judge-submission", help="Judge an external predictions file.")
    pj.add_argument("--file", required=True); pj.add_argument("--system", required=True)
    pj.add_argument("--tier", required=True); pj.add_argument("--company", default="helix")

    sub.add_parser("leaderboard", help="Render results/ into leaderboard.md.")
    sub.add_parser("usage", help="Measured token/cost usage vs estimate, across completed runs.")
    sub.add_parser("smoke", help="Run the free smoke tests.")

    pc = sub.add_parser("cost", help="Estimate token/USD cost per system/tier (pre-run).")
    pc.add_argument("--tier", default=None, help="One tier; omit with --all for all tiers.")
    pc.add_argument("--all", action="store_true", help="All tiers.")
    pc.add_argument("--company", default="helix")

    pp = sub.add_parser("plan", help="Pick every (system, tier) under a cost budget; emit commands + prereqs.")
    pp.add_argument("--budget", type=float, required=True, help="Max all-in USD per (system, tier).")
    pp.add_argument("--company", default="helix")
    pp.add_argument("--write-script", default=None,
                    help="Also write a runnable bash script to this path.")

    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    if args.cmd == "stats":
        from .corpus import corpus_stats
        from .questions import questions_stats
        print(json.dumps({"corpus": corpus_stats(args.tier, args.company),
                          "questions": questions_stats(args.tier, args.company)}, indent=2))
        return 0

    if args.cmd == "list":
        from .adapters.registry import available, CONTESTANTS
        print("contestants:", ", ".join(CONTESTANTS))
        print("available adapters (importable now):", ", ".join(available()))
        return 0

    if args.cmd == "run":
        if args.execute:
            os.environ["ORGMEMBENCH_DRY_RUN"] = "0"
        from .runner import run_system
        run = run_system(args.system, args.tier, args.company, limit=args.limit, parallel=args.parallel)
        print(json.dumps(run.metrics, indent=2))
        return 0

    if args.cmd == "judge-submission":
        from .submission import judge_submission, validate_submission, load_submission
        print(json.dumps(validate_submission(load_submission(args.file), args.tier, args.company), indent=2))
        run = judge_submission(args.file, args.system, args.tier, args.company)
        print(json.dumps(run.metrics, indent=2))
        return 0

    if args.cmd == "leaderboard":
        from .leaderboard import write_leaderboard
        print(f"wrote {write_leaderboard()}")
        return 0

    if args.cmd == "usage":
        from .usage import usage_report
        print(usage_report())
        return 0

    if args.cmd == "smoke":
        import subprocess
        from pathlib import Path
        t = Path(__file__).resolve().parent.parent / "tests" / "test_smoke.py"
        return subprocess.call([sys.executable, str(t)])

    if args.cmd == "cost":
        from .cost import format_report
        tiers = ["small", "medium", "large"] if args.all or not args.tier else [args.tier]
        print(format_report(tiers, args.company))
        return 0

    if args.cmd == "plan":
        from .cost import format_plan, write_run_script
        print(format_plan(args.budget, args.company))
        if args.write_script:
            write_run_script(args.budget, args.write_script, args.company)
            print(f"\nwrote runnable script: {args.write_script}")
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
