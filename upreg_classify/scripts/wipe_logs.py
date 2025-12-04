import argparse
import os
import sys
import shutil
import time
from pathlib import Path
from typing import Iterable, List, Tuple


def resolve_repo_root() -> Path:
	"""Resolve repo root assuming this file is at upreg_classify/scripts/.

	Falls back to CWD if the expected layout is not present.
	"""
	here = Path(__file__).resolve()
	# upreg_classify/scripts/wipe_logs.py -> repo root is parents[2]
	cand = here.parents[2]
	if (cand / 'upreg_classify').exists():
		return cand
	return Path.cwd()


def iter_items(path: Path) -> List[Path]:
	try:
		return list(path.iterdir()) if path.exists() else []
	except Exception:
		return []


def filter_older_than(items: Iterable[Path], older_than_days: float | None) -> List[Path]:
	if not older_than_days or older_than_days <= 0:
		return list(items)
	cutoff = time.time() - older_than_days * 86400.0
	out = []
	for p in items:
		try:
			mtime = p.stat().st_mtime
			if mtime < cutoff:
				out.append(p)
		except Exception:
			# If stat fails, include to be safe (delete)
			out.append(p)
	return out


def apply_keep_latest(items: List[Path], keep_latest: int | None) -> List[Path]:
	if not keep_latest or keep_latest <= 0:
		return items
	# Sort by mtime desc (newest first)
	items_sorted = sorted(items, key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
	return items_sorted[keep_latest:]


def delete_path(p: Path, dry_run: bool) -> Tuple[Path, bool, str | None]:
	try:
		if dry_run:
			return p, True, None
		if p.is_dir():
			shutil.rmtree(p)
		else:
			p.unlink(missing_ok=True)
		return p, True, None
	except Exception as e:
		return p, False, str(e)


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description='Wipe time-curve logs and runs safely.')
	p.add_argument('--logs', action='store_true', help='Wipe time_curve_logs (stdout/stderr logs).')
	p.add_argument('--runs', action='store_true', help='Wipe time_curve_runs (per-run model folders and results).')
	p.add_argument('--older-than-days', type=float, default=None, help='Only delete items older than N days.')
	p.add_argument('--keep-latest', type=int, default=0, help='Keep the N most recent items in each folder, delete the rest.')
	p.add_argument('--keep-results', action='store_true', help='When wiping runs, keep results_*.csv files.')
	p.add_argument('--dry-run', action='store_true', help='Show what would be deleted, but do not delete.')
	p.add_argument('-y', '--yes', action='store_true', help='Do not prompt for confirmation.')
	p.add_argument('--repo-root', type=str, default=None, help='Override repo root path; default is auto-resolved.')
	return p


def main():
	args = build_parser().parse_args()
	repo = Path(args.repo_root) if args.repo_root else resolve_repo_root()
	logs_dir = repo / 'upreg_classify' / 'models' / 'time_curve_logs'
	runs_dir = repo / 'upreg_classify' / 'models' / 'time_curve_runs'

	# If neither flag is set, do both
	do_logs = args.logs or (not args.logs and not args.runs)
	do_runs = args.runs or (not args.logs and not args.runs)

	to_delete: list[Path] = []

	if do_logs:
		items = iter_items(logs_dir)
		items = filter_older_than(items, args.older_than_days)
		items = apply_keep_latest(items, args.keep_latest)
		to_delete.extend(items)

	if do_runs:
		items = iter_items(runs_dir)
		# If keeping results_*.csv, filter them out
		if args.keep_results:
			items = [p for p in items if not (p.is_file() and p.name.startswith('results_') and p.suffix.lower() == '.csv')]
		items = filter_older_than(items, args.older_than_days)
		items = apply_keep_latest(items, args.keep_latest)
		to_delete.extend(items)

	# De-duplicate while preserving order
	seen = set()
	unique_delete = []
	for p in to_delete:
		s = str(p.resolve())
		if s not in seen:
			unique_delete.append(p)
			seen.add(s)

	# Print plan
	print('Repo root:', repo)
	if do_logs:
		print('Logs dir :', logs_dir)
	if do_runs:
		print('Runs dir :', runs_dir)
	print('\nPlanned deletions ({}):'.format(len(unique_delete)))
	for p in unique_delete:
		kind = 'DIR ' if p.is_dir() else 'FILE'
		try:
			mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime)) if p.exists() else 'N/A'
		except Exception:
			mtime = 'N/A'
		print(f'  [{kind}] {p} (mtime={mtime})')

	if not unique_delete:
		print('\nNothing to delete. Exiting.')
		return 0

	# Confirm
	if not args.yes:
		ans = input('\nProceed with deletion? [y/N]: ').strip().lower()
		if ans not in ('y', 'yes'):
			print('Aborted by user.')
			return 0

	# Delete
	failures = []
	for p in unique_delete:
		_p, ok, err = delete_path(p, args.dry_run)
		if not ok:
			failures.append((_p, err))

	if args.dry_run:
		print('\nDry-run complete. No files were deleted.')
		return 0

	if failures:
		print('\nCompleted with errors:')
		for p, err in failures:
			print('  ', p, '->', err)
		return 1

	print('\nDeletion complete.')
	return 0


if __name__ == '__main__':
	sys.exit(main())

