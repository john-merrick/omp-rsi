"""Run a fresh, bounded OMP routing evaluation once per day without user-supplied tasks."""

import argparse
import fcntl
import json
import os
import random
import subprocess
import sys
from datetime import date
from pathlib import Path

from route_eval import DEFAULT_POLICY, run_eval


def build_manifest(directory, day):
    directory.mkdir(parents=True, exist_ok=False)
    repo = directory / 'fixture'
    repo.mkdir()
    (repo / 'answer.txt').write_text('pending')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=OMP RSI',
                    '-c', 'user.email=rsi@localhost', 'commit', '-qm', 'fixture'], check=True)
    rng = random.Random(day.toordinal())
    numbers = iter(rng.sample(range(11, 50), 4))
    tasks = []
    for name, split in (('train', 'train'), ('holdout', 'holdout')):
        a, b = next(numbers), next(numbers)
        answer = a * a + b
        prompt = (f'Explain briefly how to compute {a} squared plus {b}, then write only the '
                  f'numeric result to answer.txt with no other file changes or trailing newline. '
                  + 'Context: the calculation uses ordinary integer arithmetic. ' * 8)
        tasks.append({'id': name, 'split': split, 'prompt': prompt,
                      'check': ['python3', '-c', f"assert open('answer.txt').read() == '{answer}'"]})
    return {'repo': str(repo), 'tasks': tasks}


def run_once(state_dir, day=None, runner=run_eval, policy_path=DEFAULT_POLICY):
    day = day or date.today()
    state_dir = Path(state_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / 'cycle.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        stamp = state_dir / 'last-attempt.txt'
        if stamp.exists() and stamp.read_text().strip() == day.isoformat():
            return False
        # Record even failures: a broken provider never creates an unbounded retry loop.
        stamp.write_text(day.isoformat() + '\n')
        work = state_dir / day.isoformat()
        manifest = build_manifest(work, day)
        try:
            report = runner(manifest, work / 'results', policy_path=policy_path,
                            repeats=2, max_seconds=60, auto=True, max_cost=.25)
            (work / 'outcome.json').write_text(json.dumps({'promoted': report['promoted'],
                                                           'winner': report['winner']}) + '\n')
        except Exception as error:
            (work / 'error.txt').write_text(f'{type(error).__name__}: {error}\n')
            raise
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, default=Path(os.environ.get(
        'OMP_RSI_AUTO_DIR', '~/.omp/agent/rsi-auto')).expanduser())
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    try:
        print('evaluated' if run_once(args.state_dir, policy_path=args.policy) else 'already attempted today')
    except Exception as error:
        print(f'RSI evaluation stopped: {error}', file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
