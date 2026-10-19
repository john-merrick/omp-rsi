"""Install the OMP router for every default-profile session; run once per machine."""

import argparse
import os
from pathlib import Path


def install(agent_dir, target):
    destination = Path(agent_dir).expanduser() / 'extensions' / 'omp-rsi-router.js'
    target = Path(target).resolve()
    if destination.is_symlink() and destination.resolve() == target:
        return destination
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'{destination} already exists; refusing to replace it')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(target)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-dir', type=Path, default=Path(os.environ.get(
        'PI_CODING_AGENT_DIR', '~/.omp/agent')).expanduser(),
        help='OMP agent directory for the target profile')
    args = parser.parse_args()
    print(install(args.agent_dir, Path(__file__).with_name('router.js')))
    print('Routing runs on every OMP session; bounded evaluation runs automatically at most daily.')


if __name__ == '__main__':
    main()
