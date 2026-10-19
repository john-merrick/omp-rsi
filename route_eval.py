"""Compare OMP router policies on isolated train/holdout tasks and promote evidence-backed changes."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from rsi import execute, score_trial, validate_manifest

ROUTER = Path(__file__).with_name('router.js').resolve()
DEFAULT_POLICY = ROUTER.with_name('policy.json')


def candidate_policies(policy, auto=False):
    results = []
    for key, delta, minimum, maximum in (
        ('simpleMaxChars', -200, 150, 900), ('simpleMaxChars', 200, 150, 900),
        *(() if auto else (('longPromptChars', -400, 600, 3000), ('longPromptChars', 400, 600, 3000))),
    ):
        candidate = copy.deepcopy(policy)
        candidate['version'] = policy['version'] + 1
        candidate['thresholds'][key] = max(minimum, min(maximum, policy['thresholds'][key] + delta))
        if candidate not in results and candidate['thresholds'] != policy['thresholds']:
            results.append(candidate)
    if not auto:
        for tier, alternative in (('simple', 'standard'), ('standard', 'simple'), ('complex', 'standard')):
            candidate = copy.deepcopy(policy)
            candidate['version'] = policy['version'] + 1
            replacement = policy['models'][alternative][0]
            if replacement == policy['models'][tier][0]:
                continue
            candidate['models'][tier] = [replacement, *[m for m in candidate['models'][tier] if m != replacement]]
            if candidate not in results:
                results.append(candidate)
    return results


def compare(base, candidate):
    before = {t['task']: t for t in base}
    after = {t['task']: t for t in candidate}
    if before.keys() != after.keys():
        return {'acceptable': False, 'gain': 0, 'savings': None}
    if any(before[k]['passed'] and not after[k]['passed'] for k in before):
        return {'acceptable': False, 'gain': -1, 'savings': None}
    gain = sum(after[k]['passed'] for k in before) - sum(before[k]['passed'] for k in before)
    costs = all(before[k]['costUsd'] is not None and after[k]['costUsd'] is not None for k in before)
    base_cost = sum(before[k]['costUsd'] for k in before) if costs else None
    new_cost = sum(after[k]['costUsd'] for k in before) if costs else None
    savings = (base_cost - new_cost) / base_cost if base_cost and new_cost is not None else None
    # A cheaper failure is never an improvement, regardless of price.
    acceptable = gain > 0 or (gain == 0 and all(before[k]['passed'] for k in before)
                              and savings is not None and savings >= .05)
    if gain > 0 and savings is not None and savings < -.5:
        acceptable = False
    return {'acceptable': acceptable, 'gain': gain, 'savings': savings}


def choose_candidate(trials):
    base = trials['base']
    eligible = [(key, compare(base, records)) for key, records in trials.items() if key != 'base']
    eligible = [(key, result) for key, result in eligible if result['acceptable']]
    if not eligible:
        return None
    return max(eligible, key=lambda pair: (pair[1]['gain'], pair[1]['savings'] or 0))[0]


def classify_many(prompts, policy):
    run = subprocess.run(['node', str(ROUTER), '--classify'], input=json.dumps({'prompts': prompts,
                         'policy': policy}), text=True, capture_output=True, timeout=10, check=True)
    return json.loads(run.stdout)

def route_signatures(prompts, policy):
    return [(tier, policy['models'][tier][0]) for tier in classify_many(prompts, policy)]


def total_cost(log):
    try:
        end = next(json.loads(line) for line in reversed(log.splitlines())
                   if line.startswith('{\"type\"'))
        if not end.get('isTerminal'):
            return None, None
        messages = [m for m in end['messages'] if m.get('role') == 'assistant']
        if any(m.get('usage', {}).get('cost', {}).get('total') is None for m in messages):
            return None, None
        return sum(m['usage']['cost']['total'] for m in messages), messages[-1].get('model')
    except (StopIteration, KeyError, ValueError, IndexError, TypeError):
        return None, None


def run_eval(manifest, output, policy_path=DEFAULT_POLICY, repeats=2, max_seconds=180,
             auto=False, max_cost=None):
    if repeats < 2:
        raise ValueError('promotion requires at least two paired runs per task')
    repo = validate_manifest(manifest)
    output = Path(output).resolve()
    policy_path = Path(policy_path).resolve()
    if output == repo or repo in output.parents or output == policy_path or output in policy_path.parents:
        raise ValueError('output must be outside repository and policy')
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty')
    policy = json.loads(policy_path.read_text())
    if not isinstance(policy['version'], int) or policy['version'] < 1:
        raise ValueError('policy version must be positive')
    output.mkdir(parents=True, exist_ok=True)
    candidates = candidate_policies(policy, auto=auto)
    policies = {'base': policy, **{f'candidate{i}': value for i, value in enumerate(candidates)}}
    for label, value in policies.items():
        (output / f'{label}.json').write_text(json.dumps(value))
    tasks = manifest['tasks']
    train = [t for t in tasks if t['split'] == 'train']
    holdout = [t for t in tasks if t['split'] == 'holdout']
    holdout_id = hashlib.sha256(json.dumps({'repo': str(repo), 'tasks': holdout},
                                          sort_keys=True).encode()).hexdigest()
    ledger_path = policy_path.with_name(policy_path.name + '.evaluated.json')
    used = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    if holdout_id in used:
        raise ValueError('holdout tasks were already evaluated; supply fresh unseen tasks')
    records = {'train': {}, 'holdout': {}}
    spent = 0.0

    def run_task(task, label, repetition):
        nonlocal spent
        if max_cost is not None and spent >= max_cost:
            raise RuntimeError(f'evaluation cost ceiling reached: ${spent:.4f}')
        name = f"{task['id']}-{label}-{repetition}"
        workspace = output / name
        code, log = execute(['git', 'clone', '--quiet', '--no-hardlinks', str(repo), str(workspace)], output, 30)
        if code:
            raise RuntimeError(f'clone failed: {log}')
        env = dict(os.environ, OMP_RSI_POLICY=str(output / f'{label}.json'),
                   OMP_RSI_FEEDBACK=str(output / 'observations.jsonl'), OMP_RSI_AUTOCYCLE='off')
        argv = ['omp', '-p', '--mode=json', '--no-session', '--no-extensions', '-e', str(ROUTER),
                '--no-skills', '--no-rules', '--no-title', '--approval-mode=yolo',
                '--tools=read,edit,write,grep,glob,lsp', '--max-time', str(max_seconds),
                '--cwd', str(workspace), task['prompt']]
        try:
            code, log = execute(argv, workspace, max_seconds + 10, env=env)
            (output / f'{name}.jsonl').write_text(log)
            cost, model = total_cost(log)
            check = score_trial(workspace, task['check'], max_seconds) if code == 0 else {'passed': False, 'check_output': f'OMP exited {code}'}
        except TimeoutError as exc:
            cost, model, check = None, None, {'passed': False, 'check_output': str(exc)}
        if auto and cost is None:
            raise RuntimeError('missing OMP cost measurement; automatic promotion disabled')
        spent += cost or 0
        if max_cost is not None and spent > max_cost:
            raise RuntimeError(f'evaluation cost ceiling exceeded: ${spent:.4f}')
        return {'task': f"{task['id']}:{repetition}", 'passed': check['passed'],
                'costUsd': cost, 'model': model, 'check_output': check['check_output']}

    def evaluate(split, selected):
        for label in selected:
            records[split][label] = [run_task(task, label, repeat)
                                      for task in (train if split == 'train' else holdout)
                                      for repeat in range(repeats)]

    evaluate('train', ['base'])
    base_routes = route_signatures([t['prompt'] for t in train], policy)
    challengers = []
    seen = {tuple(base_routes)}
    for label, challenger in list(policies.items())[1:]:
        signature = tuple(route_signatures([t['prompt'] for t in train], challenger))
        if signature not in seen:
            challengers.append(label)
            seen.add(signature)
    evaluate('train', challengers)
    winner = choose_candidate(records['train'])
    promoted = False
    if winner:
        evaluate('holdout', ['base', winner])
        promoted = compare(records['holdout']['base'], records['holdout'][winner])['acceptable']
        # The policy must change the intended route and the model actually used.
        promoted = promoted and (route_signatures([t['prompt'] for t in holdout], policy) !=
                                 route_signatures([t['prompt'] for t in holdout], policies[winner]))
        promoted = promoted and any(base['model'] != candidate['model'] and
                                     base['model'] is not None and candidate['model'] is not None
                                     for base, candidate in zip(records['holdout']['base'],
                                                                records['holdout'][winner]))
    report = {'promoted': promoted, 'winner': winner, 'train': records['train'],
              'holdout': records['holdout'], 'policyVersion': policy['version'],
              'holdoutId': holdout_id if winner else None}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if winner:
        # Never reuse holdout evidence for later policy searches, including rejected proposals.
        with tempfile.NamedTemporaryFile('w', dir=ledger_path.parent, delete=False) as stream:
            stream.write(json.dumps([*used, holdout_id], indent=2) + '\n')
            temporary = stream.name
        os.replace(temporary, ledger_path)
    if promoted:
        # Install only after completed independent holdout; same-file atomic replacement.
        with tempfile.NamedTemporaryFile('w', dir=policy_path.parent, delete=False) as stream:
            stream.write(json.dumps(policies[winner], indent=2) + '\n')
            temporary = stream.name
        os.replace(temporary, policy_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Use the same manifest format as rsi.py: a clean Git repo and trusted train/holdout tasks '
        'with argv check commands. New labeled tasks can be evaluated repeatedly; no raw user prompts '
        'or unlabeled session telemetry are treated as correctness evidence.'))
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--max-seconds', type=int, default=180)
    args = parser.parse_args()
    report = run_eval(json.loads(args.manifest.read_text()), args.output, args.policy,
                      args.repeats, args.max_seconds)
    print(json.dumps({'promoted': report['promoted'], 'winner': report['winner'],
                      'report': str(args.output / 'report.json')}))


if __name__ == '__main__':
    main()
