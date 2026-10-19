import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import route_eval


class LivePolicyTests(unittest.TestCase):
    def test_promoted_policy_changes_future_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'repo'
            repo.mkdir()
            (repo / 'answer.txt').write_text('old')
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(repo), '-c', 'user.name=T', '-c', 'user.email=t@example.com',
                            'commit', '-qm', 'initial'], check=True)
            policy_path = root / 'policy.json'
            policy_path.write_text(json.dumps({'version': 1, 'models': {'simple': ['luna'],
                'standard': ['sol'], 'complex': ['astra']},
                'thresholds': {'simpleMaxChars': 500, 'longPromptChars': 1200}}))
            tasks = [{'id': name, 'split': split, 'prompt': 'Explain and write ' + name +
                      ' to answer.txt. ' + 'context ' * 69,
                      'check': ['python3', '-c', f"assert open('answer.txt').read() == '{name}'"]}
                     for name, split in [('train', 'train'), ('hold', 'holdout')]]
            calls = []
            original = route_eval.execute

            def fake_execute(args, cwd, timeout, env=None):
                if args[0] != 'omp':
                    return original(args, cwd, timeout, env)
                config = json.loads(Path(env['OMP_RSI_POLICY']).read_text())
                tier = route_eval.classify_many([args[-1]], config)[0]
                model = config['models'][tier][0]
                calls.append((args[-1], model))
                if model == 'luna':
                    (Path(cwd) / 'answer.txt').write_text('hold' if 'hold' in args[-1] else 'train')
                message = {'role': 'assistant', 'model': model,
                           'usage': {'cost': {'total': .01 if model == 'luna' else .1}}}
                return 0, json.dumps({'type': 'agent_end', 'isTerminal': True, 'messages': [message]})

            with patch('route_eval.execute', side_effect=fake_execute):
                report = route_eval.run_eval({'repo': str(repo), 'tasks': tasks}, root / 'results',
                                             policy_path, repeats=2, max_seconds=5)
            self.assertTrue(report['promoted'])
            self.assertEqual(json.loads(policy_path.read_text())['thresholds']['simpleMaxChars'], 700)
            self.assertEqual(route_eval.classify_many([tasks[1]['prompt']], json.loads(policy_path.read_text())), ['simple'])
            self.assertEqual((repo / 'answer.txt').read_text(), 'old')
            self.assertTrue(all('hold' not in prompt for prompt, _ in calls[:2]))
            with self.assertRaisesRegex(ValueError, 'already evaluated'):
                route_eval.run_eval({'repo': str(repo), 'tasks': tasks}, root / 'again',
                                    policy_path, repeats=2, max_seconds=5)


if __name__ == '__main__':
    unittest.main()
