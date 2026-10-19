import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi import run_experiment, score_trial, validate_manifest


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        (self.repo / 'answer.txt').write_text('old')
        subprocess.run(['git', '-C', str(self.repo), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Test', '-c', 'user.email=t@example.com', 'commit', '-qm', 'initial'], check=True)
        self.tasks = [
            {'id': 'train', 'split': 'train', 'prompt': 'Change answer.txt to train', 'check': ['python3', '-c', "assert open('answer.txt').read() == 'train'"]},
            {'id': 'hold', 'split': 'holdout', 'prompt': 'Change answer.txt to hold', 'check': ['python3', '-c', "assert open('answer.txt').read() == 'hold'"]},
        ]

    def test_rejects_unsafe_manifest(self):
        with self.assertRaises(ValueError):
            validate_manifest({'repo': str(self.repo), 'tasks': [{**self.tasks[0], 'check': 'rm -rf /'}]})
        with self.assertRaises(ValueError):
            validate_manifest({'repo': str(self.repo), 'tasks': [self.tasks[0]]})

    def test_promotes_only_with_holdout_gain_and_no_regression(self):
        manifest = {'repo': str(self.repo), 'tasks': self.tasks, 'baseline_guidance': 'Work carefully'}
        prompts = []
        def fake_omp(prompt, cwd, output, timeout, model):
            prompts.append(prompt)
            if 'Propose ONE' in prompt:
                return 'Read task carefully and write the requested value.'
            value = 'hold' if 'hold' in prompt else 'train'
            if 'Read task carefully' in prompt:
                (cwd / 'answer.txt').write_text(value)
            return 'done'
        with patch('rsi.call_omp', side_effect=fake_omp):
            report = run_experiment(manifest, self.root / 'results', max_seconds=5,
                                    library=self.root / 'lessons.txt')
        self.assertTrue(report['promoted'])
        self.assertEqual((self.root / 'results' / 'promoted.txt').read_text(), report['candidate'])
        self.assertEqual((self.root / 'lessons.txt').read_text(), report['candidate'] + '\n')
        self.assertTrue(any('Work carefully\nRead task carefully' in p for p in prompts))
        self.assertEqual((self.repo / 'answer.txt').read_text(), 'old')
        self.assertEqual(len(report['trials']), 4)
        self.assertNotEqual(report['trials'][0]['workspace'], report['trials'][1]['workspace'])

    def test_holdout_regression_blocks_promotion(self):
        manifest = {'repo': str(self.repo), 'tasks': self.tasks}
        def fake_omp(prompt, cwd, output, timeout, model):
            if 'Propose ONE' in prompt:
                return 'Candidate guidance.'
            if 'hold' in prompt and 'Candidate guidance:' in prompt:
                (cwd / 'answer.txt').write_text('wrong')
            else:
                (cwd / 'answer.txt').write_text('hold' if 'hold' in prompt else 'train')
            return 'done'
        with patch('rsi.call_omp', side_effect=fake_omp):
            report = run_experiment(manifest, self.root / 'results', max_seconds=5)
        self.assertFalse(report['promoted'])
        self.assertFalse((self.root / 'results' / 'promoted.txt').exists())

    def test_previous_lesson_reaches_next_baseline_without_holdout_leak(self):
        library = self.root / 'lessons.txt'
        library.write_text('Use previous evidence.\n')
        prompts = []
        def fake_omp(prompt, cwd, output, timeout, model):
            prompts.append(prompt)
            if prompt.startswith('Propose ONE'):
                return 'Another lesson.'
            return 'done'
        with patch('rsi.call_omp', side_effect=fake_omp):
            report = run_experiment({'repo': str(self.repo), 'tasks': self.tasks},
                                    self.root / 'results', max_seconds=5, library=library)
        self.assertIn('Use previous evidence.', prompts[0])
        self.assertNotIn('Change answer.txt to hold', next(p for p in prompts if p.startswith('Propose ONE')))
        self.assertFalse(report['promoted'])
        self.assertEqual(library.read_text(), 'Use previous evidence.\n')

    def test_check_failure_is_not_a_success(self):
        self.assertFalse(score_trial(self.repo, ['python3', '-c', 'raise SystemExit(1)'], 5)['passed'])


if __name__ == '__main__':
    unittest.main()
