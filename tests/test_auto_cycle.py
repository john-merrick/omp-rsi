import tempfile
import unittest
from datetime import date
from pathlib import Path

from auto_cycle import build_manifest, run_once
from rsi import score_trial


class AutomaticCycleTests(unittest.TestCase):
    def test_generated_tasks_have_real_failing_checks_and_fresh_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            first = build_manifest(Path(directory) / 'first', date(2026, 10, 19))
            second = build_manifest(Path(directory) / 'second', date(2026, 10, 20))
            self.assertEqual({t['split'] for t in first['tasks']}, {'train', 'holdout'})
            self.assertNotEqual(first['tasks'][1]['prompt'], second['tasks'][1]['prompt'])
            for task in first['tasks']:
                self.assertFalse(score_trial(Path(first['repo']), task['check'], 5)['passed'])
                self.assertIn('Explain', task['prompt'])

    def test_runs_at_most_once_daily_even_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def failing_runner(manifest, output, **options):
                calls.append((manifest, options))
                raise RuntimeError('provider unavailable')
            with self.assertRaisesRegex(RuntimeError, 'provider unavailable'):
                run_once(root, date(2026, 10, 19), runner=failing_runner)
            self.assertFalse(run_once(root, date(2026, 10, 19), runner=failing_runner))
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0][1]['auto'])
            self.assertLessEqual(calls[0][1]['max_cost'], .25)


if __name__ == '__main__':
    unittest.main()
