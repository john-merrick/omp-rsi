import copy
import unittest

from route_eval import compare, candidate_policies, choose_candidate

POLICY = {'version': 1, 'models': {'simple': ['a'], 'standard': ['b'], 'complex': ['c']},
          'thresholds': {'longPromptChars': 1200, 'simpleMaxChars': 500}}


def record(task, variant, passed, cost):
    return {'task': task, 'variant': variant, 'passed': passed, 'costUsd': cost}


class RouteEvaluationTests(unittest.TestCase):
    def test_generates_bounded_distinct_challengers(self):
        candidates = candidate_policies(POLICY)
        self.assertEqual(len(candidates), 7)
        self.assertEqual(POLICY['thresholds']['simpleMaxChars'], 500)
        self.assertTrue(all(c['version'] == 2 for c in candidates))
        self.assertTrue(any(c['models']['standard'][0] == 'a' for c in candidates))
        self.assertTrue(any(c['models']['complex'][0] == 'b' for c in candidates))
        auto = candidate_policies(POLICY, auto=True)
        self.assertEqual(len(auto), 2)
        self.assertTrue(all(c['models'] == POLICY['models'] and
                            c['thresholds']['longPromptChars'] == 1200 for c in auto))

    def test_rejects_any_regression_even_when_cheaper(self):
        base = [record('a', 'base', True, 1), record('b', 'base', False, 1)]
        worse = [record('a', 'challenger', False, .01), record('b', 'challenger', True, .01)]
        self.assertFalse(compare(base, worse)['acceptable'])

    def test_requires_measured_gain_not_mock_or_missing_cost(self):
        base = [record('a', 'base', True, 1)]
        self.assertTrue(compare(base, [record('a', 'challenger', True, .8)])['acceptable'])
        self.assertFalse(compare(base, [record('a', 'challenger', True, None)])['acceptable'])
        self.assertFalse(compare(base, [record('a', 'challenger', True, .98)])['acceptable'])
        failing = [record('a', 'base', False, 1)]
        self.assertFalse(compare(failing, [record('a', 'challenger', False, .01)])['acceptable'])

    def test_train_selection_never_looks_at_holdout(self):
        trials = {
            'base': [record('train', 'base', True, 1)],
            'cheap': [record('train', 'cheap', True, .7)],
            'bad': [record('train', 'bad', False, .01)],
        }
        self.assertEqual(choose_candidate(trials), 'cheap')


if __name__ == '__main__':
    unittest.main()
