import test from 'node:test';
import assert from 'node:assert/strict';
import router, { classify, chooseModel } from '../router.js';

const policy = {
  models: {
    simple: ['openai-codex/gpt-6-luna'],
    standard: ['openai-codex/gpt-6-sol'],
    complex: ['openai-codex/gpt-6-astra', 'openai-codex/gpt-6-sol'],
  },
  thresholds: { longPromptChars: 1200, simpleMaxChars: 500 },
};

test('routes clear low-risk work cheaply and uncertain coding to standard', () => {
  assert.equal(classify('Explain what this function does', policy), 'simple');
  assert.equal(classify('Fix the auth integration', policy), 'standard');
  assert.equal(classify('Refactor payment authorization and audit security', policy), 'complex');
  assert.equal(classify('x'.repeat(1200), policy), 'complex');
});

test('promoted threshold changes routing without changing the classifier implementation', () => {
  const prompt = 'Explain this change: ' + 'context '.repeat(72);
  assert.equal(classify(prompt, policy), 'standard');
  assert.equal(classify(prompt, { ...policy, thresholds: { ...policy.thresholds, simpleMaxChars: 700 } }), 'simple');
});

test('selects an available model, falling back without inventing availability', () => {
  const resolve = name => name.endsWith('gpt-6-sol') ? { provider: 'openai-codex', id: 'gpt-6-sol' } : undefined;
  assert.equal(chooseModel('complex', policy, resolve)?.id, 'gpt-6-sol');
  assert.equal(chooseModel('simple', policy, resolve), undefined);
});

test('starts one managed hourly trigger and skips subagent sessions', () => {
  const hooks = new Map();
  router({ on: (name, handler) => hooks.set(name, handler) });
  const timers = [];
  const prior = process.env.OMP_RSI_AUTOCYCLE;
  process.env.OMP_RSI_AUTOCYCLE = 'off';
  try {
    hooks.get('session_start')({}, { agent: { kind: 'main' }, setInterval: (...args) => timers.push(args) });
    hooks.get('session_start')({}, { agent: { kind: 'sub' }, setInterval: (...args) => timers.push(args) });
    assert.equal(timers.length, 1);
    assert.equal(timers[0][1], 60 * 60 * 1000);
  } finally {
    if (prior === undefined) delete process.env.OMP_RSI_AUTOCYCLE;
    else process.env.OMP_RSI_AUTOCYCLE = prior;
  }
});
