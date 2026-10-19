import { readFileSync, appendFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { homedir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const policyPath = () => process.env.OMP_RSI_POLICY || join(dirname(fileURLToPath(import.meta.url)), 'policy.json');
const feedbackPath = () => process.env.OMP_RSI_FEEDBACK || join(homedir(), '.omp', 'agent', 'rsi-feedback.jsonl');
const cyclePath = join(dirname(fileURLToPath(import.meta.url)), 'auto_cycle.py');

function scheduleCycle() {
  if (process.env.OMP_RSI_AUTOCYCLE === 'off' || process.env.OMP_RSI_ROUTING === 'off') return;
  // Finite background job, not an extra LLM call in the user's interactive turn.
  const child = spawn('python3', [cyclePath, '--policy', policyPath()], {
    detached: true, stdio: 'ignore', env: { ...process.env, OMP_RSI_AUTOCYCLE: 'off' },
  });
  child.on('error', () => {});
  child.unref();
}

export function classify(prompt, policy) {
  const text = prompt.toLowerCase();
  if (prompt.length >= policy.thresholds.longPromptChars ||
      /\b(security|vulnerabilit\w*|authenticat\w*|authoriz\w*|payment\w*|billing|migration\w*|production|incident|architecture|data loss|race condition|multi[- ]?repo)\b/i.test(text)) {
    return 'complex';
  }
  if (/^\s*(explain|summarize|summarise|list|describe|what is|where is|translate|proofread|fix typo)\b/i.test(text)
      && prompt.length < (policy.thresholds.simpleMaxChars ?? 500)
      && !/\b(implement|refactor|deploy|delete|rewrite)\b/i.test(text)) {
    return 'simple';
  }
  return 'standard';
}

export function chooseModel(tier, policy, resolve) {
  for (const selector of policy.models[tier]) {
    const model = resolve(selector);
    if (model) return model;
  }
  return undefined;
}

// Share precisely the live classifier with the offline evaluator.
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1] && process.argv[2] === '--classify') {
  const input = JSON.parse(readFileSync(0, 'utf8'));
  process.stdout.write(JSON.stringify(input.prompts.map(prompt => classify(prompt, input.policy))) + '\n');
}

export default function (pi) {
  let lastRoute;
  pi.on('session_start', (_event, ctx) => {
    if (ctx.agent?.kind === 'sub') return;
    scheduleCycle();
    ctx.setInterval?.(scheduleCycle, 60 * 60 * 1000);
  });
  pi.on('before_agent_start', async (event, ctx) => {
    if (ctx.agent?.kind === 'sub' || process.env.OMP_RSI_ROUTING === 'off') return;
    let policy;
    try {
      policy = JSON.parse(readFileSync(policyPath(), 'utf8'));
    } catch (error) {
      ctx.ui.notify(`RSI policy unavailable: ${error.message}`, 'warning');
      return;
    }
    const tier = classify(event.prompt, policy);
    const model = chooseModel(tier, policy, name => ctx.models.resolve(name));
    if (!model) return;
    const previous = ctx.models.current();
    if (previous?.provider !== model.provider || previous?.id !== model.id) {
      if (!await pi.setModel(model)) return;
    }
    lastRoute = { tier, model: `${model.provider}/${model.id}`, policyVersion: policy.version };
  });

  pi.on('turn_end', (event, ctx) => {
    if (ctx.agent?.kind === 'sub' || !lastRoute || process.env.OMP_RSI_ROUTING === 'off') return;
    const usage = event.message?.usage;
    const record = {
      at: new Date().toISOString(), ...lastRoute,
      costUsd: usage?.cost?.total ?? null,
      inputTokens: usage?.input ?? null,
      outputTokens: usage?.output ?? null,
      stopReason: event.message?.stopReason ?? null,
    };
    try {
      appendFileSync(feedbackPath(), JSON.stringify(record) + '\n');
    } catch { /* Telemetry must never disrupt an agent turn. */ }
  });
}
