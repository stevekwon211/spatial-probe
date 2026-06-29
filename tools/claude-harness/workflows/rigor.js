export const meta = {
  name: 'rigor',
  description: 'One method spine (research OR build): ground-truth scan -> seal falsifiable claim -> execute -> adversarial self-refute -> honest report',
  phases: [
    { title: 'Scan', detail: 'ground-truth content scan of the task area (CONFIRMED/INFERENCE/UNVERIFIED)' },
    { title: 'Seal', detail: 'research: commit a pre-registration; build: spec + failing test' },
    { title: 'Execute', detail: 'do the work, one variable at a time, standard-over-homegrown' },
    { title: 'Verify', detail: 'independent skeptics try to refute; CI not point-estimate' },
    { title: 'Report', detail: 'honest synthesis — negatives as headlines, relative gaps, reproducible' },
  ],
}

// args = { mode: 'research'|'build', task: string, config?: object }
const a = (typeof args === 'object' && args) ? args : {}
const MODE = a.mode === 'build' ? 'build' : 'research'
const TASK = a.task || 'UNSPECIFIED TASK — state it in args.task'
const CONFIG = a.config ? JSON.stringify(a.config) : '(none)'
const AT = { agentType: 'rigorist' }

const SPINE = `Follow the rigor spine. Label every factual claim CONFIRMED (read this run) / INFERENCE /
UNVERIFIED. Use tested libraries for metrics/stats (never hand-roll without unit-testing vs the standard
on tie/edge cases). Suspect the apparatus before the hypothesis. Mode=${MODE}. Config=${CONFIG}.`

const SCAN = { type: 'object', additionalProperties: false, required: ['inventory', 'risks', 'unknowns'], properties: {
  inventory: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['fact', 'evidence', 'label'], properties: {
    fact: { type: 'string' }, evidence: { type: 'string', description: 'file:line / command / value read this run' },
    label: { type: 'string', enum: ['CONFIRMED', 'INFERENCE', 'UNVERIFIED'] } } } },
  risks: { type: 'array', items: { type: 'string' } },
  unknowns: { type: 'array', items: { type: 'string', description: 'load-bearing premise not yet verified' } },
} }

const VERDICT = { type: 'object', additionalProperties: false, required: ['refuted', 'reason', 'severity'], properties: {
  refuted: { type: 'boolean', description: 'true if this skeptic believes the claim/that-it-works is NOT established' },
  reason: { type: 'string' }, severity: { type: 'string', enum: ['critical', 'major', 'minor', 'note'] },
  primary_source_checked: { type: 'string', description: 'what primary source (code/data/behavior) you verified by content' } } }

phase('Scan')
const scan = await agent(
  `${SPINE}\n\nGROUND-TRUTH SCAN for: ${TASK}\nScan the real artifacts (ls/find/wc -l/git log/read data/logs).
Return the verified inventory, the load-bearing risks, and the unknowns. Do NOT trust docs/summaries/memory.`,
  { ...AT, phase: 'Scan', schema: SCAN })

phase('Seal')
const sealPrompt = MODE === 'research'
  ? `${SPINE}\n\nSEAL (research) for: ${TASK}\nUsing the scan:\n${JSON.stringify(scan)}\nAuthor a falsifiable
pre-registration.md (hypothesis + success + KILL criteria + independence ledger + the single discriminating
observation) and COMMIT it to git BEFORE any data is examined. Return the committed path + commit hash + the
KILL criteria verbatim. If you cannot commit, say so loudly — an unsealed run is not valid.`
  : `${SPINE}\n\nSEAL (build) for: ${TASK}\nUsing the scan:\n${JSON.stringify(scan)}\nWrite the spec /
acceptance criteria and a FAILING test that encodes them, BEFORE implementation. Return the spec, the test
path, and proof the test currently fails.`
const seal = await agent(sealPrompt, { ...AT, phase: 'Seal' })

phase('Execute')
const exec = await agent(
  `${SPINE}\n\nEXECUTE for: ${TASK}\nScan:\n${JSON.stringify(scan)}\nSealed plan:\n${seal}\nDo the work per the
sealed plan ONLY. Change one variable at a time. ${MODE === 'build' ? 'Make the failing test pass; do not edit the test to fit the code.' : 'Do not pick the analysis using the outcome.'}
Return exactly what was produced + the exact reproduction command (+ seed).`,
  { ...AT, phase: 'Execute' })

phase('Verify')
const lenses = MODE === 'research'
  ? ['statistical sufficiency + CI of the decision statistic (not a point estimate)',
     'apparatus / wiring / homegrown-metric bug (suspect the instrument)',
     'independence + circularity of any oracle/ground-truth',
     'does the claim survive on held-out / re-run, and is it the RELATIVE gap not an absolute cutoff']
  : ['does it actually work end-to-end (run it, not just compile/green)',
     'edge cases / failure paths / the failing test really now passes for the right reason',
     'scope creep + did it touch only what the spec allowed',
     'security / auth / input-validation / silent fallback']
const votes = (await parallel(lenses.map((lens, i) => () =>
  agent(`${SPINE}\n\nADVERSARIAL VERIFY (lens ${i + 1}: ${lens}) of:\nTASK: ${TASK}\nRESULT:\n${exec}\nTry to
REFUTE that the claim/that-it-works is established, via THIS lens. Verify against the PRIMARY source by
content. Default to refuted=true if you cannot confirm.`,
    { ...AT, phase: 'Verify', schema: VERDICT })))).filter(Boolean)
const refuted = votes.filter(v => v.refuted).length
const survives = refuted < Math.ceil(lenses.length / 2)

phase('Report')
const report = await agent(
  `${SPINE}\n\nHONEST REPORT for: ${TASK}\nScan:\n${JSON.stringify(scan)}\nSeal:\n${seal}\nResult:\n${exec}
Adversarial verdicts (${refuted}/${lenses.length} refuted; survives=${survives}):\n${JSON.stringify(votes, null, 2)}
Write a blunt report: what is CONFIRMED vs INDETERMINATE, negatives as headlines, the relative gap vs baseline
(not an absolute), the exact reproduction, and the unresolved unknowns. Do NOT overclaim — if it did not
survive adversarial verify, say the claim is not established and what would settle it.`,
  { ...AT, phase: 'Report' })

return { mode: MODE, task: TASK, scan, seal, exec, votes, refuted, survives, report }
