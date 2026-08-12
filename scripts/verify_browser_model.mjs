/* Does the browser implementation agree with the Python one?
 *
 * The landing page claims to run the real model. That claim is only worth making if
 * it is checked, so this scores a grid of applicants in JavaScript and compares
 * against the values scripts/verify_browser_model.py wrote from the service's own
 * code path. Any disagreement beyond the tolerances below fails the build.
 *
 *   node scripts/verify_browser_model.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");

// Tolerances. The score must match to floating-point noise; SHAP values are summed
// over 157 trees and 300 background rows, so a slightly looser bound is honest
// rather than convenient.
const SCORE_TOLERANCE = 1e-9;
const SHAP_TOLERANCE = 1e-7;

const expected = JSON.parse(readFileSync(join(ROOT, "dist-verify", "expected.json"), "utf8"));

// The module fetches its own payload; give it one from disk.
const payload = JSON.parse(readFileSync(join(ROOT, "web", "assets", "model.json"), "utf8"));
globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => payload });

// A bare Windows path is not a valid ESM specifier; it needs a file:// URL.
const scorer = await import(pathToFileURL(join(ROOT, "web", "js", "scorer.js")).href);
await scorer.loadModel();

let worstScore = 0;
let worstShap = 0;
let worstBase = 0;
const failures = [];

const base = scorer.baseValue();
worstBase = Math.abs(base - expected.base_value);

const started = Date.now();
for (const c of expected.cases) {
    const x = scorer.buildVector(c.applicant);

    for (let i = 0; i < x.length; i += 1) {
        const diff = Math.abs(x[i] - c.vector[i]);
        if (diff > SCORE_TOLERANCE) {
            failures.push(`${c.name}: feature ${expected.features[i]} differs by ${diff.toExponential(2)}`);
        }
    }

    const raw = scorer.rawScore(x);
    worstScore = Math.max(worstScore, Math.abs(raw - c.raw_score));
    if (Math.abs(raw - c.raw_score) > SCORE_TOLERANCE) {
        failures.push(`${c.name}: raw score ${raw} vs ${c.raw_score}`);
    }

    const phi = scorer.shapValues(x);
    for (let i = 0; i < phi.length; i += 1) {
        const diff = Math.abs(phi[i] - c.shap[i]);
        worstShap = Math.max(worstShap, diff);
        if (diff > SHAP_TOLERANCE) {
            failures.push(
                `${c.name}: SHAP for ${expected.features[i]} ${phi[i].toFixed(9)} vs ${c.shap[i].toFixed(9)}`,
            );
        }
    }

    const decision = scorer.decide(scorer.sigmoid(raw));
    if (decision !== c.decision) {
        failures.push(`${c.name}: decision ${decision} vs ${c.decision}`);
    }

    const codes = scorer.reasonCodes(phi, decision).map((r) => r.code);
    if (JSON.stringify(codes) !== JSON.stringify(c.reason_codes)) {
        failures.push(`${c.name}: reason codes [${codes}] vs [${c.reason_codes}]`);
    }
}
const elapsed = Date.now() - started;

console.log(`  cases                 ${expected.cases.length}`);
console.log(`  worst score diff      ${worstScore.toExponential(2)}   (tolerance ${SCORE_TOLERANCE})`);
console.log(`  worst SHAP diff       ${worstShap.toExponential(2)}   (tolerance ${SHAP_TOLERANCE})`);
console.log(`  base value diff       ${worstBase.toExponential(2)}`);
console.log(`  time per case         ${(elapsed / expected.cases.length).toFixed(0)} ms  (score + full SHAP)`);

if (failures.length) {
    console.error(`\n  ${failures.length} disagreement(s):`);
    for (const f of failures.slice(0, 12)) console.error(`    ${f}`);
    process.exit(1);
}
console.log("\n  browser implementation agrees with the service.");
