import { api } from "../api.js";
import { featureLabel, num, pct, sci, shortHash, usd } from "../format.js";
import { empty, html, mount, onAction, onSubmit, raw, skeleton } from "../render.js";
import { copyToClipboard, store, toast } from "../store.js";

const DTI_BANDS = [
    "<20%", "20%-<30%", "30%-<36%",
    "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
    "50%-60%", ">60%",
];

const PRESETS = {
    marginal: {
        label: "Marginal applicant",
        values: {
            income_usd: 96000, loan_amount_usd: 465000, property_value_usd: 495000,
            cltv: 94, dti_band: "47", loan_term: 360,
            tract_to_msa_income_pct: 96, units_band: "1",
        },
    },
    strong: {
        label: "Strong applicant",
        values: {
            income_usd: 185000, loan_amount_usd: 415000, property_value_usd: 620000,
            cltv: 67, dti_band: "30%-<36%", loan_term: 360,
            tract_to_msa_income_pct: 128, units_band: "1",
        },
    },
    stretched: {
        label: "Stretched applicant",
        values: {
            income_usd: 66000, loan_amount_usd: 345000, property_value_usd: 375000,
            cltv: 92, dti_band: ">60%", loan_term: 360,
            tract_to_msa_income_pct: 88, units_band: "1",
        },
    },
};

let cleanup = [];

export function mountPane(root) {
    mount(root, layout());
    const results = root.querySelector("#score-results");

    // The form is rendered once and never re-rendered: replacing its markup while a
    // field has focus would drop the caret mid-keystroke. Only the results update.
    applyPreset(root, "marginal");

    cleanup.push(
        onSubmit(root, async (form) => {
            if (form.id !== "applicant-form") return;
            mount(results, skeleton(4));
            try {
                const payload = readForm(form);
                const result = await api.score(payload);
                store.set({ lastDecision: result });
                mount(results, renderResult(result));
            } catch (err) {
                mount(results, renderError(err));
            }
        }),
    );

    cleanup.push(
        onAction(root, {
            preset: (el) => applyPreset(root, el.dataset.preset),
            copyId: (el) => copyToClipboard(el.dataset.value),
            gotoLedger: (el) => {
                store.set({ lookupId: el.dataset.value });
                location.hash = "#/ledger";
            },
            notice: (el) => window.open(api.adverseActionUrl(el.dataset.value), "_blank"),
        }),
    );

    mount(results, empty("Submit an application to see the decision, the reasons behind it, and what would have changed it."));
}

export function unmountPane() {
    cleanup.forEach((fn) => fn());
    cleanup = [];
}

function layout() {
    return html`
        <div class="page-head">
            <span class="label">Pane 1</span>
            <h1>Score an application</h1>
            <p>
                Every submission is scored, explained, and written to the append-only
                decision ledger in a single operation. There is no path through this
                service that produces a decision without recording it.
            </p>
        </div>

        <div class="grid-2">
            <div class="card">
                <div class="card-head"><h2>Application</h2></div>
                <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px">
                    ${raw(
                        Object.entries(PRESETS)
                            .map(
                                ([key, p]) =>
                                    `<button type="button" class="tiny ghost" data-action="preset" data-preset="${key}">${p.label}</button>`,
                            )
                            .join(""),
                    )}
                </div>

                <form id="applicant-form">
                    <fieldset class="fieldset">
                        <legend>Loan</legend>
                        <div class="field-row">
                            <div class="field">
                                <label for="loan_amount_usd">Loan amount <span class="hint">USD</span></label>
                                <!-- step is validated against min, so step=5000 with
                                     min=1000 would reject 345,000. Hence step=any. -->
                                <input type="number" id="loan_amount_usd" name="loan_amount_usd" min="1000" step="any" required>
                            </div>
                            <div class="field">
                                <label for="loan_term">Term <span class="hint">months</span></label>
                                <input type="number" id="loan_term" name="loan_term" min="60" max="480" step="any" required>
                            </div>
                        </div>
                        <div class="field">
                            <label for="cltv">Combined loan-to-value <span class="hint">percent</span></label>
                            <input type="number" id="cltv" name="cltv" min="0" max="200" step="any" required>
                        </div>
                    </fieldset>

                    <fieldset class="fieldset">
                        <legend>Property</legend>
                        <div class="field-row">
                            <div class="field">
                                <label for="property_value_usd">Property value <span class="hint">USD</span></label>
                                <input type="number" id="property_value_usd" name="property_value_usd" min="1000" step="any">
                            </div>
                            <div class="field">
                                <label for="units_band">Units</label>
                                <select id="units_band" name="units_band">
                                    ${raw(["1", "2", "3", "4"].map((u) => `<option value="${u}">${u}</option>`).join(""))}
                                </select>
                            </div>
                        </div>
                        <div class="field">
                            <label for="tract_to_msa_income_pct">Tract-to-area income <span class="hint">percent</span></label>
                            <input type="number" id="tract_to_msa_income_pct" name="tract_to_msa_income_pct" min="0" max="300" step="any">
                        </div>
                    </fieldset>

                    <fieldset class="fieldset">
                        <legend>Applicant</legend>
                        <div class="field">
                            <label for="income_usd">Annual income <span class="hint">USD</span></label>
                            <input type="number" id="income_usd" name="income_usd" min="1000" step="any" required>
                        </div>
                        <div class="field">
                            <label for="dti_band">Debt-to-income band</label>
                            <select id="dti_band" name="dti_band">
                                ${raw(DTI_BANDS.map((b) => `<option value="${b}">${b}</option>`).join(""))}
                            </select>
                        </div>
                        <label class="switch"><input type="checkbox" name="preapproval"> Preapproval requested</label>
                        <label class="switch"><input type="checkbox" name="interest_only"> Interest-only period</label>
                        <label class="switch"><input type="checkbox" name="balloon"> Balloon payment</label>
                        <label class="switch"><input type="checkbox" name="neg_amortization"> Negative amortization</label>
                    </fieldset>

                    <p class="tiny faint" style="margin-bottom:12px">
                        There is no field here for race, ethnicity, sex, or age. The
                        request schema has no place to put them.
                    </p>

                    <button type="submit" class="primary" style="width:100%;justify-content:center">
                        Score application
                    </button>
                </form>
            </div>

            <div id="score-results"></div>
        </div>
    `;
}

function applyPreset(root, key) {
    const preset = PRESETS[key];
    if (!preset) return;
    Object.entries(preset.values).forEach(([name, value]) => {
        const field = root.querySelector(`[name="${name}"]`);
        if (field) field.value = value;
    });
}

function readForm(form) {
    const data = new FormData(form);
    const payload = {};
    for (const [key, value] of data.entries()) {
        if (["dti_band", "units_band"].includes(key)) payload[key] = value;
        else if (["preapproval", "interest_only", "balloon", "neg_amortization"].includes(key))
            payload[key] = true;
        else if (value !== "") payload[key] = Number(value);
    }
    ["preapproval", "interest_only", "balloon", "neg_amortization"].forEach((k) => {
        if (!(k in payload)) payload[k] = false;
    });
    return payload;
}

function renderResult(r) {
    const approved = r.decision === "APPROVED";
    const cls = approved ? "approved" : "declined";
    const p = r.probability;
    const t = r.threshold;

    const worst = r.reason_codes.length
        ? Math.max(...r.reason_codes.map((c) => Math.abs(c.contribution)))
        : 1;

    return html`
        <div class="decision-card ${cls}">
            <div class="decision-head">
                <span class="dot ${approved ? "ok" : "bad"}"></span>
                <span class="decision-word">${r.decision}</span>
                <span class="pill ${approved ? "approve" : "decline"}" style="margin-left:auto">
                    model ${r.model.version}
                </span>
            </div>

            <div class="gauge">
                <div class="gauge-track">
                    <div class="gauge-fill" style="width:${(p * 100).toFixed(2)}%"></div>
                    <div class="gauge-threshold" style="left:${(t * 100).toFixed(2)}%"
                         data-label="threshold ${t}"></div>
                    <div class="gauge-value" style="left:${Math.min(94, Math.max(6, p * 100)).toFixed(2)}%">
                        ${num(p, 4)}
                    </div>
                </div>
                <div class="gauge-scale"><span>0.0</span><span>P(approve)</span><span>1.0</span></div>
            </div>

            ${raw(
                r.reason_codes.length
                    ? `<div class="label" style="margin-bottom:8px">Principal reasons for the decision</div>
                       <div class="reasons">${r.reason_codes
                           .map(
                               (c) => `
                        <div class="reason">
                            <span class="rank">${c.rank}</span>
                            <span class="text">${c.text}<span class="code">${c.code}</span></span>
                            <span class="contrib">
                                <span class="val">${c.contribution.toFixed(3)}</span>
                                <span class="bar" style="width:${Math.round((Math.abs(c.contribution) / worst) * 88)}%"></span>
                            </span>
                        </div>`,
                           )
                           .join("")}</div>`
                    : `<div class="callout"><span class="label">No adverse action reasons</span>
                       This application was approved. Regulation B attaches the duty to state
                       specific principal reasons to adverse action, so none are produced.</div>`,
            )}

            ${raw(
                r.counterfactual
                    ? `<div class="callout accent" style="margin-top:16px">
                           <span class="label">What would have changed the outcome</span>
                           <em>${r.counterfactual.sentence}</em>
                           ${
                               r.counterfactual.achievable
                                   ? `<div class="tiny faint" style="margin-top:8px">
                                        Verified by re-scoring: P(approve) becomes
                                        ${num(r.counterfactual.verified_probability, 4)} →
                                        ${r.counterfactual.verified_decision}.
                                        Method: ${r.counterfactual.method}.
                                      </div>`
                                   : ""
                           }
                       </div>`
                    : "",
            )}

            <div class="decision-id">
                <span class="label">decision_id</span>
                <span class="mono">${r.decision_id}</span>
                <button class="tiny ghost" data-action="copyId" data-value="${r.decision_id}">Copy</button>
                <button class="tiny ghost" data-action="gotoLedger" data-value="${r.decision_id}">
                    Verify &amp; replay →
                </button>
                ${raw(
                    approved
                        ? ""
                        : `<button class="tiny ghost" data-action="notice" data-value="${r.decision_id}">
                              Adverse action notice (PDF)
                           </button>`,
                )}
            </div>

            <div class="kv" style="margin-top:16px;padding-top:16px;border-top:1px solid var(--line)">
                <dt>Ledger position</dt><dd>#${r.ledger.seq}</dd>
                <dt>Row hash</dt><dd class="mono">${shortHash(r.ledger.row_hash, 12, 8)}</dd>
                <dt>Links to</dt><dd class="mono">${shortHash(r.ledger.prev_hash, 12, 8)}</dd>
                <dt>Model hash</dt><dd class="mono">${shortHash(r.model.hash, 12, 8)}</dd>
                <dt>Decision margin</dt><dd>${num(r.margin, 4)} log-odds from the cutoff</dd>
            </div>
        </div>

        <div class="card">
            <div class="card-head">
                <h3>Full attribution</h3>
                <span class="tiny faint">measured against the decision threshold, not the population mean</span>
            </div>
            <div class="barlist">
                ${raw(contributionBars(r.contributions))}
            </div>
        </div>
    `;
}

function contributionBars(contributions) {
    const entries = Object.entries(contributions).sort((a, b) => b[1] - a[1]);
    const max = Math.max(...entries.map(([, v]) => Math.abs(v)), 1e-9);
    return entries
        .map(([name, value]) => {
            const width = (Math.abs(value) / max) * 50;
            const left = value >= 0 ? 50 : 50 - width;
            const colour = value >= 0 ? "var(--approve)" : "var(--decline)";
            return `
            <div class="item">
                <div class="track">
                    <span class="name">${featureLabel(name)}</span>
                    <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--line-strong)"></div>
                    <div class="fill" style="left:${left}%;width:${width}%;background:${colour};opacity:.6"></div>
                </div>
                <span class="num tiny" style="color:${colour}">${value >= 0 ? "+" : ""}${value.toFixed(3)}</span>
            </div>`;
        })
        .join("");
}

function renderError(err) {
    const details = err.details && Object.keys(err.details).length
        ? `<pre class="tiny mono" style="white-space:pre-wrap;color:var(--muted)">${JSON.stringify(err.details, null, 2)}</pre>`
        : "";
    toast(err.message, "bad");
    return html`<div class="card">
        <div class="callout warn">
            <span class="label">${err.code || "error"}</span>
            ${err.message}
            ${raw(details)}
        </div>
    </div>`;
}
