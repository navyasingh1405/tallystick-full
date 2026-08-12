import { api } from "../api.js";
import { featureLabel, int, num, pct, pp, pvalue } from "../format.js";
import { empty, html, mount, onAction, raw, skeleton } from "../render.js";
import { toast } from "../store.js";

let cleanup = [];
let frontierData = null;

export function mountPane(root) {
    mount(root, layout());

    cleanup.push(
        onAction(root, {
            adopt: (el) => runAdoption(root, el.dataset.value),
        }),
    );

    load(root);
}

export function unmountPane() {
    cleanup.forEach((fn) => fn());
    cleanup = [];
}

function layout() {
    return html`
        <div class="page-head">
            <span class="label">Pane 3</span>
            <h1>Fair lending analysis</h1>
            <p>
                Three analyses on the out-of-time frame, which the model has never seen.
                Protected attributes live only on that frame; they are structurally
                incapable of reaching a feature vector.
            </p>
        </div>
        <div id="fairness-body"></div>
    `;
}

async function load(root) {
    const body = root.querySelector("#fairness-body");
    mount(body, skeleton(6));
    try {
        const [fairness, frontier] = await Promise.all([api.fairness(), api.frontier()]);
        frontierData = frontier;
        mount(body, render(fairness, frontier));
    } catch (err) {
        mount(
            body,
            html`<div class="card"><div class="callout warn">
                <span class="label">${err.code}</span>${err.message}
            </div></div>`,
        );
    }
}

function render(f, frontier) {
    return html`
        ${raw(summaryStats(f))}
        ${raw(rawDisparity(f))}
        ${raw(controlled(f))}
        ${raw(proxyScan(f))}
        ${raw(ldaSection(frontier))}
        <div id="adoption"></div>
    `;
}

function summaryStats(f) {
    return `
    <div class="card">
        <div class="stats">
            <div class="stat">
                <div class="k">Evaluation frame</div>
                <div class="v">${int(f.evaluation_frame.n)}</div>
                <div class="s">${f.evaluation_frame.activity_year}, never trained on</div>
            </div>
            <div class="stat">
                <div class="k">Four-fifths flags</div>
                <div class="v" style="color:${f.summary.four_fifths_flags ? "var(--warn)" : "var(--approve)"}">
                    ${f.summary.four_fifths_flags}
                </div>
                <div class="s">screening heuristic</div>
            </div>
            <div class="stat">
                <div class="k">Significant after controls</div>
                <div class="v" style="color:${f.summary.significant_controlled ? "var(--warn)" : "var(--approve)"}">
                    ${f.summary.significant_controlled}
                </div>
                <div class="s">the analysis that carries weight</div>
            </div>
            <div class="stat">
                <div class="k">Proxy flags</div>
                <div class="v" style="color:${f.summary.proxy_flags ? "var(--warn)" : "var(--approve)"}">
                    ${f.summary.proxy_flags}
                </div>
                <div class="s">features above the noise floor</div>
            </div>
        </div>
    </div>`;
}

function rawDisparity(f) {
    const block = (attr) => {
        const b = f.raw_disparity[attr];
        if (!b) return "";
        return `
        <h3 style="margin:20px 0 10px">${b.label}</h3>
        <div class="table-wrap"><table>
            <thead><tr>
                <th>Group</th><th class="num">n</th><th class="num">Approval rate</th>
                <th class="num">95% interval</th><th class="num">Ratio to highest</th><th>Screen</th>
            </tr></thead>
            <tbody>${b.groups
                .map((g) => {
                    if (g.status !== "ok") {
                        return `<tr><td>${g.group}</td><td class="num">${int(g.n)}</td>
                            <td colspan="4" class="faint tiny"><em>insufficient sample — no rate is reported</em></td></tr>`;
                    }
                    return `<tr class="${g.four_fifths_flag ? "is-flagged" : g.is_reference ? "is-reference" : ""}">
                        <td><strong>${g.group}</strong>${g.is_reference ? ' <span class="pill">reference</span>' : ""}</td>
                        <td class="num">${int(g.n)}</td>
                        <td class="num">${pct(g.approval_rate)}</td>
                        <td class="num faint tiny">${pct(g.ci_low)}–${pct(g.ci_high)}</td>
                        <td class="num">${num(g.ratio, 3)}</td>
                        <td>${
                            g.four_fifths_flag
                                ? '<span class="pill warn">below 0.80</span>'
                                : '<span class="pill approve">above 0.80</span>'
                        }</td>
                    </tr>`;
                })
                .join("")}</tbody>
        </table></div>`;
    };

    return `
    <div class="card">
        <div class="card-head">
            <div><h2>7a · Raw approval rate disparity</h2></div>
        </div>
        <div class="callout warn">
            <span class="label">What the four-fifths rule is, and is not</span>
            ${f.raw_disparity.caveat}
        </div>
        ${["race", "sex", "age"].map(block).join("")}
    </div>`;
}

function controlled(f) {
    const spec = (attr) => {
        const s = f.controlled.specifications[attr]?.primary;
        if (!s) return "";
        if (s.status !== "ok") {
            return `<h3 style="margin:20px 0 10px">${s.label}</h3>
                    <div class="tiny faint"><em>${s.reason || s.status}</em></div>`;
        }
        return `
        <h3 style="margin:20px 0 10px">${s.label} <span class="tiny faint">— reference ${s.reference_group}</span></h3>
        <div class="table-wrap"><table>
            <thead><tr>
                <th>Group</th><th class="num">n</th><th class="num">Coefficient</th>
                <th class="num">Std. error</th><th class="num">p</th>
                <th class="num">95% interval</th><th class="num">Marginal effect</th>
            </tr></thead>
            <tbody>${Object.entries(s.groups)
                .map(([name, g]) => {
                    if (g.status !== "ok") {
                        return `<tr><td>${name}</td><td class="num">${int(g.n_focal)}</td>
                            <td colspan="5" class="faint tiny"><em>${String(g.status).replace(/_/g, " ")}</em></td></tr>`;
                    }
                    return `<tr class="${g.significant ? "is-flagged" : ""}">
                        <td><strong>${name}</strong></td>
                        <td class="num">${int(g.n_focal)}</td>
                        <td class="num coef">${g.coef >= 0 ? "+" : ""}${num(g.coef, 3)}</td>
                        <td class="num faint">${num(g.std_err, 3)}</td>
                        <td class="num coef ${g.significant ? "sig" : ""}">${pvalue(g.p_value)}</td>
                        <td class="num faint tiny">${num(g.ci_low, 2)} to ${num(g.ci_high, 2)}</td>
                        <td class="num coef ${g.significant ? "sig" : ""}">${pp(g.ame_pp)}</td>
                    </tr>`;
                })
                .join("")}</tbody>
        </table></div>
        <div class="tiny faint" style="margin-top:8px">
            Fitted on ${int(s.n_obs)} applications with ${s.controls_used.length} controls,
            standard errors clustered by county; McFadden pseudo R² ${num(s.pseudo_r2, 4)}.
        </div>`;
    };

    const hist = f.controlled.historical_labels;
    const prim = f.controlled.specifications.race?.primary?.groups || {};
    const comparison =
        hist && hist.status === "ok"
            ? `
        <h3 style="margin:24px 0 10px">The model's decisions against the decisions it learned from</h3>
        <p class="tiny dim">${f.controlled.historical_note}</p>
        <div class="table-wrap"><table>
            <thead><tr><th>Group</th><th class="num">Model decision</th>
            <th class="num">Historical decision</th><th class="num">Difference</th></tr></thead>
            <tbody>${Object.entries(prim)
                .filter(([n, g]) => g.status === "ok" && hist.groups[n]?.status === "ok")
                .map(
                    ([n, g]) => `<tr>
                        <td><strong>${n}</strong></td>
                        <td class="num coef">${pp(g.ame_pp)}</td>
                        <td class="num coef">${pp(hist.groups[n].ame_pp)}</td>
                        <td class="num faint">${pp(g.ame_pp - hist.groups[n].ame_pp)}</td>
                    </tr>`,
                )
                .join("")}</tbody>
        </table></div>`
            : "";

    return `
    <div class="card">
        <div class="card-head">
            <div>
                <h2>7b · Disparity surviving controls</h2>
                <p>
                    A logistic regression of the model's decision on protected class plus
                    the model's own credit factors. A raw gap can be explained by
                    differences in the applications; a coefficient that stays significant
                    after the lender's own stated credit factors are included cannot be.
                </p>
            </div>
        </div>
        ${["race", "sex", "age"].map(spec).join("")}
        ${comparison}
        <div class="callout" style="margin-top:20px">
            <span class="label">On controlling for geography</span>
            ${f.controlled.geography_note}
        </div>
    </div>`;
}

function proxyScan(f) {
    const p = f.proxy;
    const max = Math.max(...p.ranked.map((r) => r.max_nmi), p.threshold) * 1.08;
    const thresholdPct = (p.threshold / max) * 100;

    return `
    <div class="card">
        <div class="card-head">
            <div>
                <h2>7c · Proxy scan</h2>
                <p>
                    How much each feature reveals about a protected attribute. Excluding
                    race from a model does not remove it if another feature carries the
                    same information.
                </p>
            </div>
        </div>

        <div class="barlist">
            ${p.ranked
                .slice(0, 12)
                .map(
                    (r) => `
                <div class="item">
                    <div class="track">
                        <span class="name">${featureLabel(r.feature)}${
                            r.in_model ? "" : ' <span class="faint">(not in model)</span>'
                        }</span>
                        <div class="fill ${r.flagged ? "flagged" : ""}" style="width:${(r.max_nmi / max) * 100}%"></div>
                        <div class="threshold" style="left:${thresholdPct}%"></div>
                    </div>
                    <span class="num tiny ${r.flagged ? "" : "faint"}">${num(r.max_nmi, 4)}</span>
                </div>`,
                )
                .join("")}
            <div class="item">
                <div class="track" style="opacity:.6">
                    <span class="name faint">Random negative control</span>
                    <div class="fill" style="width:${(p.negative_control.max_nmi / max) * 100}%;background:var(--faint)"></div>
                    <div class="threshold" style="left:${thresholdPct}%"></div>
                </div>
                <span class="num tiny faint">${num(p.negative_control.max_nmi, 4)}</span>
            </div>
        </div>

        <div class="callout" style="margin-top:16px">
            <span class="label">Calibration</span>
            ${p.threshold_note}
        </div>

        ${
            p.set_level?.status === "ok"
                ? `<div class="callout accent" style="margin-top:12px">
                     <span class="label">Joint predictability</span>
                     ${p.set_level.interpretation} ${p.set_level.note}
                   </div>`
                : ""
        }
    </div>`;
}

function ldaSection(frontier) {
    const inc = frontier.incumbent;
    const options = frontier.frontier.filter((r) => r.id !== "incumbent");

    return `
    <div class="card">
        <div class="card-head">
            <div>
                <h2>8 · Less discriminatory alternative search</h2>
                <p>${frontier.doctrine_note}</p>
            </div>
        </div>

        <div class="headline-claim">${frontier.headline}</div>

        <div class="frontier-figure">
            <img src="/api/lda/frontier.png?t=${Date.now()}" alt="Pareto frontier of challenger models">
        </div>

        <div class="stats" style="margin-top:16px">
            <div class="stat">
                <div class="k">Challengers trained</div>
                <div class="v">${frontier.n_challengers}</div>
                <div class="s">enumerated before the search ran</div>
            </div>
            <div class="stat">
                <div class="k">On the frontier</div>
                <div class="v">${frontier.frontier.length}</div>
                <div class="s">not beaten on both axes</div>
            </div>
            <div class="stat">
                <div class="k">Incumbent AUC</div>
                <div class="v">${num(inc.auc, 4)}</div>
                <div class="s">out-of-time frame</div>
            </div>
            <div class="stat">
                <div class="k">Incumbent disparity</div>
                <div class="v">${pp(inc.disparity_ame)}</div>
                <div class="s">${frontier.objective.focal_group} applicants</div>
            </div>
        </div>

        <h3 style="margin:24px 0 10px">Frontier models</h3>
        <div class="table-wrap"><table>
            <thead><tr>
                <th>Model</th><th>Feature set</th><th class="num">Features</th>
                <th class="num">AUC</th><th class="num">Disparity</th>
                <th class="num">Approval rate</th><th></th>
            </tr></thead>
            <tbody>
                <tr class="is-reference">
                    <td><strong>Incumbent</strong></td>
                    <td>baseline</td>
                    <td class="num">${inc.n_features}</td>
                    <td class="num">${num(inc.auc, 4)}</td>
                    <td class="num">${pp(inc.disparity_ame)}</td>
                    <td class="num">${pct(inc.approval_rate)}</td>
                    <td><span class="pill accent">deployed</span></td>
                </tr>
                ${options
                    .map(
                        (r) => `<tr>
                        <td class="mono">${r.id}</td>
                        <td>${r.subset.replace(/_/g, " ")}</td>
                        <td class="num">${r.n_features}</td>
                        <td class="num">${num(r.auc, 4)}</td>
                        <td class="num ${Math.abs(r.disparity_ame) < Math.abs(inc.disparity_ame) ? "coef sig" : ""}">${pp(r.disparity_ame)}</td>
                        <td class="num">${pct(r.approval_rate)}</td>
                        <td><button class="tiny ghost" data-action="adopt" data-value="${r.id}">Impact of adopting</button></td>
                    </tr>`,
                    )
                    .join("")}
            </tbody>
        </table></div>

        <div class="callout" style="margin-top:16px">
            <span class="label">What the search steered by</span>
            ${frontier.objective.metric}
        </div>
    </div>`;
}

async function runAdoption(root, challengerId) {
    const target = root.querySelector("#adoption");
    target.scrollIntoView({ behavior: "smooth", block: "start" });
    mount(
        target,
        html`<div class="card">
            <div class="card-head"><h2>Adoption impact · ${challengerId}</h2></div>
            <div style="display:flex;gap:10px;align-items:center;color:var(--muted)">
                <span class="spinner"></span> Training the challenger and comparing outcomes…
            </div>
        </div>`,
    );

    try {
        const r = await api.adopt(challengerId, true);
        mount(target, renderAdoption(r));
    } catch (err) {
        toast(err.message, "bad");
        mount(target, html`<div class="card"><div class="callout warn">${err.message}</div></div>`);
    }
}

function renderAdoption(r) {
    const inc = r.performance.incumbent;
    const chl = r.performance.challenger;

    return html`
    <div class="card">
        <div class="card-head">
            <div>
                <h2>Adoption impact · ${r.challenger_id}</h2>
                <p>${r.threshold.note}</p>
            </div>
        </div>

        <div class="stats">
            <div class="stat">
                <div class="k">AUC change</div>
                <div class="v">${num(chl.auc - inc.auc, 4)}</div>
                <div class="s">${num(inc.auc, 4)} → ${num(chl.auc, 4)}</div>
            </div>
            <div class="stat">
                <div class="k">Approvals gained</div>
                <div class="v" style="color:var(--approve)">${int(r.flips.approvals_gained)}</div>
                <div class="s">declined before, approved now</div>
            </div>
            <div class="stat">
                <div class="k">Approvals lost</div>
                <div class="v" style="color:var(--decline)">${int(r.flips.approvals_lost)}</div>
                <div class="s">approved before, declined now</div>
            </div>
            <div class="stat">
                <div class="k">Unchanged</div>
                <div class="v">${int(r.flips.unchanged)}</div>
                <div class="s">of ${int(r.flips.n_evaluated)} applications</div>
            </div>
        </div>

        <h3 style="margin:24px 0 10px">Who changes outcome</h3>
        <div class="table-wrap"><table>
            <thead><tr><th>Group</th><th class="num">n</th><th class="num">Gained</th>
            <th class="num">Lost</th><th class="num">Net</th><th class="num">Net rate</th></tr></thead>
            <tbody>${raw(
                r.flips.composition
                    .map(
                        (c) => `<tr>
                    <td><strong>${c.group}</strong></td>
                    <td class="num">${int(c.n)}</td>
                    <td class="num" style="color:var(--approve)">${int(c.approvals_gained)}</td>
                    <td class="num" style="color:var(--decline)">${int(c.approvals_lost)}</td>
                    <td class="num">${c.net >= 0 ? "+" : ""}${int(c.net)}</td>
                    <td class="num">${pp(c.net_rate_pp)}</td>
                </tr>`,
                    )
                    .join(""),
            )}</tbody>
        </table></div>

        <h3 style="margin:24px 0 10px">Disparity change</h3>
        <div class="table-wrap"><table>
            <thead><tr><th>Group</th><th class="num">n</th><th class="num">Incumbent</th>
            <th class="num">Challenger</th><th class="num">Change</th><th></th></tr></thead>
            <tbody>${raw(
                r.disparity.groups
                    .map((g) =>
                        g.status !== "ok"
                            ? `<tr><td>${g.group}</td><td colspan="5" class="faint tiny"><em>${String(g.status).replace(/_/g, " ")}</em></td></tr>`
                            : `<tr>
                                <td><strong>${g.group}</strong></td>
                                <td class="num">${int(g.n)}</td>
                                <td class="num coef">${pp(g.incumbent_ame_pp)}</td>
                                <td class="num coef">${pp(g.challenger_ame_pp)}</td>
                                <td class="num coef">${pp(g.delta_pp)}</td>
                                <td>${
                                    g.improved
                                        ? '<span class="pill approve">improved</span>'
                                        : '<span class="pill warn">worse</span>'
                                }</td>
                            </tr>`,
                    )
                    .join(""),
            )}</tbody>
        </table></div>

        <div class="callout" style="margin-top:16px">
            <span class="label">Before deciding</span>
            ${r.interpretation_note}
        </div>

        ${raw(
            r.feature_delta.removed.length || r.feature_delta.added.length
                ? `<div class="callout accent" style="margin-top:12px">
                     <span class="label">Feature changes</span>
                     ${r.feature_delta.removed.length ? `Removed: ${r.feature_delta.removed.map(featureLabel).join(", ")}. ` : ""}
                     ${r.feature_delta.added.length ? `Added: ${r.feature_delta.added.map(featureLabel).join(", ")}.` : ""}
                   </div>`
                : "",
        )}
    </div>`;
}
