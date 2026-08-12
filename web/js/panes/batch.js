import { api, pollJob } from "../api.js";
import { int, num, pct, pp, pvalue, shortHash } from "../format.js";
import { html, mount, onAction, raw, skeleton } from "../render.js";
import { toast } from "../store.js";

let cleanup = [];

export function mountPane(root) {
    mount(root, layout());

    const input = root.querySelector("#batch-file");
    const zone = root.querySelector("#dropzone");

    cleanup.push(
        onAction(root, {
            pick: () => input.click(),
            sample: (el) => runSample(root, Number(el.dataset.n)),
            template: () => window.open("/api/batch/template.csv", "_blank"),
        }),
    );

    const onChange = () => input.files[0] && upload(root, input.files[0]);
    input.addEventListener("change", onChange);
    cleanup.push(() => input.removeEventListener("change", onChange));

    const over = (e) => {
        e.preventDefault();
        zone.classList.add("over");
    };
    const leave = () => zone.classList.remove("over");
    const drop = (e) => {
        e.preventDefault();
        zone.classList.remove("over");
        const file = e.dataTransfer.files[0];
        if (file) upload(root, file);
    };
    zone.addEventListener("dragover", over);
    zone.addEventListener("dragleave", leave);
    zone.addEventListener("drop", drop);
    cleanup.push(() => {
        zone.removeEventListener("dragover", over);
        zone.removeEventListener("dragleave", leave);
        zone.removeEventListener("drop", drop);
    });
}

export function unmountPane() {
    cleanup.forEach((fn) => fn());
    cleanup = [];
}

function layout() {
    return html`
        <div class="page-head">
            <span class="label">Pane 5</span>
            <h1>Portfolio review</h1>
            <p>
                Score a book of business, not one application. Every row lands in the
                decision ledger as a single chained batch with one receipt hash, and the
                same three fairness analyses run against that batch rather than against
                the static evaluation frame. This is the quarterly review a fair lending
                team actually performs.
            </p>
        </div>

        <div class="card">
            <div class="dropzone" id="dropzone" data-action="pick">
                <div class="big">Drop a portfolio CSV here, or click to choose one</div>
                <div class="tiny dim">
                    The file may carry race, ethnicity, sex, and age columns — fairness
                    cannot be measured without them. Those columns are parsed into a
                    separate object that the scoring path cannot accept.
                </div>
                <input type="file" id="batch-file" accept=".csv" style="display:none">
            </div>

            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:16px;align-items:center">
                <!-- 10,000 first: below roughly that, the controlled regression has
                     too few minority observations to converge and correctly reports
                     that it cannot say anything. The 2,000 option is kept precisely
                     to show that guard working. -->
                <span class="label" style="margin-right:4px">Or sample one</span>
                <button class="ghost tiny" data-action="sample" data-n="10000">10,000 applications</button>
                <button class="ghost tiny" data-action="sample" data-n="20000">20,000</button>
                <button class="ghost tiny" data-action="sample" data-n="2000">2,000 (too small to analyse)</button>
                <button class="ghost tiny" data-action="template" style="margin-left:auto">
                    Download template CSV
                </button>
            </div>

            <div class="progress-bar" id="batch-progress" style="display:none">
                <div class="fill" style="width:0%"></div>
            </div>
            <div class="progress-log" id="batch-log"></div>
        </div>

        <div id="batch-result"></div>
    `;
}

async function runSample(root, n) {
    await run(root, () => api.sampleBatch(n), `sampling ${n.toLocaleString()} applications`);
}

async function upload(root, file) {
    const form = new FormData();
    form.append("file", file);
    await run(root, () => api.submitBatch(form), `uploading ${file.name}`);
}

async function run(root, submit, label) {
    const log = root.querySelector("#batch-log");
    const bar = root.querySelector("#batch-progress");
    const fill = bar.querySelector(".fill");
    const result = root.querySelector("#batch-result");

    bar.style.display = "block";
    fill.style.width = "5%";
    log.innerHTML = `<div class="line">· ${label}</div>`;
    mount(result, "");

    try {
        const started = await submit();
        const job = await pollJob(api.batchStatus, started.job_id, (j) => {
            fill.style.width = `${Math.round(Math.max(0.05, j.progress) * 100)}%`;
            if (j.detail) log.innerHTML = `<div class="line">· ${j.detail}</div>`;
        });

        if (job.status === "error") throw new Error(job.error);

        log.innerHTML = `<div class="line done">✓ Scored and recorded</div>`;
        mount(result, renderBatch(job.result));
        toast(`${int(job.result.row_count)} applications scored and chained`);
    } catch (err) {
        const details = err.details?.errors;
        log.innerHTML = `<div class="line">✕ ${err.message}</div>`;
        mount(
            result,
            html`<div class="card">
                <div class="callout warn">
                    <span class="label">${err.code || "error"}</span>${err.message}
                </div>
                ${raw(
                    details && details.length
                        ? `<div class="table-wrap" style="margin-top:16px"><table>
                             <thead><tr><th class="num">Row</th><th>Column</th><th>Problem</th></tr></thead>
                             <tbody>${details
                                 .slice(0, 25)
                                 .map(
                                     (e) =>
                                         `<tr><td class="num">${e.row}</td><td class="mono tiny">${e.column}</td><td>${e.message}</td></tr>`,
                                 )
                                 .join("")}</tbody>
                           </table></div>`
                        : "",
                )}
            </div>`,
        );
        toast("Batch failed", "bad");
    }
}

function renderBatch(b) {
    const f = b.fairness;

    return html`
    <div class="card">
        <div class="card-head">
            <div>
                <h2>Batch ${shortHash(b.batch_id, 8, 4)}</h2>
                <p>${b.filename || "sampled portfolio"} · model ${b.model_version}</p>
            </div>
            <a class="btn tiny" href="${api.batchResultsUrl(b.batch_id)}">Download results CSV</a>
        </div>

        <div class="stats">
            <div class="stat">
                <div class="k">Applications</div>
                <div class="v">${int(b.row_count)}</div>
                <div class="s">ledger #${int(b.first_seq)}–${int(b.last_seq)}</div>
            </div>
            <div class="stat">
                <div class="k">Approval rate</div>
                <div class="v">${pct(b.approval_rate)}</div>
                <div class="s">${int(b.declined)} declined</div>
            </div>
            <div class="stat">
                <div class="k">Batch receipt</div>
                <div class="v mono" style="font-size:13px">${shortHash(b.batch_root_hash, 8, 6)}</div>
                <div class="s">hash of the final chained row</div>
            </div>
            <div class="stat">
                <div class="k">Row errors</div>
                <div class="v" style="color:${b.errors.length ? "var(--warn)" : "var(--approve)"}">
                    ${b.errors.length}
                </div>
                <div class="s">rejected at parse time</div>
            </div>
        </div>
    </div>

    ${raw(f.status === "ok" ? batchFairness(f) : batchFairnessUnavailable(f))}
    `;
}

function batchFairness(f) {
    const raw_ = f.raw_disparity.race;
    const ctrl = f.controlled;

    return `
    <div class="card">
        <div class="card-head">
            <div>
                <h2>Fair lending analysis of this batch</h2>
                <p>
                    The same three analyses as the static evaluation frame, computed on
                    the ${int(f.n)} applications you just scored.
                </p>
            </div>
        </div>

        <h3 style="margin-bottom:10px">Approval rate by group</h3>
        <div class="table-wrap"><table>
            <thead><tr><th>Group</th><th class="num">n</th><th class="num">Approval rate</th>
            <th class="num">Ratio</th><th>Screen</th></tr></thead>
            <tbody>${raw_.groups
                .map((g) =>
                    g.status !== "ok"
                        ? `<tr><td>${g.group}</td><td class="num">${int(g.n)}</td>
                           <td colspan="3" class="faint tiny"><em>insufficient sample</em></td></tr>`
                        : `<tr class="${g.four_fifths_flag ? "is-flagged" : g.is_reference ? "is-reference" : ""}">
                            <td><strong>${g.group}</strong></td>
                            <td class="num">${int(g.n)}</td>
                            <td class="num">${pct(g.approval_rate)}</td>
                            <td class="num">${num(g.ratio, 3)}</td>
                            <td>${
                                g.four_fifths_flag
                                    ? '<span class="pill warn">below 0.80</span>'
                                    : '<span class="pill approve">above 0.80</span>'
                            }</td>
                        </tr>`,
                )
                .join("")}</tbody>
        </table></div>

        ${
            ctrl.status === "ok"
                ? `<h3 style="margin:24px 0 10px">Disparity surviving controls</h3>
                   <div class="table-wrap"><table>
                     <thead><tr><th>Group</th><th class="num">n</th><th class="num">Coefficient</th>
                     <th class="num">p</th><th class="num">Marginal effect</th></tr></thead>
                     <tbody>${Object.entries(ctrl.groups)
                         .map(([n, g]) =>
                             g.status !== "ok"
                                 ? `<tr><td>${n}</td><td class="num">${int(g.n_focal)}</td>
                                    <td colspan="3" class="faint tiny"><em>${String(g.status).replace(/_/g, " ")}</em></td></tr>`
                                 : `<tr class="${g.significant ? "is-flagged" : ""}">
                                     <td><strong>${n}</strong></td>
                                     <td class="num">${int(g.n_focal)}</td>
                                     <td class="num coef">${g.coef >= 0 ? "+" : ""}${num(g.coef, 3)}</td>
                                     <td class="num coef ${g.significant ? "sig" : ""}">${pvalue(g.p_value)}</td>
                                     <td class="num coef ${g.significant ? "sig" : ""}">${pp(g.ame_pp)}</td>
                                   </tr>`,
                         )
                         .join("")}</tbody>
                   </table></div>`
                : `<div class="callout warn" style="margin-top:20px">
                     <span class="label">Controlled analysis not available for this batch</span>
                     ${ctrl.reason || String(ctrl.status).replace(/_/g, " ")}
                     <div style="margin-top:8px">
                       A batch this size carries too few applications from any one
                       protected group to fit the regression. Reporting nothing is the
                       correct answer; reporting a coefficient here would be inventing
                       precision the data cannot support.
                     </div>
                   </div>`
        }
    </div>`;
}

function batchFairnessUnavailable(f) {
    return `
    <div class="card">
        <div class="callout warn">
            <span class="label">Fairness analysis not available for this batch</span>
            ${f.note}
        </div>
    </div>`;
}
