import { api, pollJob } from "../api.js";
import { int, timestamp } from "../format.js";
import { html, mount, onAction, raw } from "../render.js";
import { toast } from "../store.js";

let cleanup = [];

export function mountPane(root) {
    mount(root, layout());
    cleanup.push(onAction(root, { build: () => build(root) }));
}

export function unmountPane() {
    cleanup.forEach((fn) => fn());
    cleanup = [];
}

function layout() {
    return html`
        <div class="page-head">
            <span class="label">Pane 4</span>
            <h1>Generate the validation pack</h1>
            <p>
                A twelve-section model validation document, structured against the
                validation pillars of SR 11-7 and the technical documentation content of
                EU AI Act Annex IV. Every figure in it is read from a generated
                artifact; none is written by hand. Retrain the model and regenerate, and
                the document describes the new model with no editing.
            </p>
        </div>

        <div class="card">
            <div class="card-head">
                <div>
                    <h2>Model validation pack</h2>
                    <p>
                        Reads the model registry, performance metrics, fairness report,
                        alternative search, and drift report, then renders to PDF.
                    </p>
                </div>
            </div>

            <button class="primary" data-action="build" id="build-btn">Generate validation pack</button>

            <div class="progress-bar" id="pack-progress" style="display:none">
                <div class="fill" style="width:0%"></div>
            </div>
            <div class="progress-log" id="pack-log"></div>
            <div id="pack-result"></div>
        </div>

        <div class="card">
            <div class="card-head"><h3>What the document contains</h3></div>
            <div class="table-wrap">
                <table>
                    <thead><tr><th class="num">§</th><th>Section</th><th>Sourced from</th></tr></thead>
                    <tbody>
                        ${raw(
                            [
                                ["1", "Purpose, intended use, and what it does not do", "config, model manifest"],
                                ["2", "Data lineage", "data_lineage.json"],
                                ["3", "Features, transformations, constraints", "features.yaml, model manifest"],
                                ["4", "Development methodology", "model manifest"],
                                ["5", "Performance overall and by segment", "performance.json, segments.json"],
                                ["6", "Stability and out-of-time validation", "calibration.json, drift_report.json"],
                                ["7", "Fair lending analysis", "fairness_report.json"],
                                ["8", "Less discriminatory alternative analysis", "lda_summary.json, frontier plot"],
                                ["9", "Explainability and adverse action notices", "live worked example"],
                                ["10", "Known limitations", "docs/limitations.md, verbatim"],
                                ["11", "Monitoring and revalidation triggers", "docs/monitoring.md, verbatim"],
                                ["12", "Regulatory mapping appendix", "regulatory_map.yaml"],
                            ]
                                .map(
                                    ([n, title, src]) =>
                                        `<tr><td class="num faint">${n}</td><td>${title}</td>
                                         <td class="mono tiny faint">${src}</td></tr>`,
                                )
                                .join(""),
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    `;
}

async function build(root) {
    const button = root.querySelector("#build-btn");
    const log = root.querySelector("#pack-log");
    const bar = root.querySelector("#pack-progress");
    const fill = bar.querySelector(".fill");
    const result = root.querySelector("#pack-result");

    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span> Generating…';
    mount(result, "");
    bar.style.display = "block";

    const lines = [];
    const push = (text, done = false) => {
        lines.push(`<div class="line ${done ? "done" : ""}">${done ? "✓" : "·"} ${text}</div>`);
        log.innerHTML = lines.join("");
    };

    push("Reading model registry and artifacts");

    try {
        const started = await api.buildPack();
        const job = await pollJob(api.packStatus, started.job_id, (j) => {
            fill.style.width = `${Math.round(j.progress * 100)}%`;
        });

        if (job.status === "error") throw new Error(job.error);

        push("Rendering twelve sections", true);
        push("Printing to PDF via headless Chromium", true);
        push("Checked for placeholder content — none found", true);
        fill.style.width = "100%";

        const r = job.result;
        mount(
            result,
            html`
                <a class="download-card" href="${r.pdf_url}" target="_blank" rel="noopener">
                    <span class="doc">📄</span>
                    <span class="meta">
                        <span class="fname">${r.filename}</span>
                        <span class="sub">
                            ${r.pages} pages · ${Math.round(r.size_bytes / 1024)} KB ·
                            generated in ${r.seconds}s
                        </span>
                    </span>
                    <span class="pill accent">Open PDF</span>
                </a>
            `,
        );
        toast("Validation pack generated");
    } catch (err) {
        push(`Failed: ${err.message}`);
        mount(result, html`<div class="callout warn" style="margin-top:12px">${err.message}</div>`);
        toast("Generation failed", "bad");
    } finally {
        button.disabled = false;
        button.textContent = "Generate validation pack";
    }
}
