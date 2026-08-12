import { api } from "../api.js";
import { int, num, sci, shortHash, timestamp } from "../format.js";
import { empty, html, mount, onAction, onSubmit, raw, skeleton } from "../render.js";
import { copyToClipboard, store, toast } from "../store.js";

let cleanup = [];

/* The three verification steps are revealed one at a time, roughly a quarter second
   apart. The work really is happening — the chain walk and the replay both run
   server-side before any of this renders — but showing the steps resolve in sequence
   is what makes it read as verification rather than as a lookup. */
const STEP_DELAY = 260;

export function mountPane(root) {
    mount(root, layout());
    const panel = root.querySelector("#ledger-panel");

    cleanup.push(
        onSubmit(root, async (form) => {
            if (form.id !== "lookup-form") return;
            const id = form.querySelector("[name=decision_id]").value.trim();
            if (!id) return;
            await load(panel, id);
        }),
    );

    cleanup.push(
        onAction(root, {
            copy: (el) => copyToClipboard(el.dataset.value),
            verify: (el) => runVerification(panel, el.dataset.value),
            notice: (el) => window.open(api.adverseActionUrl(el.dataset.value), "_blank"),
            pick: (el) => {
                root.querySelector("[name=decision_id]").value = el.dataset.value;
                load(panel, el.dataset.value);
            },
            refresh: () => loadRecent(root),
            showOverride: (el) => {
                const box = panel.querySelector("#override-box");
                box.hidden = !box.hidden;
                if (!box.hidden) box.querySelector("textarea").focus();
                el.textContent = box.hidden ? "Override this decision" : "Cancel override";
            },
            submitOverride: (el) => submitOverride(panel, el.dataset.value),
        }),
    );

    loadRecent(root);

    const pending = store.get().lookupId;
    if (pending) {
        root.querySelector("[name=decision_id]").value = pending;
        store.set({ lookupId: "" });
        load(panel, pending);
    } else {
        mount(panel, empty("Paste a decision identifier, or pick one from the recent decisions below."));
    }
}

export function unmountPane() {
    cleanup.forEach((fn) => fn());
    cleanup = [];
}

function layout() {
    return html`
        <div class="page-head">
            <span class="label">Pane 2</span>
            <h1>Retrieve and verify a decision</h1>
            <p>
                Every decision is chained to the one before it by hash. Retrieval walks
                the chain from genesis to prove nothing has been altered, then loads the
                exact model version the decision was made with and re-scores the stored
                application. This is the question an examiner asks: explain the decision
                you made on this applicant, months ago, and prove it.
            </p>
        </div>

        <div class="card">
            <form id="lookup-form" style="display:flex;gap:12px;align-items:flex-end">
                <div class="field" style="flex:1;margin:0">
                    <label for="decision_id">Decision identifier</label>
                    <input type="text" id="decision_id" name="decision_id" class="mono"
                           placeholder="8f3a1c62-…-c21" autocomplete="off">
                </div>
                <button type="submit" class="primary">Retrieve</button>
            </form>
        </div>

        <div id="ledger-panel" style="margin-top:16px"></div>

        <div class="card" style="margin-top:16px">
            <div class="card-head">
                <h3>Recent decisions</h3>
                <button class="tiny ghost" data-action="refresh">Refresh</button>
            </div>
            <div id="recent"></div>
        </div>
    `;
}

async function loadRecent(root) {
    const target = root.querySelector("#recent");
    mount(target, skeleton(3));
    try {
        const data = await api.recent(12);
        if (!data.rows.length) {
            mount(target, empty("The ledger is empty. Score an application to create the first entry."));
            return;
        }
        mount(
            target,
            html`
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr><th class="num">#</th><th>Recorded</th><th>Decision</th>
                            <th class="num">P(approve)</th><th>Identifier</th><th></th></tr>
                        </thead>
                        <tbody>
                            ${raw(
                                data.rows
                                    .map(
                                        (r) => `
                                <tr>
                                    <td class="num faint">${r.seq}</td>
                                    <td class="tiny">${timestamp(r.created_at)}</td>
                                    <td><span class="pill ${r.decision === "APPROVED" ? "approve" : "decline"}">${r.decision}</span></td>
                                    <td class="num">${num(r.probability, 4)}</td>
                                    <td class="mono tiny">${shortHash(r.decision_id, 8, 6)}</td>
                                    <td><button class="tiny ghost" data-action="pick" data-value="${r.decision_id}">Open</button></td>
                                </tr>`,
                                    )
                                    .join(""),
                            )}
                        </tbody>
                    </table>
                </div>
                <div class="tiny faint" style="margin-top:12px">
                    Ledger height ${int(data.height)} · head
                    <span class="mono">${shortHash(data.head_hash, 10, 8)}</span>
                </div>
            `,
        );
    } catch (err) {
        mount(target, html`<div class="callout warn">${err.message}</div>`);
    }
}

async function load(panel, decisionId) {
    mount(panel, skeleton(5));
    try {
        const d = await api.decision(decisionId);
        mount(panel, renderDecision(d));
    } catch (err) {
        toast(err.message, "bad");
        mount(panel, html`<div class="card"><div class="callout warn">
            <span class="label">${err.code}</span>${err.message}
        </div></div>`);
    }
}

function renderDecision(d) {
    const approved = d.decision === "APPROVED";
    const prev = d.neighbours.previous;
    const next = d.neighbours.next;

    return html`
        <div class="card">
            <div class="card-head">
                <div>
                    <h2>Decision #${d.seq}</h2>
                    <p>Recorded ${timestamp(d.created_at)} under model ${d.model_version}.</p>
                </div>
                <span class="pill ${approved ? "approve" : "decline"}">${d.decision}</span>
            </div>

            <div class="chain">
                ${raw(prev ? blockHtml(prev, "previous") : "")}
                ${raw(prev ? '<div class="link"></div>' : "")}
                ${raw(blockHtml(d, "this decision", true))}
                ${raw(next ? '<div class="link"></div>' : "")}
                ${raw(next ? blockHtml(next, "next") : "")}
            </div>

            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:16px">
                <button class="primary" data-action="verify" data-value="${d.decision_id}">
                    Verify &amp; replay
                </button>
                <button class="ghost" data-action="copy" data-value="${d.decision_id}">Copy identifier</button>
                ${raw(
                    approved
                        ? ""
                        : `<button class="ghost" data-action="notice" data-value="${d.decision_id}">
                              Adverse action notice (PDF)
                           </button>`,
                )}
                ${raw(
                    d.superseded_by
                        ? ""
                        : `<button class="ghost" data-action="showOverride">Override this decision</button>`,
                )}
            </div>

            <div id="override-box" class="override-box" hidden>
                <div class="label">Record a manual override</div>
                <p class="tiny dim">
                    This does not modify the decision above. It appends a new entry
                    that supersedes it, so both the machine decision and the human one
                    stay in the record — which is the point of recording either.
                </p>
                <div class="field">
                    <label for="ov-decision">New decision</label>
                    <select id="ov-decision">
                        <option value="APPROVED">APPROVED</option>
                        <option value="DECLINED">DECLINED</option>
                    </select>
                </div>
                <div class="field">
                    <label for="ov-reason">
                        Documented reason <span class="hint">at least 10 characters</span>
                    </label>
                    <textarea id="ov-reason" rows="3"
                        placeholder="Compensating factors reviewed: verified reserves of 14 months and a documented one-off income event."></textarea>
                </div>
                <div class="override-actor tiny dim" id="override-actor">
                    Recorded against <strong>${
                        store.get().user?.email || "your signed-in account"
                    }</strong>, taken from your session rather than from this form.
                </div>
                <button class="primary" data-action="submitOverride"
                        data-value="${d.decision_id}">Record override</button>
            </div>

            <div id="verify-steps"></div>
        </div>

        <div class="card">
            <div class="card-head"><h3>Stored record</h3></div>
            <div class="grid-2" style="grid-template-columns:1fr 1fr">
                <div>
                    <div class="label" style="margin-bottom:8px">Application as received</div>
                    <div class="kv">
                        ${raw(
                            Object.entries(d.input)
                                .map(
                                    ([k, v]) =>
                                        `<dt>${k.replace(/_/g, " ")}</dt><dd>${
                                            typeof v === "boolean" ? (v ? "yes" : "no") : v
                                        }</dd>`,
                                )
                                .join(""),
                        )}
                    </div>
                </div>
                <div>
                    <div class="label" style="margin-bottom:8px">Reasons recorded</div>
                    ${raw(
                        d.reason_codes.length
                            ? `<ol style="margin:0;padding-left:18px;font-size:12.5px;color:var(--fg-dim)">
                                 ${d.reason_codes.map((c) => `<li style="margin-bottom:6px">${c.text}<span class="code mono faint"> ${c.code}</span></li>`).join("")}
                               </ol>`
                            : '<div class="tiny faint">None — this application was approved.</div>',
                    )}
                    ${raw(
                        d.superseded_by
                            ? `<div class="callout warn" style="margin-top:16px">
                                 <span class="label">Superseded</span>
                                 This decision was later overridden. The override is a separate
                                 ledger entry; this record was never modified.
                               </div>`
                            : "",
                    )}
                    ${raw(
                        d.supersedes
                            ? `<div class="callout accent" style="margin-top:16px">
                                 <span class="label">Manual override</span>
                                 Supersedes <span class="mono">${shortHash(d.supersedes, 8, 6)}</span>.
                                 ${d.override_actor ? `Recorded by ${d.override_actor}.` : ""}
                                 ${d.override_reason ? `<div style="margin-top:6px"><em>${d.override_reason}</em></div>` : ""}
                               </div>`
                            : "",
                    )}
                </div>
            </div>
        </div>
    `;
}

function blockHtml(row, label, current = false) {
    return `
    <div class="block ${current ? "current" : ""}">
        <div class="who">
            <span>${label} · #${row.seq}</span>
            <span class="mono">${shortHash(row.decision_id, 8, 4)}</span>
        </div>
        <div class="hashes">row_hash ${shortHash(row.row_hash, 16, 10)}</div>
    </div>`;
}

async function submitOverride(panel, decisionId) {
    const decision = panel.querySelector("#ov-decision").value;
    const reason = panel.querySelector("#ov-reason").value.trim();
    const button = panel.querySelector('[data-action="submitOverride"]');

    if (reason.length < 10) {
        toast("An override needs a documented reason.", "bad");
        panel.querySelector("#ov-reason").focus();
        return;
    }

    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span> Recording…';
    try {
        // No actor is sent. The server takes it from the session — a field the client
        // could set would let anyone write anyone else's name into the audit trail.
        const result = await api.override(decisionId, { decision, reason });
        toast(`Override recorded as entry #${result.seq}`);
        await load(panel, result.decision_id);
    } catch (err) {
        toast(err.message, "bad");
        button.disabled = false;
        button.textContent = "Record override";
    }
}

async function runVerification(panel, decisionId) {
    const target = panel.querySelector("#verify-steps");
    const steps = [
        { id: "chain", text: "Walk the hash chain from genesis" },
        { id: "model", text: "Load the pinned model version" },
        { id: "replay", text: "Re-score the stored application" },
    ];

    mount(
        target,
        html`<div class="verify-steps">
            ${raw(
                steps
                    .map(
                        (s) => `
                <div class="step" id="step-${s.id}">
                    <span class="mark">○</span>
                    <span>${s.text}</span>
                    <span class="detail"></span>
                </div>`,
                    )
                    .join(""),
            )}
        </div>`,
    );

    const setStep = (id, cls, mark, detail) => {
        const el = target.querySelector(`#step-${id}`);
        if (!el) return;
        el.className = `step ${cls}`;
        el.querySelector(".mark").textContent = mark;
        el.querySelector(".detail").textContent = detail || "";
    };

    steps.forEach((s) => setStep(s.id, "active", "○", "checking…"));

    let result;
    try {
        result = await api.verify(decisionId);
    } catch (err) {
        setStep("chain", "fail", "✕", err.message);
        return;
    }

    const { chain, replay } = result;
    const sequence = [
        () =>
            setStep(
                "chain",
                chain.ok ? "pass" : "fail",
                chain.ok ? "✓" : "✕",
                chain.ok
                    ? `${int(chain.rows_checked)} decisions, intact`
                    : `broken at #${chain.first_broken_seq} — ${chain.first_broken_reason}`,
            ),
        () =>
            setStep(
                "model",
                replay.model_hash_matches ? "pass" : "fail",
                replay.model_hash_matches ? "✓" : "✕",
                replay.model_hash_matches
                    ? `${replay.model_version} · hash matches`
                    : "model artifact does not match the recorded hash",
            ),
        () =>
            setStep(
                "replay",
                replay.match ? "pass" : "fail",
                replay.match ? "✓" : "✕",
                `|stored − recomputed| = ${sci(replay.abs_diff)}`,
            ),
    ];

    for (const [i, step] of sequence.entries()) {
        await new Promise((r) => setTimeout(r, STEP_DELAY * (i + 1)));
        step();
    }

    if (chain.ok && replay.match) {
        toast("Chain intact and score reproduced exactly");
    } else {
        toast("Verification failed — see the steps above", "bad");
    }
}
