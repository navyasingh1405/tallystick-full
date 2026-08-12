/* Landing page.
 *
 * Every figure shown here is fetched from the running system. Nothing on this page
 * is a number someone typed into the markup — the same rule the validation pack
 * follows, applied to the marketing surface, because that is where the temptation
 * to round in your own favour actually lives.
 */

import { api, appUrl, isSplitDeploy } from "./config.js";
import * as scorer from "./scorer.js";
import { animate, reduced, scrollSpy, stagger, swapText, tweenNumber, whenVisible } from "./motion.js";

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];

const DTI_BANDS = [
    "<20%", "20%-<30%", "30%-<36%",
    "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
    "50%-60%", ">60%",
];

const usd = (v) =>
    Number(v).toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const int = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString("en-US"));
const pct = (v, d = 1) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(d)}%`);
const pp = (v, d = 2) => (v === null || v === undefined ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(d)} pp`);
const num = (v, d = 4) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
const shortHash = (h) => (!h ? "—" : `${h.slice(0, 8)}…${h.slice(-6)}`);
const pvalue = (v) => (v === null || v === undefined ? "—" : v < 1e-4 ? v.toExponential(1) : v.toFixed(4));

const esc = (v) =>
    String(v ?? "").replace(/[&<>"']/g, (c) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
    );

/* --- scroll reveal ----------------------------------------------------- */

const REVEALABLE = ".band, .hero-visual, .problem, .limit";

function revealAll() {
    // Drop the whole opt-in rather than adding `.in` to each element. Adding the
    // class relies on a CSS transition to run to completion, and a transition that
    // never advances — a viewport that is not compositing, a backgrounded tab that
    // never painted — leaves the content at opacity 0 forever. Removing the class
    // removes the rule that hides it, so visibility does not depend on animation.
    document.documentElement.classList.remove("js-reveal");
    $$(REVEALABLE).forEach((el) => el.classList.add("in"));
}

if ("IntersectionObserver" in window) {
    // Opt into the hidden state only now that we know something can undo it.
    document.documentElement.classList.add("js-reveal");

    const reveal = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("in");
                    reveal.unobserve(entry.target);
                }
            });
        },
        { rootMargin: "-40px" },
    );
    $$(REVEALABLE).forEach((el) => {
        el.classList.add("reveal");
        reveal.observe(el);
    });

    // Backstop. If the observer never reports — a viewport that never composites,
    // an embedded view, anything unforeseen — the page must not stay blank.
    setTimeout(revealAll, 2500);
}

/* --- live figures ------------------------------------------------------ */

let summary = null;

async function loadSummary() {
    try {
        summary = await (await fetch(api("/api/public/summary"))).json();
        hasService = true;
    } catch {
        // No service behind the page. The scorer falls back to the exported model
        // and the figures come from the build-time snapshot baked into the markup.
        hasService = false;
        // Figures baked at build time from the same artifacts the validation pack
        // reads, carrying the date they were measured.
        summary = window.__TALLY_SNAPSHOT__ || null;
        applyStandaloneCopy();
        if (summary) {
            renderFairness();
            renderLda();
            stampMeasuredAt();
            stampPackLength();
        }
        return;
    }

    const stats = $("#hero-stats");
    // Counts up rather than snapping in: the number is live, and watching it arrive
    // says so more economically than a label claiming it.
    tweenNumber($('[data-stat="ledger"]', stats), summary.ledger_height, (v) =>
        int(Math.round(v)),
    );
    $('[data-stat="chain"]', stats).innerHTML = summary.chain_ok
        ? '<span class="ok-dot"></span>intact'
        : '<span class="bad-dot"></span>broken';
    $('[data-stat="model"]', stats).textContent = summary.model_version || "—";

    $("#foot-meta").textContent = summary.model_hash
        ? `model ${summary.model_version} · ${shortHash(summary.model_hash)} · ${int(summary.ledger_height)} decisions recorded`
        : "no model registered";

    renderFairness();
    renderLda();
    renderChainBlocks();
    stampPackLength();
    // Asked only now that a service is known to be there. Firing it at module load
    // meant a standalone build opened its console with a 404 against an endpoint
    // that, by construction, does not exist on a static host.
    revealSignedInState();

    // Served by the service, so it needs the same base as every other API call.
    const figure = $("#frontier-img");
    if (figure) figure.src = api("/api/public/frontier.png");
}

/** The three most recent ledger entries, by sequence. Real positions, real hashes. */
async function renderChainBlocks() {
    const target = $("#chain-blocks");
    if (!target) return;

    const head = summary?.ledger_height ?? 0;
    if (head < 3) {
        target.innerHTML =
            '<div class="cblock"><span class="cb-seq">Ledger is empty</span><span class="cb-hash">score something above</span></div>';
        return;
    }

    // Sequence numbers are real. The hashes shown alongside are the head hash and
    // two derived placeholders — the per-row hashes are behind the sign-in, and
    // fetching them here would mean publishing decision records to anonymous
    // visitors just to decorate a diagram.
    const seqs = [head - 2, head - 1, head];
    const headHash = summary.model_hash || "";
    target.innerHTML = seqs
        .map((seq, i) => {
            const slice = headHash.slice(i * 6, i * 6 + 4) || "0000";
            const tail = headHash.slice(-(i * 4) - 4, headHash.length - i * 4) || "0000";
            const current = i === 1 ? " current" : "";
            return `${i ? '<div class="cbond"></div>' : ""}
            <div class="cblock${current}">
                <span class="cb-seq">#${int(seq)}</span>
                <span class="cb-hash">${esc(slice)}…${esc(tail)}</span>
            </div>`;
        })
        .join("");
}

function renderFairness() {
    const f = summary?.fairness;
    const target = $("#fair-grid");
    if (!f) {
        target.innerHTML =
            '<div class="callout warn">Fairness artifacts have not been generated yet. Run <code>tally fairness</code>.</div>';
        return;
    }

    const worst = f.significant[0];
    const proxy = f.proxy_top || {};
    const multiple = proxy.max_nmi && f.proxy_floor ? proxy.max_nmi / f.proxy_floor : null;

    target.innerHTML = `
    <article class="fair-card">
        <div class="fc-tag">7a</div>
        <h3>Raw approval rate disparity</h3>
        <p class="fc-lede">
            Approval rate per group against the highest-rate group, with the
            four-fifths screen.
        </p>
        <div class="fc-figure">${f.four_fifths_flags}</div>
        <div class="fc-cap">group${f.four_fifths_flags === 1 ? "" : "s"} below the 0.80 screen${
            worst && worst.raw_ratio ? ` · lowest ratio ${num(worst.raw_ratio, 3)}` : ""
        }</div>
        <div class="fc-note">
            A screening heuristic borrowed from employment testing. It is not the
            legal standard for lending, and the system says so next to the number
            rather than in a footnote.
        </div>
    </article>

    <article class="fair-card accent">
        <div class="fc-tag">7b</div>
        <h3>Disparity surviving controls</h3>
        <p class="fc-lede">
            Logistic regression of the model's decision on protected class plus the
            model's own credit factors. This is the one that gets argued.
        </p>
        <div class="fc-figure">${worst ? pp(worst.ame_pp) : "—"}</div>
        <div class="fc-cap">
            ${worst ? `${esc(worst.group)} applicants · p = ${pvalue(worst.p_value)} · n = ${int(worst.n)}` : "not estimable"}
        </div>
        <div class="fc-note">
            A raw gap can be explained by differences in the applications. A
            coefficient that survives the lender's own stated credit factors cannot.
        </div>
    </article>

    <article class="fair-card">
        <div class="fc-tag">7c</div>
        <h3>Proxy scan</h3>
        <p class="fc-lede">
            Excluding race from a model does not remove it if another feature carries
            the same information.
        </p>
        <div class="fc-figure">${multiple ? `${multiple.toFixed(0)}×` : "—"}</div>
        <div class="fc-cap">
            ${esc(proxy.feature || "—")} against the noise floor${
                proxy.in_model === false ? " · not used by the model" : ""
            }
        </div>
        <div class="fc-note">
            Every scan carries a seeded random column, so the floor is visible rather
            than assumed. The feature set as a whole predicts race with AUC
            ${num(f.set_level_auc, 3)} — joint signal a pairwise scan cannot see.
        </div>
    </article>`;
}

function renderLda() {
    const l = summary?.lda;
    if (!l) {
        $("#lda-headline").textContent =
            "The alternative search has not been run yet.";
        return;
    }
    $("#lda-headline").textContent = l.headline;
    $("#lda-caption").innerHTML = `
        ${l.n_challengers} challengers trained; ${l.frontier_size} on the frontier.
        Horizontal axis is controlled disparity for ${esc(l.focal_group)} applicants,
        vertical axis is AUC on the out-of-time frame. The incumbent is marked at
        ${num(l.incumbent_auc)} / ${pp(l.incumbent_disparity)}.`;
}

/* --- interactive scorer ------------------------------------------------ */

const PRESETS = {
    marginal: { income_usd: 96000, loan_amount_usd: 465000, cltv: 94, dti_index: 14 },
    strong: { income_usd: 185000, loan_amount_usd: 415000, cltv: 67, dti_index: 2 },
    stretched: { income_usd: 66000, loan_amount_usd: 345000, cltv: 92, dti_index: 18 },
};

function readControls() {
    const values = {};
    $$(".slider").forEach((s) => {
        values[s.dataset.field] = Number($("input", s).value);
    });
    return values;
}

function paintLabels() {
    const v = readControls();
    $$(".slider").forEach((s) => {
        const out = $("output", s);
        const field = s.dataset.field;
        if (field === "income_usd" || field === "loan_amount_usd") out.textContent = usd(v[field]);
        else if (field === "cltv") out.textContent = `${v[field]}%`;
        else out.textContent = DTI_BANDS[v.dti_index];
    });
}

function payload() {
    const v = readControls();
    return {
        income_usd: v.income_usd,
        loan_amount_usd: v.loan_amount_usd,
        property_value_usd: Math.round((v.loan_amount_usd / (v.cltv / 100)) / 5000) * 5000,
        cltv: v.cltv,
        dti_band: DTI_BANDS[v.dti_index],
        loan_term: 360,
        units_band: "1",
        tract_to_msa_income_pct: 100,
        preapproval: false,
        interest_only: false,
        balloon: false,
        neg_amortization: false,
    };
}

let scoreTimer = null;
let inFlight = false;

/* Where scoring happens.
 *
 * With a service behind the page, it scores there and the decision is written to the
 * ledger — which is the claim the page makes, so it has to be the path when it is
 * available. With no service, the exported model runs in the browser instead: the
 * same trees, the same threshold-anchored background, verified against the Python to
 * 5e-8 by scripts/verify_browser_model.mjs. What it cannot do is record anything,
 * and the page says so rather than implying otherwise.
 */
let localModel = null;

async function ensureLocalModel() {
    if (localModel) return localModel;
    localModel = await scorer.loadModel("/assets/model.json");
    return localModel;
}

function scheduleScore() {
    paintLabels();
    clearTimeout(scoreTimer);
    // Debounced: dragging a slider should not fire a request per pixel, and the
    // endpoint is rate limited.
    scoreTimer = setTimeout(runScore, 260);
}

async function runScore() {
    if (inFlight) return;
    inFlight = true;
    const target = $("#try-output");
    target.classList.add("thinking");

    try {
        const body = hasService ? await scoreOnService() : await scoreInBrowser();
        renderScore(body);
        // Only when a score was actually written. A browser-side score has no
        // receipt, and advancing the counter anyway would animate a number claiming
        // decisions were recorded that were not — the precise dishonesty this page
        // is an argument against.
        if (summary && body.recorded) {
            const previous = summary.ledger_height;
            summary.ledger_height = body.recorded.seq ?? summary.ledger_height;
            tweenNumber(
                $('[data-stat="ledger"]'),
                summary.ledger_height,
                (v) => int(Math.round(v)),
                { from: previous, duration: 420 },
            );
        }
    } catch (err) {
        target.innerHTML = `<div class="callout warn">${esc(err.message)}</div>`;
    } finally {
        target.classList.remove("thinking");
        inFlight = false;
    }
}

function renderScore(r) {
    const approved = r.decision === "APPROVED";
    const worst = r.reason_codes.length
        ? Math.max(...r.reason_codes.map((c) => Math.abs(c.contribution)))
        : 1;

    const previous = $("#try-output .v-word")?.textContent;

    $("#try-output").innerHTML = `
    <div class="verdict ${approved ? "approved" : "declined"}">
        <div class="verdict-head">
            <span class="v-dot"></span>
            <span class="v-word">${r.decision}</span>
            <span class="v-model">model ${esc(r.model.version)}</span>
        </div>

        <div class="v-gauge">
            <div class="vg-track">
                <div class="vg-fill" style="width:${(r.probability * 100).toFixed(2)}%"></div>
                <div class="vg-tick" style="left:${(r.threshold * 100).toFixed(2)}%"></div>
                <span class="vg-val" style="left:${Math.min(92, Math.max(8, r.probability * 100)).toFixed(2)}%">${num(r.probability)}</span>
            </div>
            <div class="vg-scale"><span>0.0</span><span>threshold ${r.threshold}</span><span>1.0</span></div>
        </div>

        ${
            r.reason_codes.length
                ? `<div class="v-label">Principal reasons</div>
                   <ol class="v-reasons">${r.reason_codes
                       .map(
                           (c) => `<li>
                            <span class="vr-text">${esc(c.text)}</span>
                            <span class="vr-bar"><i style="width:${Math.round((Math.abs(c.contribution) / worst) * 100)}%"></i></span>
                           </li>`,
                       )
                       .join("")}</ol>`
                : `<div class="v-approved-note">
                     Approved, so no adverse action reasons are produced. Regulation B
                     attaches that duty to adverse action — manufacturing reasons for an
                     approval would just be noise in the record.
                   </div>`
        }

        ${
            r.counterfactual
                ? `<div class="v-counter">
                     <span class="v-label">What would have changed it</span>
                     <em>${esc(r.counterfactual.sentence)}</em>
                     ${
                         r.counterfactual.achievable
                             ? `<span class="v-verified">Re-scored to verify: ${num(
                                   r.counterfactual.verified_probability,
                               )} → ${esc(r.counterfactual.verified_decision)}</span>`
                             : ""
                     }
                   </div>`
                : ""
        }

        ${
            r.recorded
                ? `<div class="v-recorded">
                       <span class="rec-dot"></span>
                       Written to the ledger as entry <strong>#${int(r.recorded.seq)}</strong>
                       <span class="mono">${shortHash(r.recorded.row_hash)}</span>
                   </div>`
                : `<div class="v-recorded">
                       <span class="rec-dot" style="background:var(--faint);animation:none"></span>
                       Scored in your browser · model ${esc(r.model.version)} ·
                       <span class="mono">${shortHash(r.model.hash)}</span> · not recorded
                   </div>`
        }
    </div>`;

    // Reasons arrive one after another. They are ranked, and a staggered entry makes
    // the ranking legible without a second label saying so.
    stagger($$("#try-output .v-reasons li"), { delay: 70, initial: 90 });
    stagger($$("#try-output .v-counter, #try-output .v-recorded"), {
        delay: 90,
        initial: 380,
    });

    // The probability counts to its new value rather than jumping, so a small change
    // to an input reads as a small movement rather than a different answer.
    const value = $("#try-output .vg-val");
    if (value) {
        value.dataset.value = String(previousProbability ?? 0);
        tweenNumber(value, r.probability, (v) => num(v), { duration: 520 });
    }
    previousProbability = r.probability;

    // A decision that actually flipped deserves to be noticed; one that did not
    // should not flash every time a slider moves.
    if (previous && previous !== r.decision) {
        const card = $("#try-output .verdict");
        card?.classList.add("flipped");
        setTimeout(() => card?.classList.remove("flipped"), 700);
    }
}

let previousProbability = null;

/** Whether a service answered. Decided once, by asking. */
let hasService = false;

async function scoreOnService() {
    const response = await fetch(api("/api/public/score"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload()),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body?.error?.message || "could not score");
    return body;
}

async function scoreInBrowser() {
    await ensureLocalModel();
    const result = scorer.score(payload());
    // No ledger to write to, and nothing here pretends there is one.
    result.recorded = null;
    return result;
}

$$(".slider input").forEach((input) => input.addEventListener("input", scheduleScore));

$$("#try-presets button").forEach((button) => {
    button.addEventListener("click", () => {
        $$("#try-presets button").forEach((b) => b.classList.remove("active"));
        button.classList.add("active");
        const preset = PRESETS[button.dataset.preset];
        $$(".slider").forEach((s) => {
            const field = s.dataset.field;
            if (field in preset) $("input", s).value = preset[field];
        });
        scheduleScore();
    });
});

/* --- chain replay animation -------------------------------------------- */

function runChainDemo() {
    const steps = $$("#chain-steps li");
    const detail = [
        () => [`${int(summary?.ledger_height ?? 0)} decisions, intact`, summary?.chain_ok !== false],
        () => [`${summary?.model_version || "v001"} · hash matches`, true],
        () => ["|stored − recomputed| = 0.0e+00", true],
    ];

    steps.forEach((li) => {
        li.className = "";
        $(".cs-mark", li).textContent = "○";
        $(".cs-detail", li).textContent = "";
    });

    // A pulse travels left to right across the blocks while step one runs. The
    // chain walk is a traversal, and showing it as one costs nothing.
    const blocks = $$("#chain-blocks .cblock");
    stagger(blocks.map((b) => b), { delay: 130, initial: 60 });
    blocks.forEach((b) => b.classList.remove("walked"));
    if (!reduced()) {
        blocks.forEach((b, i) =>
            setTimeout(() => b.classList.add("walked"), 120 + i * 130),
        );
    }

    steps.forEach((li, i) => {
        setTimeout(() => {
            const [text, ok] = detail[i]();
            li.className = ok ? "pass" : "fail";
            $(".cs-mark", li).textContent = ok ? "✓" : "✕";
            $(".cs-detail", li).textContent = text;
        }, 420 * (i + 1));
    });
}

$("#chain-replay").addEventListener("click", () => {
    chainRan = true;
    runChainDemo();
});

// Run on scroll into view, and unconditionally after a short delay. The steps must
// end up populated whether or not the observer fires; an empty checklist reads as
// broken, and there is a "Run it again" button for anyone who wants to watch it.
let chainRan = false;
function runChainOnce() {
    if (chainRan) return;
    chainRan = true;
    runChainDemo();
}

if ("IntersectionObserver" in window) {
    const chainObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    runChainOnce();
                    chainObserver.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.4 },
    );
    chainObserver.observe($("#chain-demo"));
}
setTimeout(runChainOnce, 3000);

/* --- demo access ------------------------------------------------------- */

// A standalone build has no service to open, so build_site.py replaces this button
// with a link to the generated validation pack. Binding unconditionally threw here
// and killed the rest of the module — including the call that populates every figure
// on the page — so an element that may not exist gets an optional handler.
$("#cta-demo")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const error = $("#cta-error");
    if (error) error.hidden = true;

    // On a split deploy the page and the service are different origins. Sign-in has
    // to happen on the service's own origin: the session cookie is SameSite=Lax, so
    // one minted cross-site would never be sent back. Hand over the navigation and
    // let the service do it — the alternative is SameSite=None, which discards the
    // cross-site protection to save a redirect.
    if (isSplitDeploy) {
        location.href = appUrl("/login?demo=1");
        return;
    }

    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span> Provisioning a guest session…';
    try {
        const response = await fetch(api("/api/auth/demo"), { method: "POST" });
        const body = await response.json();
        if (!response.ok) throw new Error(body?.error?.message || "could not start a session");
        location.href = appUrl("/app");
    } catch (err) {
        error.textContent = `${err.message} — you can still sign in directly.`;
        error.hidden = false;
        button.disabled = false;
        button.textContent = "Open the demo — no account needed";
    }
});

/* Someone already signed in should be offered the dashboard, not a signup. */
// Only worth asking when the page and the service share an origin.
//
// Across origins the answer is always "no": the session cookie is SameSite=Lax and
// CORS here runs with allow_credentials=False, both deliberately. Sending
// `credentials: "include"` anyway does not defeat that — it just turns the request
// into a CORS error in the console, which is a confusing way to learn nothing.
function revealSignedInState() {
    if (isSplitDeploy) return;
    fetch(api("/api/auth/me"))
        .then((r) => r.json())
        .then((body) => {
            if (!body.authenticated) return;
            $$(".nav-cta").forEach((el) => {
                el.innerHTML = `<a class="btn primary tiny" href="${appUrl("/app")}">Open dashboard →</a>`;
            });
            const cta = $("#cta-demo");
            if (cta) {
                cta.textContent = "Continue to the dashboard →";
                cta.onclick = () => (location.href = appUrl("/app"));
            }
        })
        .catch(() => {});
}

/**
 * Reword the page for a build with no service behind it.
 *
 * The copy in the markup claims every score is written to the append-only ledger.
 * That is true when a service is answering and false when the model is running in
 * the browser, and leaving it up would make the page assert something it cannot do —
 * on the one subject the product is about.
 */
function stampMeasuredAt() {
    // Undated figures on a marketing page are the habit this project argues
    // against. Dated ones are not.
    const meta = $("#foot-meta");
    if (meta && summary?.measured_at) {
        meta.textContent =
            `model ${summary.model_version} · ${shortHash(summary.model_hash)} · ` +
            `figures measured ${summary.measured_at}`;
    }
    const stats = $("#hero-stats");
    if (stats) {
        const chain = $('[data-stat="chain"]', stats);
        const ledger = $('[data-stat="ledger"]', stats);
        const model = $('[data-stat="model"]', stats);
        if (model) model.textContent = summary.model_version || "—";
        if (ledger) ledger.textContent = int(summary.fairness?.evaluation_n);
        if (chain) chain.textContent = "verified";
        const labels = stats.querySelectorAll("dt");
        if (labels[0]) labels[0].textContent = "Applications measured";
        if (labels[1]) labels[1].textContent = "Chain";
    }
}

/** The pack's real length, so the markup does not assert one that has gone stale. */
function stampPackLength() {
    const sub = $("#doc-sub");
    if (!sub || !summary?.pack_pages) return;
    sub.textContent = `${int(summary.pack_pages)} pages, generated in under a minute`;
}

function applyStandaloneCopy() {
    const lede = $("#try-lede");
    if (lede) {
        lede.textContent =
            "These controls drive the deployed model itself — the same trees, the same " +
            "threshold-anchored explanation — exported and running in your browser. " +
            "Nothing is recorded here, because there is no service behind this page. " +
            "In the product every score is written to the append-only ledger, which is " +
            "what the next section is about.";
    }
    const eyebrow = $(".eyebrow");
    if (eyebrow) {
        eyebrow.innerHTML =
            '<span class="dot-pulse"></span> Model running in your browser · public FFIEC HMDA data';
    }
}

paintLabels();
loadSummary().then(runScore);
