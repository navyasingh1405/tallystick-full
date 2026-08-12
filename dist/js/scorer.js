/* The model, running in the browser.
 *
 * The marketing site has no backend, so the landing page scores against the real
 * exported ensemble rather than a lookalike. Everything here is checked against the
 * Python implementation by scripts/verify_browser_model.mjs, which scores a grid of
 * applicants in both and fails on any disagreement beyond 1e-9. A "roughly right"
 * reimplementation would be worse than a static screenshot: it would look live while
 * quietly disagreeing with the system it claims to be.
 *
 * The interesting part is the SHAP implementation. Reason codes are measured against
 * the decision threshold, which requires *interventional* attribution over an
 * explicit background — the same construction the service uses, and the reason a
 * naive `pred_contrib` call would not do.
 */

let MODEL = null;

export async function loadModel(url = "/assets/model.json") {
    if (MODEL) return MODEL;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`could not load the model (${response.status})`);
    MODEL = await response.json();
    MODEL.backgroundRows = toRows(MODEL.background, MODEL.features);
    MODEL.treeFeatures = MODEL.trees.map(collectFeatures);
    return MODEL;
}

export const model = () => MODEL;

function toRows(columns, features) {
    const n = columns[features[0]].length;
    const rows = [];
    for (let i = 0; i < n; i += 1) {
        rows.push(features.map((f) => columns[f][i]));
    }
    return rows;
}

/** Which features a tree actually splits on. Used to keep the SHAP cost down. */
function collectFeatures(tree) {
    const seen = new Set();
    for (const node of tree.nodes) {
        if (node.f >= 0) seen.add(node.f);
    }
    return [...seen].sort((a, b) => a - b);
}

/* --- feature construction ---------------------------------------------- */

/**
 * Mirror of features.feature_matrix. Any divergence here is invisible in the output
 * and fatal to the claim that this is the same model, so it is covered by the
 * verification grid rather than by care alone.
 */
export function buildVector(applicant) {
    const m = MODEL;
    const p = m.fit_params;
    const cap = p.income_cap > 0 ? p.income_cap : Infinity;
    const bands = m.dti_bands;

    const raw = {
        log_income: Math.log1p(Math.min(applicant.income_usd, cap)),
        log_loan_amount: Math.log1p(applicant.loan_amount_usd),
        cltv: clamp(applicant.cltv, 0, 200),
        dti_band_ord: bands.order.indexOf(String(applicant.dti_band).trim()),
        loan_term: clamp(applicant.loan_term ?? 360, 60, 480),
        interest_only: applicant.interest_only ? 1 : 0,
        balloon: applicant.balloon ? 1 : 0,
        neg_amortization: applicant.neg_amortization ? 1 : 0,
        tract_to_msa_income_pct: clamp(applicant.tract_to_msa_income_pct ?? NaN, 0, 300),
        units_ord: bands.units_order.indexOf(String(applicant.units_band ?? "1").trim()),
        preapproval: applicant.preapproval ? 1 : 0,
    };

    // An unrecognised band is missing, not zero. indexOf returns -1, which would
    // otherwise be scored as a real value below the lowest band.
    if (raw.dti_band_ord < 0) raw.dti_band_ord = NaN;
    if (raw.units_ord < 0) raw.units_ord = NaN;

    return m.features.map((name) => {
        const value = raw[name];
        if (Number.isFinite(value)) return value;
        // Every feature is imputed at its training median before the model sees it:
        // LightGBM does not honour monotone constraints across a missing-value split.
        const median = p.feature_medians?.[name];
        return Number.isFinite(median) ? median : 0;
    });
}

const clamp = (v, lo, hi) => (Number.isFinite(v) ? Math.min(Math.max(v, lo), hi) : NaN);

/* --- prediction --------------------------------------------------------- */

function leafValue(tree, x) {
    let i = 0;
    for (;;) {
        const node = tree.nodes[i];
        if (node.f < 0) return node.v;
        const value = x[node.f];
        // LightGBM splits with `<=`, and routes a missing value by `default_left`.
        const goLeft = Number.isNaN(value) ? node.d : value <= node.t;
        i = goLeft ? node.l : node.r;
    }
}

/** Raw margin, in log-odds. All attribution happens in this space. */
export function rawScore(x) {
    let total = 0;
    for (const tree of MODEL.trees) total += leafValue(tree, x);
    return total;
}

export const sigmoid = (z) => 1 / (1 + Math.exp(-z));
export const probability = (x) => sigmoid(rawScore(x));
export const decide = (p) => (p > MODEL.threshold ? "APPROVED" : "DECLINED");

/* --- interventional SHAP ------------------------------------------------ */
/*
 * SHAP(x, R) is the mean over background rows r of the single-reference Shapley
 * values SHAP(x, r) — interventional attribution is linear in the background, which
 * is what makes this tractable at all.
 *
 * For one reference and one tree, the exact Shapley values are computed by brute
 * force over coalitions. That sounds ruinous, and would be over all eleven features
 * (2^11 per tree per reference). But a tree only depends on the features it splits
 * on — typically four or five — and a Shapley value of a game that ignores a feature
 * is zero for that feature and unchanged for the others. So each tree enumerates
 * 2^k coalitions over its own k features, not 2^11 over all of them.
 *
 * Brute force over the definition is deliberate: it is four lines of arithmetic that
 * can be read against the formula, where a fast path-based TreeSHAP is a hundred
 * lines whose correctness rests on an argument. The verification script checks this
 * against the Python either way.
 */

const FACTORIAL = Array.from({ length: 20 }, (_, i) => (i <= 1 ? 1 : NaN));
for (let i = 2; i < 20; i += 1) FACTORIAL[i] = FACTORIAL[i - 1] * i;

/** Shapley coalition weight |S|!(d-|S|-1)!/d! */
function weight(sizeS, d) {
    return (FACTORIAL[sizeS] * FACTORIAL[d - sizeS - 1]) / FACTORIAL[d];
}

export function shapValues(x, background = MODEL.backgroundRows) {
    const d = MODEL.features.length;
    const phi = new Float64Array(d);

    for (const tree of MODEL.trees) {
        const used = MODEL.treeFeatures[tree === undefined ? 0 : MODEL.trees.indexOf(tree)];
        accumulateTree(tree, used, x, background, phi);
    }
    return Array.from(phi, (v) => v / background.length);
}

function accumulateTree(tree, used, x, background, phi) {
    const k = used.length;
    if (k === 0) return; // a constant tree attributes nothing to anybody
    const subsets = 1 << k;

    const hybrid = new Float64Array(x.length);
    const values = new Float64Array(subsets);

    for (const r of background) {
        // Evaluate the tree for every coalition of its own features.
        for (let mask = 0; mask < subsets; mask += 1) {
            for (let j = 0; j < x.length; j += 1) hybrid[j] = r[j];
            for (let b = 0; b < k; b += 1) {
                if (mask & (1 << b)) hybrid[used[b]] = x[used[b]];
            }
            values[mask] = leafValue(tree, hybrid);
        }

        // phi_i = sum over S not containing i of w(|S|) * [v(S + i) - v(S)]
        for (let b = 0; b < k; b += 1) {
            const bit = 1 << b;
            let total = 0;
            for (let mask = 0; mask < subsets; mask += 1) {
                if (mask & bit) continue;
                total += weight(popcount(mask), k) * (values[mask | bit] - values[mask]);
            }
            phi[used[b]] += total;
        }
    }
}

function popcount(v) {
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    return (((v + (v >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

/** E_background[f], which the threshold-anchored background places at the cutoff. */
export function baseValue(background = MODEL.backgroundRows) {
    let total = 0;
    for (const r of background) total += rawScore(r);
    return total / background.length;
}

/* --- reason codes ------------------------------------------------------- */

export function reasonCodes(phi, decision, limit = 4) {
    if (decision === "APPROVED") return [];
    const table = MODEL.reason_codes;

    return MODEL.features
        .map((name, i) => ({ feature: name, contribution: phi[i] }))
        .filter((entry) => entry.contribution < 0)
        .sort((a, b) => a.contribution - b.contribution)
        .slice(0, limit)
        .map((entry, index) => {
            const code = table.feature_to_code[entry.feature];
            const text = table.text[code];
            if (!text) throw new Error(`no plain-language text for ${code}`);
            return {
                rank: index + 1,
                code,
                feature: entry.feature,
                contribution: entry.contribution,
                text: text.applicant,
                factor: text.factor,
            };
        });
}

/* --- counterfactual ----------------------------------------------------- */

const SOURCE = {
    log_income: "income_usd",
    log_loan_amount: "loan_amount_usd",
    cltv: "cltv",
    dti_band_ord: "dti_band",
    loan_term: "loan_term",
    tract_to_msa_income_pct: "tract_to_msa_income_pct",
    units_ord: "units_band",
    interest_only: "interest_only",
    balloon: "balloon",
    neg_amortization: "neg_amortization",
    preapproval: "preapproval",
};

const UNIT = {
    income_usd: 1000,
    loan_amount_usd: 5000,
    cltv: 0.5,
    tract_to_msa_income_pct: 1,
    loan_term: 12,
};

const approves = (applicant, field, value) =>
    probability(buildVector({ ...applicant, [field]: value })) > MODEL.threshold;

/**
 * The value of one factor at which this application would have been approved.
 *
 * Verified by re-scoring before it is returned. A counterfactual that does not
 * actually flip the decision is worse than none at all, because someone may act on
 * it — so the number quoted is always one that has been checked.
 */
export function counterfactual(applicant, feature) {
    const field = SOURCE[feature];
    const factor = MODEL.reason_codes.text[MODEL.reason_codes.feature_to_code[feature]].factor;
    const direction = MODEL.monotone[MODEL.features.indexOf(feature)];
    const ranges = MODEL.fit_params.raw_ranges[feature] || [0, 1];

    if (field === "dti_band" || field === "units_band") {
        const order = field === "dti_band" ? MODEL.dti_bands.order : MODEL.dti_bands.units_order;
        const current = order.indexOf(String(applicant[field]));
        for (let i = current - 1; i >= 0; i -= 1) {
            if (approves(applicant, field, order[i])) {
                return done(applicant, field, order[i], factor, direction, MODEL.dti_bands.label[order[i]] || order[i]);
            }
        }
        return unachievable(factor);
    }

    if (["interest_only", "balloon", "neg_amortization", "preapproval"].includes(field)) {
        const flipped = !applicant[field];
        if (approves(applicant, field, flipped)) {
            return done(applicant, field, flipped, factor, direction, flipped ? "present" : "removed");
        }
        return unachievable(factor);
    }

    const [lo, hi] = ranges;
    if (direction === 0) return unachievable(factor);
    const endpoint = direction === 1 ? hi : lo;
    if (!approves(applicant, field, endpoint)) return unachievable(factor);

    let a = direction === 1 ? applicant[field] : endpoint;
    let b = direction === 1 ? endpoint : applicant[field];
    const unit = UNIT[field] ?? 1;
    for (let i = 0; i < 40 && Math.abs(b - a) > unit / 100; i += 1) {
        const mid = (a + b) / 2;
        if (approves(applicant, field, mid)) {
            if (direction === 1) b = mid;
            else a = mid;
        } else if (direction === 1) a = mid;
        else b = mid;
    }

    // Round to a figure worth quoting, then step until it verifies. Rounding toward
    // the applicant's current position can push the value back over the line.
    let candidate =
        (direction === 1 ? Math.ceil((direction === 1 ? b : a) / unit) : Math.floor(a / unit)) * unit;
    for (let i = 0; i < 8; i += 1) {
        if (candidate >= lo && candidate <= hi && approves(applicant, field, candidate)) {
            return done(applicant, field, candidate, factor, direction, formatValue(field, candidate));
        }
        candidate += direction === 1 ? unit : -unit;
    }
    return unachievable(factor);
}

function done(applicant, field, value, factor, direction, display) {
    const p = probability(buildVector({ ...applicant, [field]: value }));
    const phrase =
        typeof value === "boolean"
            ? `${value ? "Adding" : "Removing"} the ${factor} would have produced an approval`
            : direction === 1
              ? `An ${factor} at or above ${display} would have produced an approval`
              : `A ${factor} at or below ${display} would have produced an approval`;
    return {
        achievable: true,
        flip_display: display,
        verified_probability: p,
        verified_decision: decide(p),
        sentence: `${phrase}, holding all other factors constant.`,
    };
}

function unachievable(factor) {
    return {
        achievable: false,
        sentence:
            `Even at the most favourable value observed in training, changing the ${factor} ` +
            "alone would not have changed this decision.",
    };
}

function formatValue(field, v) {
    if (field === "income_usd" || field === "loan_amount_usd") {
        return `$${Math.round(v).toLocaleString("en-US")}`;
    }
    if (field === "cltv" || field === "tract_to_msa_income_pct") {
        return Math.abs(v - Math.round(v)) < 0.05 ? `${Math.round(v)}%` : `${v.toFixed(1)}%`;
    }
    if (field === "loan_term") return `${Math.round(v)} months`;
    return String(v);
}

/* --- one call ----------------------------------------------------------- */

export function score(applicant) {
    const x = buildVector(applicant);
    const raw = rawScore(x);
    const p = sigmoid(raw);
    const decision = decide(p);
    const phi = shapValues(x);
    const codes = reasonCodes(phi, decision);

    return {
        decision,
        probability: p,
        raw_score: raw,
        threshold: MODEL.threshold,
        margin: raw - MODEL.raw_threshold,
        model: { version: MODEL.version, hash: MODEL.model_hash },
        reason_codes: codes,
        contributions: Object.fromEntries(MODEL.features.map((f, i) => [f, phi[i]])),
        counterfactual: codes.length ? counterfactual(applicant, codes[0].feature) : null,
    };
}
