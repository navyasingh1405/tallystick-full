/* Formatting. Every number the user reads passes through here, so precision and
   units are decided once rather than at each call site. */

export const pct = (v, digits = 1) =>
    v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const pp = (v, digits = 2) =>
    v === null || v === undefined || Number.isNaN(v)
        ? "—"
        : `${v >= 0 ? "+" : ""}${v.toFixed(digits)} pp`;

export const num = (v, digits = 4) =>
    v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(digits);

export const int = (v) =>
    v === null || v === undefined ? "—" : Number(v).toLocaleString("en-US");

export const usd = (v) =>
    v === null || v === undefined
        ? "—"
        : Number(v).toLocaleString("en-US", {
              style: "currency",
              currency: "USD",
              maximumFractionDigits: 0,
          });

/** p-values below 1e-4 are unreadable in fixed notation. */
export const pvalue = (v) => {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    if (v < 1e-4) return v.toExponential(1);
    return v.toFixed(4);
};

/** Scientific notation that shows an exact zero as exactly zero. */
export const sci = (v) => {
    if (v === null || v === undefined) return "—";
    if (v === 0) return "0.0e+00";
    return Number(v).toExponential(1);
};

/** Middle-truncate a hash so both ends stay identifiable. */
export const shortHash = (h, head = 8, tail = 6) =>
    !h ? "—" : h.length <= head + tail + 1 ? h : `${h.slice(0, head)}…${h.slice(-tail)}`;

export const timestamp = (iso) => {
    if (!iso) return "—";
    const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString("en-GB", {
        year: "numeric",
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });
};

export const titleCase = (s) =>
    String(s || "")
        .replace(/[_-]+/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase());

/** Feature names are internal; never show a raw one where a person will read it. */
const FEATURE_LABELS = {
    log_income: "Annual income",
    log_loan_amount: "Loan amount",
    cltv: "Loan-to-value ratio",
    dti_band_ord: "Debt-to-income ratio",
    loan_term: "Loan term",
    interest_only: "Interest-only feature",
    balloon: "Balloon payment",
    neg_amortization: "Negative amortization",
    tract_to_msa_income_pct: "Tract-to-area income",
    units_ord: "Number of units",
    preapproval: "Preapproval requested",
    tract_minority_pct: "Tract minority population %",
    _random_negative_control: "Random negative control",
};

export const featureLabel = (name) => FEATURE_LABELS[name] || titleCase(name);
