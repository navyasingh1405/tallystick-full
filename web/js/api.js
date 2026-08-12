/* API client. One error shape, one place that knows about it. */

export class ApiError extends Error {
    constructor(code, message, details, status) {
        super(message);
        this.code = code;
        this.details = details || {};
        this.status = status;
    }
}

async function request(path, options = {}) {
    const response = await fetch(path, {
        headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
        ...options,
    });

    const text = await response.text();
    let payload = null;
    if (text) {
        try {
            payload = JSON.parse(text);
        } catch {
            payload = { raw: text };
        }
    }

    if (!response.ok) {
        const err = payload && payload.error;
        throw new ApiError(
            err?.code || "http_error",
            err?.message || `${response.status} ${response.statusText}`,
            err?.details,
            response.status,
        );
    }
    return payload;
}

export const api = {
    health: () => request("/api/health"),
    model: () => request("/api/model/current"),

    me: () => request("/api/auth/me"),
    logout: () => request("/api/auth/logout", { method: "POST" }),

    score: (applicant) =>
        request("/api/score", { method: "POST", body: JSON.stringify(applicant) }),

    decision: (id) => request(`/api/decisions/${encodeURIComponent(id)}`),
    verify: (id) => request(`/api/decisions/${encodeURIComponent(id)}/verify`, { method: "POST" }),
    override: (id, body) =>
        request(`/api/decisions/${encodeURIComponent(id)}/override`, {
            method: "POST",
            body: JSON.stringify(body),
        }),
    adverseActionUrl: (id) => `/api/decisions/${encodeURIComponent(id)}/adverse-action.pdf`,

    recent: (limit = 25) => request(`/api/ledger/recent?limit=${limit}`),
    verifyLedger: () => request("/api/ledger/verify"),

    fairness: () => request("/api/fairness"),
    drift: () => request("/api/drift"),
    frontier: () => request("/api/lda/frontier"),
    adopt: (challengerId, holdApprovalRate = true) =>
        request("/api/lda/adopt", {
            method: "POST",
            body: JSON.stringify({
                challenger_id: challengerId,
                hold_approval_rate: holdApprovalRate,
            }),
        }),

    submitBatch: (formData) => request("/api/batch", { method: "POST", body: formData }),
    sampleBatch: (n) => request(`/api/batch?sample=${n}`, { method: "POST" }),
    batchStatus: (jobId) => request(`/api/batch/${jobId}`),
    batchResultsUrl: (batchId) => `/api/batches/${batchId}/results.csv`,

    buildPack: () => request("/api/validation-pack", { method: "POST" }),
    packStatus: (jobId) => request(`/api/validation-pack/${jobId}`),
};

/** Poll a job until it leaves the running state. */
export async function pollJob(fetchStatus, jobId, onProgress, intervalMs = 500) {
    for (;;) {
        const job = await fetchStatus(jobId);
        if (onProgress) onProgress(job);
        if (job.status !== "running") return job;
        await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
}
