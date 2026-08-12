/* A store in thirty lines. No framework, no dependency, no build step. */

export function createStore(initial = {}) {
    let state = initial;
    const subscribers = new Set();

    return {
        get: () => state,
        set(patch) {
            state = { ...state, ...patch };
            subscribers.forEach((fn) => fn(state));
        },
        subscribe(fn) {
            subscribers.add(fn);
            fn(state);
            return () => subscribers.delete(fn);
        },
    };
}

export const store = createStore({
    health: null,
    model: null,
    lastDecision: null,
    lookupId: "",
});

/* --- toasts ------------------------------------------------------------ */

let toastRoot = null;

export function toast(message, kind = "") {
    if (!toastRoot) {
        toastRoot = document.createElement("div");
        toastRoot.className = "toasts";
        document.body.appendChild(toastRoot);
    }
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.textContent = message;
    toastRoot.appendChild(el);
    setTimeout(() => el.remove(), 4200);
}

export async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        toast("Copied to clipboard");
    } catch {
        toast("Could not copy", "bad");
    }
}
