/* Minimal templating.
 *
 * No virtual DOM. The panes are small enough that replacing innerHTML on a container
 * is both simpler and faster than diffing, provided two rules are followed:
 *
 *   1. Every listener is delegated at the pane root, so re-rendering cannot leak
 *      handlers or leave dead ones behind.
 *   2. Anything holding focus is never re-rendered. The applicant form is built once
 *      on mount; only the results region below it updates.
 */

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value).replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

/** Tagged template that escapes interpolations. Use raw() to opt out deliberately. */
export function html(strings, ...values) {
    return strings.reduce((out, chunk, i) => {
        if (i === 0) return chunk;
        const value = values[i - 1];
        let rendered;
        if (value === null || value === undefined || value === false) {
            rendered = "";
        } else if (Array.isArray(value)) {
            rendered = value.join("");
        } else if (value && value.__raw) {
            rendered = value.value;
        } else {
            rendered = escapeHtml(value);
        }
        return out + rendered + chunk;
    }, "");
}

/** Mark a string as already-safe HTML. */
export const raw = (value) => ({ __raw: true, value: value ?? "" });

export function mount(element, markup) {
    element.innerHTML = markup;
    return element;
}

/**
 * Attach one delegated click handler to a root.
 * Returns an unsubscribe function so panes can clean up on unmount.
 */
export function onAction(root, handlers) {
    const listener = (event) => {
        const target = event.target.closest("[data-action]");
        if (!target || !root.contains(target)) return;
        const handler = handlers[target.dataset.action];
        if (!handler) return;
        event.preventDefault();
        handler(target, event);
    };
    root.addEventListener("click", listener);
    return () => root.removeEventListener("click", listener);
}

export function onSubmit(root, handler) {
    const listener = (event) => {
        event.preventDefault();
        handler(event.target, event);
    };
    root.addEventListener("submit", listener);
    return () => root.removeEventListener("submit", listener);
}

export function skeleton(lines = 3) {
    return html`<div class="skeleton-block">
        ${raw(Array.from({ length: lines }, () => '<div class="skeleton"></div>').join(""))}
    </div>`;
}

export function empty(message) {
    return html`<div class="empty">${message}</div>`;
}

/** Count a number up to its final value. Motion on state change only. */
export function animateValue(element, to, format, duration = 400) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        element.textContent = format(to);
        return;
    }
    const from = 0;
    const start = performance.now();
    const tick = (now) => {
        const t = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - t, 3);
        element.textContent = format(from + (to - from) * eased);
        if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}
