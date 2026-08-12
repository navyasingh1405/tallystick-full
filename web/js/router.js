/* Hash router.
 *
 * Each pane exports mount/unmount. The router unmounts the old one before mounting
 * the new one, and every pane's listeners are attached to its own root, so unmount is
 * a single innerHTML clear with nothing left behind.
 */

import * as batch from "./panes/batch.js";
import * as fairness from "./panes/fairness.js";
import * as ledger from "./panes/ledger.js";
import * as pack from "./panes/pack.js";
import * as score from "./panes/score.js";

export const ROUTES = [
    { path: "/score", label: "Score", key: "s", pane: score },
    { path: "/ledger", label: "Ledger", key: "l", pane: ledger },
    { path: "/fairness", label: "Fairness", key: "f", pane: fairness },
    { path: "/batch", label: "Portfolio", key: "b", pane: batch },
    { path: "/pack", label: "Validation pack", key: "p", pane: pack },
];

const DEFAULT = "/score";
let active = null;

function current() {
    const path = location.hash.replace(/^#/, "") || DEFAULT;
    return ROUTES.find((r) => r.path === path) || ROUTES[0];
}

export function startRouter(outlet, onChange) {
    const render = () => {
        const route = current();
        if (active) {
            active.pane.unmountPane();
            outlet.innerHTML = "";
        }
        active = route;
        route.pane.mountPane(outlet);
        outlet.scrollTop = 0;
        window.scrollTo({ top: 0 });
        onChange(route);
    };

    window.addEventListener("hashchange", render);
    render();
}

/* `g` then a pane key jumps panes; `/` focuses the first input. Chords are cheap to
   implement and make a demo look practised. */
export function bindKeyboard() {
    let pending = false;
    let timer = null;

    window.addEventListener("keydown", (event) => {
        const tag = document.activeElement?.tagName;
        const typing = tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";

        if (event.key === "Escape" && typing) {
            document.activeElement.blur();
            return;
        }
        if (typing) return;

        if (event.key === "/") {
            event.preventDefault();
            document.querySelector("main input, main select")?.focus();
            return;
        }
        if (event.key === "g") {
            pending = true;
            clearTimeout(timer);
            timer = setTimeout(() => (pending = false), 1200);
            return;
        }
        if (pending) {
            const route = ROUTES.find((r) => r.key === event.key);
            pending = false;
            if (route) {
                event.preventDefault();
                location.hash = `#${route.path}`;
            }
        }
    });
}
