/* The recorded walkthrough.
 *
 * Renders panes captured from a live instance by scripts/capture_walkthrough.py.
 * Nothing here computes anything: the markup arrives finished, and this module's
 * only jobs are to put it on the page, to move between panes, and to make sure a
 * visitor is never in doubt about which of the two they are looking at.
 */

const $ = (s, root = document) => root.querySelector(s);

const OUTLET = $("#outlet");

/** Everything a visitor could click that would silently do nothing. */
const INTERACTIVE = "input, select, textarea, button, a[href]";

let data = null;
let active = null;

async function load() {
    const response = await fetch("/assets/walkthrough.json");
    if (!response.ok) throw new Error(`walkthrough.json: ${response.status}`);
    data = await response.json();

    $("#wt-date").textContent = data.captured_at || "recently";

    // The rail is captured too, so the model hash and chain state shown beside the
    // panes are the ones that were actually in force when the panes were recorded.
    // Rebuilding it from a template would let the two drift apart.
    $("#account").innerHTML = data.rail.account;
    $("#status").innerHTML = data.rail.status;
    $("#account").removeAttribute("hidden");
    neutralise($("#account"));

    renderNav();
    window.addEventListener("hashchange", show);
    show();
}

function renderNav() {
    $("#nav").innerHTML = data.panes
        .map(
            (p) => `
        <a class="nav-item" href="#${p.path}" data-path="${p.path}">
            <span>${p.label}</span>
            <span class="key">g ${p.key}</span>
        </a>`,
        )
        .join("");
}

/** Strip the ability to interact without stripping the look of being interactive.
 *
 * The captured markup is full of buttons and inputs. Removing them would make the
 * walkthrough a poor likeness of the product; leaving them live would make it a
 * page where clicking "Score application" appears to fail. Disabling in place keeps
 * the likeness and removes the trap.
 */
function neutralise(root) {
    root.querySelectorAll(INTERACTIVE).forEach((el) => {
        if (el.tagName === "A") {
            // Links to files this build actually ships are left alone. build_site.py
            // repoints the frontier plot and the validation pack at their shipped
            // copies, and those two are the artifacts a visitor is most likely to
            // want open in front of them — disabling a link that works would be
            // caution at the reader's expense.
            if (el.getAttribute("href")?.startsWith("/assets/")) return;
            el.removeAttribute("href");
            el.setAttribute("role", "presentation");
        } else {
            el.disabled = true;
        }
        el.setAttribute("tabindex", "-1");
        el.classList.add("wt-inert");
    });
    // Nothing inside is focusable or reachable by a screen reader as a control,
    // which is the honest description of a screenshot rendered as markup.
    root.setAttribute("aria-hidden", "false");
}

function show() {
    const path = location.hash.replace(/^#/, "") || data.panes[0].path;
    const pane = data.panes.find((p) => p.path === path) || data.panes[0];
    if (active === pane.path) return;
    active = pane.path;

    document.querySelectorAll(".nav-item").forEach((el) => {
        el.classList.toggle("active", el.dataset.path === pane.path);
    });

    OUTLET.innerHTML = `
        <div class="wt-note">
            <div class="wt-note-tag">What this pane is for</div>
            <h2>${pane.title}</h2>
            <p>${pane.note}</p>
        </div>
        <div class="wt-capture" id="wt-capture"></div>`;

    const host = $("#wt-capture");
    host.innerHTML = pane.html;
    neutralise(host);

    OUTLET.scrollTop = 0;
    window.scrollTo({ top: 0, behavior: "instant" });
}

load().catch((error) => {
    // A walkthrough that fails should say so rather than sit on a spinner, because
    // the visitor has no way to tell a slow load from a broken one.
    OUTLET.innerHTML = `
        <div class="wt-note">
            <h2>The walkthrough could not be loaded</h2>
            <p>${String(error.message || error)}</p>
            <p><a href="/">Back to the overview</a></p>
        </div>`;
});
