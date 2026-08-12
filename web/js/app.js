import { api } from "./api.js";
import { int, shortHash } from "./format.js";
import { escapeHtml } from "./render.js";
import { ROUTES, bindKeyboard, startRouter } from "./router.js";
import { store } from "./store.js";

const outlet = document.getElementById("outlet");
const nav = document.getElementById("nav");

nav.innerHTML = ROUTES.map(
    (r) => `<a href="#${r.path}" data-path="${r.path}">
                <span>${escapeHtml(r.label)}</span>
                <span class="key">g ${r.key}</span>
            </a>`,
).join("");

startRouter(outlet, (route) => {
    nav.querySelectorAll("a").forEach((a) => {
        a.classList.toggle("active", a.dataset.path === route.path);
    });
    document.title = `${route.label} · Tallystick`;
});

bindKeyboard();

/* The rail status strip is the persistent proof that the ledger is intact. It
   refreshes after every decision, so the demo never has to argue that it is live. */
async function refreshStatus() {
    const el = {
        model: document.getElementById("st-model"),
        hash: document.getElementById("st-hash"),
        height: document.getElementById("st-height"),
        chain: document.getElementById("st-chain"),
        dot: document.getElementById("st-dot"),
    };

    try {
        const health = await api.health();
        store.set({ health });
        el.model.textContent = health.model_version || "none";
        el.hash.textContent = shortHash(health.model_hash, 6, 4);
        el.height.textContent = int(health.ledger_height);
        el.chain.textContent = health.chain_ok ? "intact" : "broken";
        el.dot.className = `dot ${health.chain_ok ? "ok" : "bad"}`;

        const missing = Object.entries(health.artifacts).filter(([, ok]) => !ok);
        if (missing.length) {
            el.chain.title = `Missing artifacts: ${missing.map(([k]) => k).join(", ")}`;
        }
    } catch {
        el.chain.textContent = "unreachable";
        el.dot.className = "dot bad";
    }
}

refreshStatus();
setInterval(refreshStatus, 8000);

/* --- signed-in identity ------------------------------------------------ */

function initials(user) {
    const source = (user.full_name || user.email || "").trim();
    const parts = source.split(/[\s@._-]+/).filter(Boolean);
    return ((parts[0]?.[0] || "") + (parts[1]?.[0] || "")).slice(0, 2) || "··";
}

async function loadAccount() {
    let body;
    try {
        body = await api.me();
    } catch {
        return;
    }

    if (!body.authenticated) {
        // The server also guards /app, but a session can expire while the tab sits
        // open. Sending them to sign in beats letting every panel fail with a 401.
        location.href = "/login?next=/app";
        return;
    }

    const user = body.user;
    store.set({ user });

    const block = document.getElementById("account");
    block.hidden = false;
    block.classList.toggle("is-demo", user.is_demo);
    document.getElementById("acct-initials").textContent = initials(user);
    document.getElementById("acct-name").textContent = user.full_name || user.email;
    document.getElementById("acct-org").textContent = user.organisation || user.email;

    const role = document.getElementById("acct-role");
    role.textContent = user.is_demo ? "guest" : user.role;
    role.className = `pill ${user.is_demo ? "warn" : "accent"}`;
}

document.getElementById("signout").addEventListener("click", async () => {
    try {
        await api.logout();
    } finally {
        location.href = "/";
    }
});

loadAccount();
