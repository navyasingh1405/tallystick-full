/* Sign in, register, and instant demo access. */

const $ = (sel) => document.querySelector(sel);

function nextUrl() {
    const next = new URLSearchParams(location.search).get("next") || "/app";
    // Only ever redirect within this origin. An open redirect on a sign-in page is
    // a phishing primitive: it lends this domain's credibility to someone else's.
    return next.startsWith("/") && !next.startsWith("//") ? next : "/app";
}

async function post(path, body) {
    const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
        throw new Error(payload?.error?.message || `${response.status} ${response.statusText}`);
    }
    return payload;
}

function showError(el, message) {
    el.textContent = message;
    el.hidden = false;
}

function busy(button, on, label) {
    button.disabled = on;
    button.innerHTML = on
        ? '<span class="spinner"></span> ' + label
        : button.dataset.label || label;
}

/* --- tabs -------------------------------------------------------------- */

document.querySelectorAll(".auth-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".auth-tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        const signin = tab.dataset.tab === "signin";
        $("#signin-form").hidden = !signin;
        $("#register-form").hidden = signin;
        $("#signin-error").hidden = true;
        $("#register-error").hidden = true;
    });
});

/* --- sign in ----------------------------------------------------------- */

$("#signin-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.target.querySelector("button[type=submit]");
    button.dataset.label = "Sign in";
    const error = $("#signin-error");
    error.hidden = true;
    busy(button, true, "Signing in…");

    try {
        await post("/api/auth/login", {
            email: $("#signin-email").value.trim(),
            password: $("#signin-password").value,
        });
        location.href = nextUrl();
    } catch (err) {
        showError(error, err.message);
        busy(button, false, "Sign in");
        $("#signin-password").value = "";
        $("#signin-password").focus();
    }
});

/* --- register ---------------------------------------------------------- */

let strengthTimer = null;

$("#reg-password").addEventListener("input", (event) => {
    const box = $("#strength");
    const value = event.target.value;
    if (!value) {
        box.hidden = true;
        return;
    }
    box.hidden = false;
    clearTimeout(strengthTimer);
    // Debounced, and the password is checked server-side against the same policy the
    // account creation path uses, so the two can never disagree.
    strengthTimer = setTimeout(async () => {
        try {
            const result = await post("/api/auth/check-password", {
                password: value,
                email: $("#reg-email").value.trim(),
            });
            const fill = box.querySelector(".strength-fill");
            const text = box.querySelector(".strength-text");
            if (result.ok) {
                const score = Math.min(100, 40 + (value.length - result.min_length) * 5);
                fill.style.width = `${Math.max(55, score)}%`;
                fill.className = "strength-fill good";
                text.textContent = "Looks good.";
                text.className = "strength-text good";
            } else {
                fill.style.width = `${Math.min(45, (value.length / result.min_length) * 45)}%`;
                fill.className = "strength-fill weak";
                text.textContent = result.problems[0];
                text.className = "strength-text weak";
            }
        } catch {
            /* strength feedback is a nicety; never block typing on it */
        }
    }, 220);
});

$("#register-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.target.querySelector("button[type=submit]");
    button.dataset.label = "Create account";
    const error = $("#register-error");
    error.hidden = true;
    busy(button, true, "Creating…");

    try {
        await post("/api/auth/register", {
            email: $("#reg-email").value.trim(),
            password: $("#reg-password").value,
            full_name: $("#reg-name").value.trim() || null,
            organisation: $("#reg-org").value.trim() || null,
        });
        location.href = nextUrl();
    } catch (err) {
        showError(error, err.message);
        busy(button, false, "Create account");
    }
});

/* --- instant demo ------------------------------------------------------ */

$("#demo-btn").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const original = button.innerHTML;
    button.disabled = true;
    button.innerHTML =
        '<span class="auth-demo-main"><span class="spinner"></span> Provisioning a guest session…</span>';

    try {
        await post("/api/auth/demo");
        location.href = nextUrl();
    } catch (err) {
        button.disabled = false;
        button.innerHTML = original;
        showError($("#signin-error"), err.message);
    }
});

/* Already signed in? Do not make someone log in twice. */
fetch("/api/auth/me")
    .then((r) => r.json())
    .then((body) => {
        if (body.authenticated) location.href = nextUrl();
    })
    .catch(() => {});
