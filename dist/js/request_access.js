/* The access request form on the static site.
 *
 * There is no server here, so this does the only honest thing a static page can do
 * with a form: compose the message and hand it to the visitor's own mail client,
 * where they can read it before it goes. Nothing is transmitted by this page, and
 * the message is shown in full so that "sent on your behalf" is never a guess.
 *
 * build_site.py substitutes the contact address at build time.
 */

const $ = (s) => document.querySelector(s);

//: Replaced by build_site.py. The literal default is a placeholder that would be
//: obvious in a mail client rather than a plausible address that silently misroutes.
const CONTACT = window.__TALLY_CONTACT__ || "hello@example.com";

const form = $("#request-form");
const sent = $("#request-sent");
const error = $("#request-error");

function compose() {
    const value = (id) => $(id).value.trim();
    const lines = [
        `Institution: ${value("#req-org")}`,
        `Name:        ${value("#req-name")}`,
        `Work email:  ${value("#req-email")}`,
    ];
    const role = value("#req-role");
    if (role) lines.push(`Role:        ${role}`);

    const interest = value("#req-interest");
    const body = interest
        ? `${lines.join("\n")}\n\nWhat we would like to see:\n${interest}\n`
        : `${lines.join("\n")}\n`;

    return { subject: `Tallystick access — ${value("#req-org") || "request"}`, body };
}

form.addEventListener("submit", (event) => {
    event.preventDefault();
    error.hidden = true;

    // Checked here rather than by the browser's own validation so the message is in
    // the page's voice, and so an empty institution is caught — required, but not
    // something a type="email" check would notice.
    const missing = ["#req-name", "#req-org", "#req-email"].filter((id) => !$(id).value.trim());
    if (missing.length) {
        error.textContent = "Name, institution, and work email are needed to set an instance up.";
        error.hidden = false;
        $(missing[0]).focus();
        return;
    }
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test($("#req-email").value.trim())) {
        error.textContent = "That work email does not look like an address.";
        error.hidden = false;
        $("#req-email").focus();
        return;
    }

    const { subject, body } = compose();

    form.hidden = true;
    sent.hidden = false;
    $("#sent-contact").textContent = CONTACT;
    $("#sent-body").textContent = `To: ${CONTACT}\nSubject: ${subject}\n\n${body}`;

    // Opens the visitor's mail client with the message ready. It is theirs to send
    // or discard; this page cannot and does not send anything itself.
    location.href =
        `mailto:${encodeURIComponent(CONTACT)}` +
        `?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
});

$("#copy-btn").addEventListener("click", async () => {
    const button = $("#copy-btn");
    try {
        await navigator.clipboard.writeText($("#sent-body").textContent);
        button.textContent = "Copied";
    } catch {
        // Clipboard access is refused in plenty of ordinary situations. Selecting the
        // text is a working fallback and says what to do next.
        const range = document.createRange();
        range.selectNodeContents($("#sent-body"));
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        button.textContent = "Selected — press Ctrl+C";
    }
    setTimeout(() => (button.textContent = "Copy the message"), 2600);
});

$("#again").addEventListener("click", (event) => {
    event.preventDefault();
    sent.hidden = true;
    form.hidden = false;
});
