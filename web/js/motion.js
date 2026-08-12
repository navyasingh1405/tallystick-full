/* Motion primitives.
 *
 * Two rules, both learned the hard way on this page:
 *
 * 1. **Nothing is ever hidden by CSS alone.** Content is visible by default; the
 *    hidden state is applied by script that has already proven it can remove it, and
 *    a backstop removes the whole opt-in class rather than adding a reveal class —
 *    because a transition that never advances (a tab that never composited, a
 *    viewport that is not painting) leaves an element at opacity 0 forever.
 *
 * 2. **Reduced motion is honoured at the source, not by a media query bolted on.**
 *    Every helper below checks it and jumps straight to the final state. Someone who
 *    has asked for less movement should get the information instantly, not a
 *    disabled animation that still delays the answer.
 */

export const reduced = () =>
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
const easeOutQuint = (t) => 1 - Math.pow(1 - t, 5);

/**
 * Animate a number up to `to`.
 *
 * The final value is written *first*, before any animation starts. requestAnimationFrame
 * does not run in a tab that is not painting — a background tab, a hidden pane — so a
 * counter that only ever gets its value from inside a rAF callback stays showing its
 * placeholder indefinitely for anyone who opened the page in a second tab. Writing the
 * answer up front costs one frame of visual backtrack and removes that whole class of
 * failure.
 */
export function tweenNumber(el, to, format, { duration = 700, from = null } = {}) {
    if (!el) return;
    const start = from === null ? Number(el.dataset.value ?? 0) : from;
    el.dataset.value = String(to);
    el.textContent = format(to);

    if (reduced() || !Number.isFinite(start) || start === to) return;

    const t0 = performance.now();
    const step = (now) => {
        const t = Math.min(1, (now - t0) / duration);
        el.textContent = format(start + (to - start) * easeOutCubic(t));
        if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

/** Reveal children one after another. Returns immediately under reduced motion. */
export function stagger(elements, { delay = 60, initial = 0 } = {}) {
    elements.forEach((el, i) => {
        if (reduced()) {
            el.classList.add("in");
            return;
        }
        setTimeout(() => el.classList.add("in"), initial + i * delay);
    });
}

/**
 * Run `fn(t)` for `duration` ms with `t` eased 0 → 1.
 * Used for anything that is not a plain number or class toggle.
 */
export function animate(duration, fn, ease = easeOutQuint) {
    if (reduced()) {
        fn(1);
        return () => {};
    }
    let cancelled = false;
    const t0 = performance.now();
    const step = (now) => {
        if (cancelled) return;
        const t = Math.min(1, (now - t0) / duration);
        fn(ease(t));
        if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
    return () => {
        cancelled = true;
    };
}

/**
 * Call `fn` once, the first time `el` scrolls into view.
 * Falls back to calling immediately where IntersectionObserver is unavailable.
 */
export function whenVisible(el, fn, { threshold = 0.35 } = {}) {
    if (!el) return;
    if (!("IntersectionObserver" in window)) {
        fn();
        return;
    }
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    observer.unobserve(entry.target);
                    fn();
                }
            });
        },
        { threshold },
    );
    observer.observe(el);
}

/**
 * Highlight the nav link for whichever section is currently in view.
 *
 * Driven by scroll position rather than by IntersectionObserver ratios: several
 * sections are taller than the viewport, so "most visible" is meaningless for them,
 * whereas "which heading did we last pass" always has an answer.
 */
export function scrollSpy(links, { offset = 120 } = {}) {
    const targets = links
        .map((link) => ({ link, section: document.querySelector(link.getAttribute("href")) }))
        .filter((entry) => entry.section);
    if (!targets.length) return;

    let frame = null;
    const update = () => {
        frame = null;
        const y = window.scrollY + offset;
        let current = null;
        for (const entry of targets) {
            if (entry.section.offsetTop <= y) current = entry;
        }
        targets.forEach(({ link }) => link.classList.toggle("current", link === current?.link));
    };

    const onScroll = () => {
        if (frame === null) frame = requestAnimationFrame(update);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    update();
}

/** Swap an element's text with a brief scale/fade, for values that change in place. */
export function swapText(el, text, { className = null } = {}) {
    if (!el) return;
    if (el.textContent === text && !className) return;

    const apply = () => {
        el.textContent = text;
        if (className) el.className = className;
    };

    if (reduced()) {
        apply();
        return;
    }
    el.classList.add("swapping");
    setTimeout(() => {
        apply();
        el.classList.remove("swapping");
    }, 120);
}
