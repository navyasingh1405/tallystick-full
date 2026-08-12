/* Where the API lives.
 *
 * The landing page can be served from the same origin as the API (running locally,
 * `tally demo`) or from a CDN with the API on a different host (the split deploy:
 * static site on Vercel, service on Fly). Both must work from one build of this file,
 * so the base URL is read at runtime rather than compiled in.
 *
 * `window.__TALLY_API__` is injected into the page by scripts/build_site.py. When it
 * is absent — which is the case when the service serves the page itself — every path
 * stays relative and nothing is cross-origin.
 */

const configured = (window.__TALLY_API__ || "").trim().replace(/\/+$/, "");

export const API_BASE = configured;

/** Resolve an API path against the configured base. */
export const api = (path) => `${API_BASE}${path}`;

/**
 * Where to send someone who wants the actual application.
 *
 * Sign-in happens on the service's own origin, never across origins. Cookies are
 * `SameSite=Lax`, so a session minted cross-site would not be sent back on
 * navigation; making it work would mean `SameSite=None`, which removes the
 * cross-site protection for the sake of a redirect we can do anyway.
 */
export const appUrl = (path = "/app") => `${API_BASE || ""}${path}`;

/** True when the page is served from somewhere other than the API. */
export const isSplitDeploy = Boolean(configured);
