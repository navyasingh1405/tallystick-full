# Deployment

The marketing page goes on a CDN. The service goes on a host with a disk. They are
separate because they have genuinely different requirements, not because splitting
things is fashionable.

## Why the service cannot be serverless

This came up as "put it on Vercel", so it is worth writing down once.

| | Measured | Vercel serverless limit |
|---|---|---|
| Python dependencies | 792 MB | 250 MB unzipped |
| Playwright Chromium | 428 MB | — |

Size is the least of it. Three properties of the product are incompatible with a
stateless function:

- **The decision ledger is a file that must survive.** On a serverless host it lives
  in an ephemeral `/tmp` and disappears between invocations. The hash chain — the
  thing the product is — would reset every few minutes.
- **The chain assumes a single appender.** Serverless scales out by design. Two
  concurrent functions would read the same head hash and both write rows claiming it
  as their predecessor, which is a broken chain that nothing downstream would notice
  until someone tried to verify it.
- **PDF generation needs a browser.** Chromium does not run in that runtime, so the
  validation pack and the adverse action notices would simply be missing.

So: static site on Vercel, service on a machine with a volume.

---

## 1. The service

Fly.io is assumed below; Render, Railway, or any host that runs a container with a
persistent volume works the same way. The constraints that matter are **one instance**
and **a mounted disk**.

```bash
# One-off
fly auth login
fly launch --no-deploy          # accept the existing fly.toml when prompted
fly volumes create tally_data --size 3 --region lhr

fly deploy
```

The volume must exist before the first deploy: the container symlinks `/app/ledger`
to `/data`, and without the mount that path lives in the image layer and is discarded
on every release.

### What is baked into the image, and why

The trained model and the generated artifacts — metrics, fairness report, alternative
search, frontier plot, drift report — are copied into the image rather than built at
boot or fetched at runtime.

That is deliberate. They are the evidence the validation pack is assembled from, and
coupling them to the image means the document always describes the model that is
actually serving. Regenerating them requires a rebuild, which is the correct amount of
friction.

The ledger and the account database are the opposite: they belong to the deployment,
not the release, and live on the volume.

### Settings that differ per environment

Baked-in YAML is reviewable and version controlled, which is where configuration
belongs — except for the handful of values that legitimately change per environment.
Those read from the environment:

| Variable | Purpose |
|---|---|
| `TALLY_HOST`, `TALLY_PORT` | Bind address. Set in `fly.toml`. |
| `TALLY_COOKIE_SECURE` | Must be `true` anywhere reachable over a network. |
| `TALLY_ALLOWED_ORIGINS` | Comma-separated origins permitted to call the public API cross-origin. |
| `TALLY_ALLOW_INSTANT_DEMO` | Set `false` to require a real account. |
| `TALLY_SESSION_HOURS` | Session lifetime. |

Once the static site has a URL:

```bash
fly secrets set TALLY_ALLOWED_ORIGINS=https://tallystick.vercel.app
```

Leave it unset and the API refuses cross-origin calls entirely, which is the right
default. **Never set it to `*`** — that would let any page on the internet drive the
public scorer from a visitor's browser.

### Create the first account

```bash
fly ssh console -C "tally user create you@yourbank.com --role admin"
```

---

## 2. The marketing site

```bash
vercel login
vercel link
vercel env add TALLY_API_BASE production     # https://tallystick.fly.dev
vercel --prod
```

`vercel.json` runs `scripts/build_site.py`, which assembles `dist/` from the landing
page and exactly the assets it references — nothing else. `app.html` and `login.html`
are deliberately excluded: serving a dashboard shell from a host with no API behind it
gives a visitor a page that 401s on every call, and a sign-in form that posts into the
void.

To check a build before shipping it:

```bash
uv run python scripts/build_site.py --api https://tallystick.fly.dev
cd dist && python -m http.server 8099
```

The build fails rather than shipping a half-working page: it verifies every local
asset the HTML references is present, and that no `/api/` path was left relative.

---

## How the two halves talk

The landing page fetches three public endpoints cross-origin — `summary`, `score`,
and `frontier.png` — and does nothing else across the boundary.

**Sign-in never crosses origins.** The demo button and the sign-in links navigate to
the service and let it mint the session on its own origin. The session cookie is
`SameSite=Lax`, so one set cross-site would not be sent back on navigation; making it
work would mean `SameSite=None`, discarding the cross-site protection to save a
redirect that costs nothing.

Correspondingly, CORS is configured with `allow_credentials=False`. A cookie must
never ride a cross-origin request — that is precisely what would turn a permissive
CORS policy into a way for another site to act as a signed-in user.

---

## Before it is public

- [ ] `TALLY_COOKIE_SECURE=true` (set in `fly.toml`; confirm it survived).
- [ ] `TALLY_ALLOWED_ORIGINS` names the real site, and is not `*`.
- [ ] `fly volumes list` shows `tally_data` mounted at `/data`.
- [ ] `max_machines_running = 1`. Two appenders will corrupt the chain.
- [ ] `tally verify-ledger` on the deployed instance returns intact.
- [ ] Decide whether `TALLY_ALLOW_INSTANT_DEMO` should stay on. It provisions a real
      account per click, and it is what makes the demo one click.
- [ ] Read `docs/security.md`. The authentication here is adequate for a
      demonstration on public data and is not a model for protecting a real credit
      system.

## What this does not include

No CI, no staging environment, no backup of the volume, no log aggregation, no
custom domain, and no rate limiting beyond the in-process throttles on the public
scorer and the sign-in form. Each is straightforward; none is done.
