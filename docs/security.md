# Authentication: what it does, and what it does not

Written plainly because the failure mode of security documentation is implying more
than was built. This system holds no customer records, no credentials of value, and
no connection to anything real. It is a demonstration on public FFIEC data. Accounts
exist for one reason: so that actions recorded in the audit trail have a name against
them.

## Why it exists at all

Before sign-in, the override endpoint took an `actor` string from the request body.
That made the audit trail a claim rather than a record — anyone calling the API could
attribute an override to a colleague, a director, or nobody at all. A tamper-evident
ledger whose "who did this" field is free text is tamper-evident about the wrong
thing.

The actor is now the authenticated user, read from the session. `OverrideRequest` has
no `actor` field, so supplying one is rejected as an unknown field rather than
ignored. There is a test for each half of that.

## What is implemented

**Password storage.** argon2id at the reference defaults (64 MiB, 3 iterations, 4
lanes), with a per-password salt. Parameters live inside the hash string, so raising
them later still verifies existing passwords and `needs_rehash` reports which to
upgrade — which happens transparently on next sign-in.

**Password policy.** Minimum twelve characters, no composition rules. Length is what
resists guessing; requiring a digit and a symbol reliably produces `Password1!` and
teaches people to write passwords down. A short blocklist catches common bases even
when padded to length, so `password1234` is refused.

**Sessions.** The cookie carries an opaque 256-bit random token. The database stores
only its SHA-256. A read of the sessions table therefore does not hand anybody a
usable cookie. Cookies are `HttpOnly` (unreachable from JavaScript, so a cross-site
scripting bug cannot lift one), `SameSite=Lax` (not sent on cross-site POSTs), and
scoped to `/`.

Sessions expire server-side, can be revoked individually or all at once, and are
invalidated wholesale when a password changes or an account is disabled. A signed
cookie carrying the user's identity — the common shortcut — would have none of those
properties.

**Sign-in throttling.** Failed attempts are counted per account and per source
address over a rolling window, with a wider allowance per source so one shared office
NAT cannot lock out a floor of people. A successful sign-in clears the account's
failure budget.

**Account enumeration.** "No such account" and "wrong password" return the same status
and the same message. When the account does not exist, the code still spends the CPU
of a hash verification, so response timing does not disclose the difference either.
Registration necessarily reveals that an address is taken — there is no way around
that for a form which must prevent duplicates — and sign-in does not.

**Authorisation.** Three roles: `demo`, `analyst`, `admin`. Permissions are an
explicit allowlist per role; anything unlisted is denied. Route protection is a
per-route dependency rather than a prefix list, so a route is public only if its own
signature says so — and a test enumerates every `/api/` route and fails if a new one
is neither protected nor explicitly listed as public with a reason.

**Role elevation** happens only through `tally user role` on the command line. The
registration form cannot grant a role; attempting to pass one is rejected.

## What is not implemented

- **No federated identity.** No SSO, no SCIM, no directory integration. A bank would
  want all three, and would not want a local password table at all.
- **No multi-tenancy.** Every account sees the same model, ledger, and artifacts.
- **No second factor.** For a demonstration on public data, a TOTP enrolment flow
  would be ceremony without benefit.
- **No CSRF token.** The API accepts only `application/json`, and `SameSite=Lax`
  withholds the cookie on cross-site POSTs, which together cover the ordinary case. A
  deployment behind a shared domain, or one accepting form encodings, would need
  explicit tokens.
- **No secure-cookie enforcement by default.** `cookie_secure` is off in config
  because the demo runs on `http://127.0.0.1`. Anything served over a network must
  turn it on and terminate TLS; the setting exists so that is a one-line change rather
  than a code change.
- **No password reset flow.** There is no mail transport here. `tally user passwd`
  covers it operationally.
- **No breached-password check.** Catching genuinely compromised passwords needs an
  external corpus and a k-anonymity lookup. The blocklist is a floor, not a substitute.
- **No audit log of authentication events** beyond failed sign-in attempts retained
  for throttling. Successful sign-ins update `last_login_at` and nothing more.

## Where accounts live

In `ledger/accounts.db`, a **different database** from `ledger/decisions.db`.

That separation is not filing tidiness. The decision ledger is append-only, enforced
by SQLite triggers, and chained by hash. User records must be updatable — password
changes, last-login timestamps, deactivation — and deletable on request. One file
cannot honour both sets of requirements: either the ledger's guarantees get weakened
to allow account management, or account management becomes impossible. A test asserts
the two paths differ and that the accounts schema contains no ledger tables.

## The guest demo path

Clicking "open the demo" provisions a fresh account with a random address and a random
password nobody ever sees, signs it in, and expires it by itself. Each click gets its
own account rather than a shared login, so guest activity stays attributable and any
one session can be revoked without signing out everyone else.

Guest accounts can do everything an analyst can, including record overrides. That is
deliberate: an evaluator should be able to see the accountability mechanism work,
and every action they take is recorded against their own guest identity.

## The public scorer

The landing page scores against the real model without a sign-in, rate limited by
source address. Those decisions **are written to the ledger**, like any other.

Carving out an unrecorded path would have made the product's central claim — that
nothing scores without being recorded — false in precisely the place it is being
advertised. The endpoint takes the same `RawApplicant` schema as everything else, so
it is structurally incapable of accepting a protected attribute.
