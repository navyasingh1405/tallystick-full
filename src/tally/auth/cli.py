"""Account administration from the command line.

Elevation to `analyst` or `admin` is only possible here. A signup form that could
grant itself a role would make the role meaningless.
"""

from __future__ import annotations

import getpass
from typing import Annotated

import typer

from tally.auth.passwords import MIN_LENGTH, PasswordPolicyError, check_policy
from tally.auth.store import ROLES, AuthStore, EmailAlreadyRegistered

app = typer.Typer(help="Manage accounts.", no_args_is_help=True)


def _prompt_password(email: str) -> str:
    """Read a password without echoing it, and never accept it from a flag.

    A password passed as a command-line argument lands in the shell history and in
    the process table, where anyone on the machine can read it.
    """
    for _ in range(3):
        first = getpass.getpass("Password: ")
        result = check_policy(first, email=email)
        if not result.ok:
            for problem in result.problems:
                print(f"  {problem}")
            continue
        if first != getpass.getpass("Confirm password: "):
            print("  Passwords did not match.")
            continue
        return first
    raise typer.Exit(1)


@app.command("create")
def create(
    email: str,
    role: Annotated[str, typer.Option(help=f"One of: {', '.join(ROLES)}")] = "analyst",
    name: Annotated[str, typer.Option(help="Full name.")] = "",
    org: Annotated[str, typer.Option(help="Organisation.")] = "",
) -> None:
    """Create an account. The password is prompted for, never passed as an argument."""
    if role not in ROLES:
        print(f"  Unknown role {role!r}. Expected one of: {', '.join(ROLES)}")
        raise typer.Exit(1)

    store = AuthStore.default()
    password = _prompt_password(email)
    try:
        user = store.create_user(
            email, password, full_name=name or None, organisation=org or None, role=role
        )
    except EmailAlreadyRegistered:
        print(f"  {email} is already registered.")
        raise typer.Exit(1) from None
    except (PasswordPolicyError, ValueError) as exc:
        print(f"  {exc}")
        raise typer.Exit(1) from None

    print(f"  created {user.email} with role {user.role}")


@app.command("list")
def list_users() -> None:
    """List accounts. Password hashes are never printed."""
    users = AuthStore.default().list_users()
    if not users:
        print("  No accounts yet. Create one with: uv run tally user create you@example.com")
        return
    print(f"  {'email':<38} {'role':<9} {'active':<7} {'last sign-in':<28} organisation")
    for u in users:
        print(
            f"  {u.email:<38} {u.role:<9} {'yes' if u.is_active else 'no':<7} "
            f"{(u.last_login_at or 'never'):<28} {u.organisation or ''}"
        )


@app.command("role")
def set_role(email: str, role: str) -> None:
    """Change an account's role."""
    if role not in ROLES:
        print(f"  Unknown role {role!r}. Expected one of: {', '.join(ROLES)}")
        raise typer.Exit(1)
    store = AuthStore.default()
    if store.get_user_by_email(email) is None:
        print(f"  No account for {email}.")
        raise typer.Exit(1)
    store.set_role(email, role)
    print(f"  {email} is now {role}")


@app.command("disable")
def disable(email: str) -> None:
    """Deactivate an account and sign out every session it holds."""
    store = AuthStore.default()
    user = store.get_user_by_email(email)
    if user is None:
        print(f"  No account for {email}.")
        raise typer.Exit(1)
    store.set_active(email, False)
    store.end_all_sessions(user.id)
    print(f"  {email} disabled and signed out everywhere")


@app.command("enable")
def enable(email: str) -> None:
    """Reactivate an account."""
    AuthStore.default().set_active(email, True)
    print(f"  {email} enabled")


@app.command("passwd")
def passwd(email: str) -> None:
    """Set a new password. Every existing session for the account is signed out."""
    store = AuthStore.default()
    if store.get_user_by_email(email) is None:
        print(f"  No account for {email}.")
        raise typer.Exit(1)
    password = _prompt_password(email)
    store.change_password(email, password)
    print(f"  password updated for {email}; all sessions signed out")


@app.command("purge-sessions")
def purge() -> None:
    """Delete expired sessions."""
    removed = AuthStore.default().purge_expired()
    print(f"  removed {removed} expired session(s)")


@app.command("policy")
def policy() -> None:
    """Show the password policy."""
    print(f"  Minimum length: {MIN_LENGTH} characters")
    print("  No composition rules. Length is what matters; forcing a digit and a")
    print("  symbol pushes people toward predictable substitutions and a sticky note.")
    print("  Hashing: argon2id (64 MiB, 3 iterations, 4 lanes).")
