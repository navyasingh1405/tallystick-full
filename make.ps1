# Thin delegator. `cal` is the real task runner; see src/tally/cli.py.
param(
    [Parameter(Position = 0)]
    [string]$Target = "demo",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

uv run tally $Target @Rest
exit $LASTEXITCODE
