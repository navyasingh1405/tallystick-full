# Thin delegator. `cal` is the real task runner; see src/tally/cli.py.
# GNU make is not present on Windows by default — use `uv run tally <target>`
# or `.\make.ps1 <target>` there.

.PHONY: setup doctor data train evaluate fairness lda drift pack batch-demo \
        verify-ledger reset-ledger demo all test test-fast fmt lint clean

setup:         ; uv run tally setup
doctor:        ; uv run tally doctor
data:          ; uv run tally data
train:         ; uv run tally train
evaluate:      ; uv run tally evaluate
fairness:      ; uv run tally fairness
lda:           ; uv run tally lda
drift:         ; uv run tally drift
pack:          ; uv run tally pack
batch-demo:    ; uv run tally batch-demo
verify-ledger: ; uv run tally verify-ledger
reset-ledger:  ; uv run tally reset-ledger
demo:          ; uv run tally demo
all:           ; uv run tally all
test:          ; uv run tally test
test-fast:     ; uv run tally test-fast
fmt:           ; uv run tally fmt
lint:          ; uv run tally lint
clean:         ; uv run tally clean

validation-pack: pack
