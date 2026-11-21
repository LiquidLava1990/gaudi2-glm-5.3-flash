#!/bin/bash
# r4e_run.sh without a boot: config assembly, refusals and the gate readers on fixture files. Host-only (bash, awk, sha256sum), no docker, no card.
# usage: bash tests/test_r4e_args.sh   (PROOF_PREFILL_SH=... to test against a staged proof_prefill.sh.new)
set -u
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # not HERE: final_run.sh sets HERE to diag-final
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
LEVERS="$TMP/levers.env"
printf '%s\n' '# test levers' VLLM_GAUDI_MIXED_PROMPT_MOE=ragged HPU_GLM53_RAGGED_GRAPH_MOE=1 HPU_GLM53_RAGGED_GRAPH_TABLE=abc12345 \
    HPU_GLM53_REUSE_V2_FOR=session $(grep '^VLLM_DECODE_PRIORITY' "$(dirname "${BASH_SOURCE[0]}")/../r4e-levers.env") > "$LEVERS"
export OUT_ROOT="$TMP" R4E_RUN_NO_MAIN=1 R4E_IMAGE=test-r4e:none R4E_IMAGE_SHA=sha256:000000000000 R4E_LEVERS_ENV="$LEVERS"
export R4E_GATES_OVERRIDE=none   # the defaults below; the base-r4e override file is tested on its own at the end
source "$KIT/r4e_run.sh" test-r4e B
fails=0; total=0
check() {  # DESCRIPTION COMMAND...: one line per case
    local what="$1"; shift
    total=$(( total + 1 ))
    if "$@"; then echo "ok   $what"; else echo "FAIL $what"; fails=$(( fails + 1 )); fi
}
negate() { ! "$@"; }
quiet() { "$@" > /dev/null; }
env_value_is() { printf '%s\n' "${ARGS[@]}" | grep -qx -- "$1"; }
every_line_in_args() { local l; for l in $(grep -v -E '^(#|$)' "$1"); do env_value_is "$l" || return 1; done; }
mkdir -p "$RUN"

check "names: arm r4e, run dir, container, quad B port" bash -c '[ "$1" = r4e ] && [ "$2" = "$3/graded/r4e" ] && [ "$4" = claude-flash-launch-final-r4e ] && [ "$5" = http://127.0.0.1:18441 ]' _ "$ARM" "$RUN" "$OUT" "$NAME" "$URL"
check "image from R4E_IMAGE" [ "$IMAGE" = test-r4e:none ]
# r4f (coordinator 04:3x): two arms of one image at once need their own run dir and container name
arm_of() { bash -c 'R4E_ARM="$2" source "$1" test-arm A > /dev/null 2>&1; printf "%s %s %s" "$ARM" "$RUN" "$NAME"' _ "$KIT/r4e_run.sh" "$1" 2> /dev/null; }
check "R4E_ARM names the arm, its run dir and its container" bash -c '[[ "$1" == "r4f-a "*"/test-arm/graded/r4f-a claude-flash-launch-final-r4f-a" ]]' _ "$(arm_of r4f-a)"
check "R4E_ARM unset keeps arm r4e" bash -c '[[ "$1" == "r4e "*"/graded/r4e claude-flash-launch-final-r4e" ]]' _ "$(arm_of "")"
check "R4E_ARM refuses a name a container cannot carry" bash -c '! R4E_ARM="bad arm" bash -c "source \"\$1\" test-arm A" _ "$1" > /dev/null 2>&1' _ "$KIT/r4e_run.sh"
check "status file is the run's own" [ "$STATUS" = "$TMP/test-r4e/status" ]
check "sourcing proof_prefill.sh wrote no driver.pid and set no EXIT trap" bash -c '[ ! -e "$1/test-r4e/driver.pid" ] && [ -z "$2" ]' _ "$TMP" "$(trap -p EXIT | grep -v rm)"
check "proof_prefill functions present" quiet declare -F profile_medium reports ratio_of say_v2_load_proof
check "final_run functions present" quiet declare -F final_args forbid static_guard take_placed_quad finish
check "r4e_args assembles" r4e_args
# Opus round 3 NIT: a lever line with no '=' would become '--env KEY' (docker passes the host's value) and forbid() never sees it
{ cat "$LEVERS"; echo VLLM_RECURRENT_TAIL_FUSE; } > "$TMP/levers-noeq.env"
levers_refused() { ( R4E_LEVERS_ENV="$1"; ! r4e_args ) > /dev/null 2>&1; }
check "r4e_args refuses a lever line with no '='" levers_refused "$TMP/levers-noeq.env"
champion_minus_levers() { grep -v -E '^(#|$)' "$FINAL_ENV" | grep -v -E "^($(grep -v -E '^(#|$)' "$LEVERS" | cut -d= -f1 | paste -sd'|'))=" > "$TMP/champion-rest.env"; }
champion_minus_levers
check "every champion env line the levers do not override is in ARGS" every_line_in_args "$TMP/champion-rest.env"
check "every lever line is in ARGS" every_line_in_args "$LEVERS"
check "no loadpath overlay mount (r4e bakes it)" negate bash -c 'printf "%s\n" "$@" | grep -q "diag-loadpath/glm53_loadpath.py"' _ "${ARGS[@]}"
check "forbid() accepts the r4e config" forbid
check "require_levers accepts the four keys" require_levers
# r4e-A2 served the r4d keys (coordinator 04:4x): c1d_args sets VLLM_GAUDI_MIXED_PROMPT_KEYS twice and set_env replaced only the first,
# so docker's last --env (128:8192) won over the lever. A lever line now replaces every entry of its key, and differing duplicates HALT.
saved=("${ARGS[@]}")
ARGS=(--env K=1 --mount x --env K=2 --env J=3)
set_env_every K K=9
check "set_env_every drops every KEY entry and appends LINE once" bash -c '[ "$*" = "--mount x --env J=3 --env K=9" ]' _ "${ARGS[@]}"
ARGS=(--env K=1 --env J=3 --env K=2 --env J=3)
check "env_conflicts names a key set twice to different values only" [ "$(env_conflicts)" = K ]
ARGS=(--env J=3 --env J=3)
check "env_conflicts is quiet on identical duplicates" [ -z "$(env_conflicts)" ]
ARGS=(--env K=1 --mount x --env J=3 --env K=2)
collapse_env
check "collapse_env keeps only each key's last entry (docker's rule)" bash -c '[ "$*" = "--mount x --env J=3 --env K=2" ]' _ "${ARGS[@]}"
ARGS=("${saved[@]}")
check "r4e_args leaves no key with differing values" [ -z "$(env_conflicts)" ]
real_keys() { bash -c 'export R4E_LEVERS_ENV="$2"; source "$1" test-keys A > /dev/null 2>&1; r4e_args > /dev/null 2>&1 || exit 9
    for (( i = 1; i < ${#ARGS[@]}; i++ )); do [ "${ARGS[i-1]}" = --env ] && case "${ARGS[i]}" in VLLM_GAUDI_MIXED_PROMPT_KEYS=*) echo "${ARGS[i]}" ;; esac; done' \
    _ "$KIT/r4e_run.sh" "$KIT/r4e-levers.env" 2> /dev/null; }
check "the r4e levers serve the 256 set, once (not c1d_args' later r4d keys)" \
    [ "$(real_keys)" = VLLM_GAUDI_MIXED_PROMPT_KEYS=1024:8192,1024:1024,256:1024,128:1024 ]
# R4E_FORBIDDEN_KEYS only adds keys (Opus 16:5x BLOCKER: replacing the list bypassed the ARM check); a forbidden key is admitted
# only by final_run.sh's baked-ARM-file check (R4E_ARM_ENV from the gates, its sha one of final_run.sh's BAKED_ARM_SHA256S)
saved=("${ARGS[@]}")
ARGS+=(--env VLLM_RECURRENT_TAIL_FUSE=1)
check "forbid() refuses the tail fuse by default" negate forbid
forbidden_with() { bash -c 'R4E_FORBIDDEN_KEYS="$2" source "$1" test-fk A > /dev/null 2>&1; printf "%s" "$FORBIDDEN_KEYS"' _ "$KIT/r4e_run.sh" "$1" 2> /dev/null; }
check "R4E_FORBIDDEN_KEYS unset keeps final_run.sh's list" bash -c '[[ "$1" == *"|VLLM_RECURRENT_TAIL_FUSE|"* ]]' _ "$(forbidden_with "")"
check "R4E_FORBIDDEN_KEYS adds to final_run.sh's list" [ "$(forbidden_with 'EXTRA_KEY')" = "$(forbidden_with '')|EXTRA_KEY" ]
check "R4E_FORBIDDEN_KEYS that leaves the tail fuse out still forbids it" \
    bash -c '[[ "$1" == *"|VLLM_RECURRENT_TAIL_FUSE|"* ]]' _ "$(forbidden_with 'HPU_GLM53_MIN_DECODE_STEPS|VLLM_RECURRENT_TAIL_FUSE_KEYED')"
check "R4E_FORBIDDEN_KEYS refuses a list that is not KEY|KEY" bash -c '! R4E_FORBIDDEN_KEYS="A B" bash -c "source \"\$1\" test-fk A" _ "$1" > /dev/null 2>&1' _ "$KIT/r4e_run.sh"
FUSE_ARM="$(mktemp)"; echo VLLM_RECURRENT_TAIL_FUSE=1 > "$FUSE_ARM"
R4E_ARM_ENV="$FUSE_ARM"
R4E_ARM_ENV_SHA256="$(sha256sum "$FUSE_ARM" | cut -d' ' -f1)"
check "forbid() refuses the tail fuse from an ARM file no bake made, even when the gates pin it" negate forbid
saved_allow=("${BAKED_ARM_SHA256S[@]}"); BAKED_ARM_SHA256S+=("$R4E_ARM_ENV_SHA256")
check "forbid() admits the tail fuse only through a baked ARM file" quiet forbid
BAKED_ARM_SHA256S=("${saved_allow[@]}")
unset R4E_ARM_ENV R4E_ARM_ENV_SHA256; rm -f "$FUSE_ARM"; ARGS=("${saved[@]}")
drop_mount none:none:ro
check "drop_mount of an absent spec keeps ARGS" bash -c '[ "$1" = "$2" ]' _ "${#saved[@]}" "${#ARGS[@]}"
ARGS=(--env A=1 --mount x:y:ro --env B=2)
drop_mount x:y:ro
check "drop_mount removes only its pair" bash -c '[ "$*" = "--env A=1 --env B=2" ]' _ "${ARGS[@]}"
ARGS=("${saved[@]}")
check "lever 1 tuned overrides the champion's VLLM_DECODE_PRIORITY=0" bash -c '[ "$(printf "%s\n" "$@" | grep -c "^VLLM_DECODE_PRIORITY=")" = 1 ] && printf "%s\n" "$@" | grep -qx VLLM_DECODE_PRIORITY=phases' _ "${ARGS[@]}"
check "the kit's r4e-levers.env carries the four lever-1 lines" bash -c 'for l in $2; do grep -qx "$l" "$1" || exit 1; done' _ "$KIT/r4e-levers.env" "$R4E_REQUIRED_LINES"
check "speed bars are lever 1's own: 1.5 / 1.3 / 0.7" [ "$SPEED_MIN_DECODE/$SPEED_MAX_TTFT/$SPEED_MIN_PREFILL" = 1.5/1.3/0.7 ]
R4E_REQUIRED_LINES="VLLM_DECODE_PRIORITY=phases VLLM_DECODE_PRIORITY_MIN_DECODE_STEPS=99"
check "require_levers refuses a lever-1 line that is not booted" negate require_levers
R4E_REQUIRED_LINES="VLLM_DECODE_PRIORITY=phases"
R4E_REQUIRED_KEYS="HPU_GLM53_REUSE_V2_FOR HPU_GLM53_NOT_SET"
check "require_levers refuses a missing lever key" negate require_levers
R4E_REQUIRED_KEYS="HPU_GLM53_REUSE_V2_FOR"
check "lever_summary prints the lever lines" bash -c '[[ "$1" == *"HPU_GLM53_REUSE_V2_FOR=session"* ]]' _ "$(lever_summary)"

log="$TMP/container.log"
printf '%s\n' 'x GLM53_GRAPHS_ONLY keys=1024:8192 ragged_pieces=100 ragged_overflows=1 ragged_reruns=1' 'x GLM53_GRAPHS_ONLY ragged_pieces=200 ragged_overflows=4 ragged_reruns=2' \
    'x HPU_GLM53_REUSE_V2_FOR armed' > "$log"
routed_counters "$log"; check "routed PASS at 2% overflow" [ "$ROUTED" = PASS ]
echo 'x GLM53_GRAPHS_ONLY ragged_pieces=200 ragged_overflows=5' >> "$log"
routed_counters "$log"; check "routed FAIL above 2% overflow" [ "$ROUTED" = FAIL ]
echo 'x GLM53_GRAPHS_ONLY ragged_pieces=200 ragged_overflows=0' >> "$log"; echo 'x lazy fallback taken' >> "$log"
routed_counters "$log"; check "routed FAIL on a lazy fallback line" [ "$ROUTED" = FAIL ]
routed_counters /dev/null; check "routed FAIL without counter lines" [ "$ROUTED" = FAIL ]

report="$TMP/loadpath_report.txt"
printf '%s\n' 'LOADPATH_PROOF loaded_gb=40 external_hit_pct=81.5' 'LOADPATH_PROOF loaded_gb=3.9 external_hit_pct=26.2' > "$report"
pin_counters "$log" "$report"; check "pin PASS: live line and hit % up" [ "$PIN" = PASS ]
printf '%s\n' 'LOADPATH_PROOF external_hit_pct=20.0' 'LOADPATH_PROOF external_hit_pct=26.2' > "$report"
pin_counters "$log" "$report"; check "pin FAIL: hit % down" [ "$PIN" = FAIL ]
pin_counters /dev/null /dev/null; check "pin FAIL: no lines, no report" [ "$PIN" = FAIL ]

lanes="$TMP/lane_report.txt"
echo 'DECODEPRIO_REPORT lanes=16 decode_tok_s=3.40 (baseline 2.25, 1.51x) prefill_tok_s=8000 (baseline 10192, 0.78x) ttft_s=8 (baseline 6.5, 1.23x)' > "$lanes"
speed_gate "$lanes"; check "speed PASS: decode 1.51x, prefill 0.78x, TTFT 1.23x" [ "$SPEED" = PASS ]
echo 'DECODEPRIO_REPORT decode_tok_s=3.0 (baseline 2.25, 1.33x) prefill_tok_s=13000 (baseline 12000, 1.08x) ttft_s=6 (baseline 6.5, 0.92x)' > "$lanes"
speed_gate "$lanes"; check "speed FAIL: decode 1.33x is under the 1.5x bar" [ "$SPEED" = FAIL ]
echo 'DECODEPRIO_REPORT lanes=0 decode_tok_s=- (baseline 0.84, -)' > "$lanes"
speed_gate "$lanes"; check "speed UNJUDGED without ratios" [ "$SPEED" = UNJUDGED ]

check "graded defaults: greedy only, eager-anchor tasks, 60 min" [ "$R4E_GRADED_ARGS|$R4E_GRADED_MAX_S|$ORACLE_COMPARE_ARGS" = "--samples 0 --max-task-tokens 9000|3600|--greedy-only" ]
check "oracle gates on the eager anchor only; c1d-final is reported" bash -c '[[ "$1" == eager=*eager/graded/results.json ]] && [[ "$2" == c1d-final=* ]]' _ "$ORACLE_REFS" "$ORACLE_INFO_REFS"
stub="$TMP/stub"; mkdir -p "$stub" "$RUN/graded" "$OUT/graded"
printf '%s\n' 'import sys, os' 'print(" ".join(sys.argv[1:]))' 'print("GRADED status=stub")' 'sys.exit(int(os.environ.get(os.path.basename(sys.argv[3]).split(".")[0].upper() + "_RC", os.environ.get("STUB_RC", "0"))))' > "$stub/graded.py"
echo '{"results": {"a": 1, "b": 2}}' > "$RUN/graded/results.json"; echo '{}' > "$TMP/ref.json"; echo '{}' > "$TMP/info.json"
G="$stub"; ORACLE_REFS="eager=$TMP/ref.json"; ORACLE_INFO_REFS="c1d-final=$TMP/info.json"
STUB_RC=0 oracle_phase; check "oracle PASS on NO_LOSS against the anchor" [ "$ORACLE" = PASS ]
check "compare ran with --greedy-only" grep -q -- '--greedy-only' "$OUT/graded/compare-r4e-vs-eager.txt"
INFO_RC=2 STUB_RC=0 oracle_phase; check "an INCONCLUSIVE info reference does not gate" [ "$ORACLE" = PASS ]
check "the info compare is still said" bash -c 'grep "COMPARE r4e-vs-c1d-final" "$1" | tail -n 1 | grep -q INCONCLUSIVE' _ "$STATUS"
STUB_RC=2 oracle_phase; check "oracle FAIL on INCONCLUSIVE against the anchor" [ "$ORACLE" = FAIL ]
ORACLE_REFS="eager=$TMP/missing.json"
STUB_RC=0 oracle_phase; check "oracle FAIL on a missing reference" [ "$ORACLE" = FAIL ]
check "result_count reads results.json" [ "$(result_count "$RUN/graded/results.json")" = 2 ]

printf '%s\n' '  prompt replay 1024:8192: median=1200 ms p90=1300 n=60' '  prompt replay 1024:1024: median=700 ms p90=760 n=40' > "$TMP/arm_report.txt"
printf '%s\n' '  prompt replay 1024:8192: median=1500 ms p90=1600 n=60' '  prompt replay 1024:1024: median=940 ms p90=1000 n=40' > "$TMP/base_report.txt"
printf '%s\n' 'LOADPATH_PROOF loaded_gb=40 external_hit_pct=80.0 ttft_p50_s=20.0' 'LOADPATH_PROOF loaded_gb=4 external_hit_pct=40.0 ttft_p50_s=30.0' > "$TMP/lp.txt"
lever_gains "$TMP/arm_report.txt" "$TMP/base_report.txt" "$TMP/lp.txt"
check "routed gain: 1024:8192 1200/1500 = 0.80x" bash -c 'grep "GAIN routed" "$1" | tail -n 1 | grep -q "1024:8192=1200/1500ms(0.80x)"' _ "$STATUS"
check "pin gain: hit 80/40 = 2.00x, TTFT 0.67x" bash -c 'grep "GAIN pin" "$1" | tail -n 1 | grep -q "external_hit_pct=80.0/40.0(2.00x).*ttft_p50_s=20.0/30.0(0.67x)"' _ "$STATUS"
lever_gains /dev/null /dev/null /dev/null
check "gains say n/a without reports" bash -c 'grep "GAIN routed" "$1" | tail -n 1 | grep -q "1024:8192=n/a/n/ams(n/a)"' _ "$STATUS"

SELF_DET=PASS ORACLE=PASS SPEED=PASS ROUTED=PASS PIN=PASS; say_verdict
check "verdict PASS when every gate passes" bash -c 'tail -n 1 "$1" | grep -q "VERDICT r4e PASS"' _ "$STATUS"
SPEED=UNJUDGED; say_verdict
check "verdict FAIL when one gate is not PASS" bash -c 'tail -n 1 "$1" | grep -q "VERDICT r4e FAIL"' _ "$STATUS"
# self-det under lever 1 (coordinator 23:0x): rc 0 PASS; rc 2 all 16/16 PASS; rc 2 with diffs UNJUDGED and the run goes on; rc 1 FAIL halts
gate_fixture() {  # FILE LINE...: a det_gate.py output
    local file="$1"; shift
    printf '%s\n' "$@" 'DET_GATE steps matched 50/90 (55.6%)' > "$file"
}
gate_fixture "$TMP/gate-moved-same.txt" 'det-16x1024: steps matched 40/40 IDENTICAL' \
    'det-16x4096: steps matched 5/25 UNJUDGED lineup diverges at step 5; lanes equal 16/16 (informational)' \
    'det-16x8192: steps matched 5/25 UNJUDGED lineup diverges at step 5; lanes equal 16/16 (informational)'
gate_fixture "$TMP/gate-moved-diff.txt" 'det-16x1024: steps matched 40/40 IDENTICAL' \
    'det-16x4096: steps matched 5/25 UNJUDGED lineup diverges at step 5; lanes equal 14/16 (informational)' \
    'det-16x8192: steps matched 5/25 UNJUDGED lineup diverges at step 5; lanes equal 16/16 (informational)'
gate_fixture "$TMP/gate-missing.txt" 'det-16x1024: steps matched 40/40 IDENTICAL' 'det-16x4096: missing in right' \
    'det-16x8192: steps matched 5/25 UNJUDGED lineup diverges at step 5; lanes equal 16/16 (informational)'
gate_fixture "$TMP/gate-two-sets.txt" 'det-16x1024: steps matched 40/40 IDENTICAL' \
    'det-16x4096: steps matched 5/25 UNJUDGED lineup diverges at step 5; lanes equal 16/16 (informational)'
verdict_is() {  # RC FILE WORD NOTE_PART: self_det_verdict sets SELF_DET=WORD and a note holding NOTE_PART
    self_det_verdict "$1" "$2"
    [ "$SELF_DET" = "$3" ] && [[ "$SELF_DET_NOTE" == *"$4"* ]]
}
check "self-det rc 0: PASS, schedule matched" verdict_is 0 "$TMP/gate-moved-same.txt" PASS "schedule matched, outputs identical"
check "self-det rc 2, every set 16/16: PASS, schedule differed, outputs identical" verdict_is 2 "$TMP/gate-moved-same.txt" PASS "schedule differed, outputs identical"
check "self-det rc 2, a set 14/16: UNJUDGED (batch-composition numerics)" verdict_is 2 "$TMP/gate-moved-diff.txt" UNJUDGED "batch-composition numerics"
check "self-det rc 2, a set missing: UNJUDGED, not PASS" verdict_is 2 "$TMP/gate-missing.txt" UNJUDGED "graded greedy decides"
check "self-det rc 2, fewer than DET_SETS sets: UNJUDGED, not PASS" verdict_is 2 "$TMP/gate-two-sets.txt" UNJUDGED "graded greedy decides"
check "self-det rc 1: FAIL" verdict_is 1 "$TMP/gate-moved-same.txt" FAIL "matched schedule"
texts_fixture() {  # DIR LANE5_TEXT: 3 sets of 16 lane texts, lane 5 of det-16x4096 = LANE5_TEXT
    mkdir -p "$1"
    python3 -I -c 'import json,sys; [print(json.dumps({"cell": c, "salt": "s", "texts": [sys.argv[1] if (c, i) == ("det-16x4096", 5) else f"lane {i} text" for i in range(16)]})) for c in ("det-16x1024", "det-16x4096", "det-16x8192")]' "$2" > "$1/texts.jsonl"
}
texts_fixture "$TMP/ref" "The answer is 42, so we stop."
texts_fixture "$TMP/rep" "The answer is 41, so we go on."
check "det_texts diff names the lane, the first divergent char and both sides" bash -c 'out="$(python3 -I "$1/det_texts.py" diff "$2/ref" "$2/rep")"; echo "$out" | grep -q "^DET_DIFF det-16x4096 lane 5: first divergent char 15/29 token n/a ref=\"2, so we stop.\" rep=\"1, so we go on.\"$" && echo "$out" | grep -qx "DET_DIFF_SUMMARY lanes_differing=1 sets=3 \[det-16x4096:5\]"' _ "$KIT" "$TMP"
check "det_texts diff says when no texts were kept" bash -c 'python3 -I "$1/det_texts.py" diff "$2/none-a" "$2/none-b" | grep -q "^DET_DIFF_SUMMARY texts not captured"' _ "$KIT" "$TMP"
check "det_texts records each set's texts and returns them unchanged" bash -c 'python3 -I -c "
import sys, types, json
from pathlib import Path
sys.path.insert(0, sys.argv[1]); import det_texts
fake = types.SimpleNamespace(ANCHOR_TOKENS={1024: 2048, 4096: 262400}, complete=lambda port, prompts, tokens, salt, timeout: [\"a\", \"b\"])
det_texts.record_texts(fake, Path(sys.argv[2]))
assert fake.complete(1, [[0] * 262400, [0]], 4, \"salt-x\", 9) == [\"a\", \"b\"]
assert json.loads(Path(sys.argv[2]).read_text()) == {\"cell\": \"det-16x4096\", \"salt\": \"salt-x\", \"texts\": [\"a\", \"b\"]}
" "$1" "$2/rec.jsonl"' _ "$KIT" "$TMP"
mkdir -p "$TMP/stubcells"
printf '%s\n' 'import det_drive' 'def main(argv):' '    wrapped = det_drive.complete.__qualname__ == "Recorder.complete"' '    return 7 if wrapped and argv == ["--plan", "det"] else 1' > "$TMP/stubcells/drive_cells.py"
check "det_texts drive runs drive_cells.main with its args on the wrapped det_drive (rc passes through)" bash -c 'cd "$3" && python3 -I "$1/det_texts.py" drive "$2/stubcells" "$2/t.jsonl" -- --plan det; [ "$?" = 7 ]' _ "$KIT" "$TMP" "$L"
printf '%s\n' 'import os, sys' 'print(open(os.environ["GATE_OUT"]).read())' 'sys.exit(int(os.environ["GATE_RC"]))' > "$TMP/stubcells/det_gate.py"
phase_with() {  # RC FIXTURE: self_det_phase on stub drives and a stub det_gate.py; its exit code
    det_pass() { mkdir -p "$1"; cp "$TMP/$(basename "$1" | sed 's/det-//')/texts.jsonl" "$1/texts.jsonl"; }
    GATE_RC="$1" GATE_OUT="$2" CELLS="$TMP/stubcells" DOCKER=echo self_det_phase
}
saved_port="${PORT[$QUAD]}"; PORT[$QUAD]=9   # a closed port: the diff never asks a live engine in a test
unjudged_goes_on() {  # self_det_phase returns 0 on UNJUDGED and says the summary and the lane's first divergent token
    phase_with 2 "$TMP/gate-moved-diff.txt" || return 1
    grep -q "SELF_DET arm=r4e UNJUDGED (batch-composition numerics" "$STATUS" \
        && grep -q "SELF_DET_DIFF arm=r4e DET_DIFF_SUMMARY lanes_differing=1 sets=3 \[det-16x4096:5\]" "$STATUS" \
        && grep -q "SELF_DET_DIFF arm=r4e DET_DIFF det-16x4096 lane 5: first divergent char 15/29 token n/a" "$STATUS"
}
same_outputs_pass() {  # det_gate rc 2 with every set 16/16: self_det_phase returns 0 and says PASS with the note
    phase_with 2 "$TMP/gate-moved-same.txt" || return 1
    [ "$(grep "^self_det rc" "$RUN/rc" | tail -n 1)" = "self_det rc=2" ] \
        && tail -n 1 "$STATUS" | grep -q "SELF_DET arm=r4e PASS (schedule differed, outputs identical)"
}
check "self_det_phase: UNJUDGED with diffs goes on (rc 0) and says the diff lane" unjudged_goes_on
check "self_det_phase: rc 2 every set 16/16 is PASS (rc 0)" same_outputs_pass
check "self_det_phase: rc 1 halts (rc 1)" negate phase_with 1 "$TMP/gate-moved-same.txt"
PORT[$QUAD]="$saved_port"
SELF_DET=UNJUDGED SELF_DET_NOTE="batch-composition numerics" ORACLE=PASS SPEED=PASS ROUTED=PASS PIN=PASS; say_verdict
check "verdict PASS: self-det UNJUDGED beside oracle NO_LOSS, note said" bash -c 'tail -n 1 "$1" | grep -q "VERDICT r4e PASS self_det=UNJUDGED (batch-composition numerics) oracle=PASS"' _ "$STATUS"
ORACLE=FAIL; say_verdict
check "verdict FAIL: self-det UNJUDGED without oracle NO_LOSS" bash -c 'tail -n 1 "$1" | grep -q "VERDICT r4e FAIL self_det=UNJUDGED"' _ "$STATUS"
SELF_DET=FAIL ORACLE=PASS; say_verdict
check "verdict FAIL: self-det FAIL" bash -c 'tail -n 1 "$1" | grep -q "VERDICT r4e FAIL self_det=FAIL"' _ "$STATUS"
check "r4e_measure halts on a self-det FAIL only" bash -c 'grep -q "self_det_phase || { say \"HALT" "$1/r4e_run.sh"' _ "$KIT"
check "det_texts compiles" python3 -I -m py_compile "$KIT/det_texts.py"

R4E_NEEDLE_BY=00:00 needle_phase
check "needle phase skipped once past R4E_NEEDLE_BY" bash -c 'tail -n 1 "$1" | grep -q "LONG_NEEDLE arm=r4e skipped: past 00:00"' _ "$STATUS"
# coordinator 02:1x: the 150K needles fold the long-context row into r4e, beside the 33K ones
check "needles default to 33k and 150k" [ "${R4E_NEEDLE_LENGTHS:-}" = 33k,150k ]
check "needle phase passes R4E_NEEDLE_LENGTHS to the probe" grep -q 'PROBE_LENGTHS="$R4E_NEEDLE_LENGTHS"' "$KIT/r4e_run.sh"
check "needle cap fits a 150K prefill (>= 1200 s)" [ "$R4E_NEEDLE_MAX_S" -ge 1200 ]
check "before(): midnight today is already past" negate before 00:00
printf '%s\n' '[{"id": "needle-33k-d10-00", "category": "needle"}, {"id": "needle-150k-d50-00", "category": "needle"}, {"id": "needle-8k-d10-00", "category": "needle"}, {"id": "arith-000", "category": "arithmetic"}]' > "$TMP/tasks.json"
check "long_needle select keeps 33k and 150k only" bash -c 'python3 -I "$1/long_needle.py" select "$2/tasks.json" "$2/long.json" --lengths 33k,150k > /dev/null && python3 -I -c "import json,sys; print(sorted(t[\"id\"] for t in json.load(open(sys.argv[1]))))" "$2/long.json" | grep -qx "\[.needle-150k-d50-00., .needle-33k-d10-00.\]"' _ "$KIT" "$TMP"
printf '%s\n' '{"results": {"needle-33k-d10-00": {"greedy": {"passed": true}}, "needle-150k-d50-00": {"greedy": {"passed": false}}}}' > "$TMP/a.json"
printf '%s\n' '{"results": {"needle-33k-d10-00": {"greedy": {"passed": true}}}}' > "$TMP/b.json"
check "long_needle score per label and all" bash -c 'out="$(python3 -I "$1/long_needle.py" score r0="$2/a.json" r1="$2/b.json" r2="$2/none.json" --tasks "$2/long.json")"; echo "$out" | grep -qx "LONG_NEEDLE r0 greedy passed/finished: 33k=1/1 (of 1) 150k=0/1 (of 1)" && echo "$out" | grep -qx "LONG_NEEDLE all greedy passed/finished: 33k=2/2 (of 3) 150k=0/1 (of 3)"' _ "$KIT" "$TMP"
check "long_needle_probe.sh parses" bash -n "$KIT/long_needle_probe.sh"
check "turn_breakdown compiles" python3 -I -m py_compile "$KIT/turn_breakdown.py"
check "r4e_run.sh parses" bash -n "$KIT/r4e_run.sh"
old_refused() {  # a proof_prefill.sh without the NO_MAIN switch is refused before it is sourced (it would run a proof)
    printf '%s\n' '#!/bin/bash' 'echo SOURCED > "$1/sourced"' > "$TMP/old_proof.sh"
    PROOF_PREFILL_SH="$TMP/old_proof.sh" R4E_RUN_NO_MAIN=1 bash "$KIT/r4e_run.sh" refused B 2> /dev/null
    [ "$?" = 2 ] && [ ! -e "$TMP/refused/sourced" ]
}
check "r4e_run.sh refuses a proof_prefill.sh without PROOF_PREFILL_NO_MAIN" old_refused
# --- base r4e gates (r4e-gates.override.env): the 4 ON keys + lever 1, no routed MoE, the attention-only pin's own lines ---
GATES="$KIT/r4e-gates.override.env"
PIN_PY="${PIN_PY:-$L/pin-kit/glm53_attn_pin.py}"
gates() {  # VAR: its value once r4e_run.sh is sourced with the override file
    bash -c 'source "$1" test-r4e B > /dev/null 2>&1; printf "%s" "${!2}"' _ "$KIT/r4e_run.sh" "$1" 2> /dev/null
}
export R4E_GATES_OVERRIDE="$GATES"
check "override: required keys are the mixed-step key and lever 1 (vector out 00:56, penalty set out 01:45, pin out: r4e-A boot refusal 03:03)" \
    [ "$(gates R4E_REQUIRED_KEYS)" = "VLLM_GAUDI_MIXED_PROMPT_KEYS VLLM_DECODE_PRIORITY" ]
check "r4e-levers.env sets no HPU_GLM53_ATTN_PIN (the pin refuses hash 128 vs scheduler block 1048576; fix proven in r4f)" \
    negate grep -q "^HPU_GLM53_ATTN_PIN=" "$KIT/r4e-levers.env"
check "override: the pin gate is off" [ "$(gates R4E_PIN_GATE)" = off ]
check "r4e-levers.env sets no HPU_GLM53_REASONING_VECTOR (the image .pth then stays uninstalled)" \
    negate grep -q "^HPU_GLM53_REASONING_VECTOR=" "$KIT/r4e-levers.env"
check "r4e-levers.env sets no HPU_GLM53_PENALTY_PROMPT_SET (the penalty overlay then serves stock prompt tensors)" \
    negate grep -q "^HPU_GLM53_PENALTY_PROMPT_SET=" "$KIT/r4e-levers.env"
check "override: no REUSE_V2_FOR and no routed key is required" \
    bash -c '! grep -q -E "REUSE_V2|MIXED_PROMPT_MOE|RAGGED" <<< "$1"' _ "$(gates R4E_REQUIRED_KEYS)"
check "override: the routed gate is off" [ "$(gates R4E_ROUTED_GATE)" = off ]
# The gates file is data, never code (coordinator 17:1x, FINAL_RUN_SH bypass): KEY=VALUE lines only, and no key that picks a script,
# a pin or an allowlist (FINAL_RUN_SH, BAKED_ARM_SHA256S, any *_SH / *_DIR / *_PY / *_SHA path or pin, L, PATH, ...)
gates_file() {  # NAME LINE...: a gates fixture, the base gates file then these lines
    { cat "$GATES"; printf '%s\n' "${@:2}"; } > "$TMP/gates-$1.env"
    echo "$TMP/gates-$1.env"
}
gates_refused() {  # GATES_FILE: sourcing r4e_run.sh with it fails
    ! R4E_GATES_OVERRIDE="$1" bash -c 'source "$1" test-gates B' _ "$KIT/r4e_run.sh" > /dev/null 2>&1
}
gates_value() {  # GATES_FILE VAR: the value r4e_run.sh ends up with
    R4E_GATES_OVERRIDE="$1" bash -c 'source "$1" test-gates B > /dev/null 2>&1 || exit 9; printf "%s" "${!2}"' _ "$KIT/r4e_run.sh" "$2" 2> /dev/null
}
mkdir -p "$TMP/forged"; cp "${FINAL_RUN_SH:-$L/diag-final/final_run.sh}" "$TMP/forged/final_run.sh"   # a working copy: only the key refuses
check "gates: a file that sets FINAL_RUN_SH (to a working copy) is refused" gates_refused "$(gates_file final "FINAL_RUN_SH=$TMP/forged/final_run.sh")"
check "gates: a file that sets BAKED_ARM_SHA256S (array) is refused" \
    gates_refused "$(gates_file allow-array "BAKED_ARM_SHA256S=($(printf '0%.0s' {1..64}))")"
check "gates: a file that sets BAKED_ARM_SHA256S (scalar) is refused" gates_refused "$(gates_file allow-scalar "BAKED_ARM_SHA256S=$(printf '0%.0s' {1..64})")"
for key in L PATH IFS BASH_ENV PROOF_PREFILL_SH REMAINDER_PY TASKS_DIR RUN4_DIR CELLS FINAL_ENV R4E_LEVERS_ENV R4E_GATES_OVERRIDE \
           LOADPATH_SHA LOADPATH_DIR R4E_IMAGE R4E_IMAGE_SHA C1D_IMAGE ARM_CHECKED_KEYS FORBIDDEN_KEYS LD_PRELOAD PYTHONPATH HERE G; do
    check "gates: a file that sets $key is refused" gates_refused "$(gates_file "key-$key" "$key=/tmp/forged")"
done
# Opus round 3: an allowlist of the keys gates files use, not a denylist (a child process obeys GCONV_PATH, NODE_OPTIONS, ...)
for key in GCONV_PATH NODE_OPTIONS DOCKER_HOST LOCPATH PERL5OPT DOCKER_CONFIG TMPDIR PIN_PY ROUTED_EXTRA; do
    check "gates: a key outside the allowlist ($key) is refused" gates_refused "$(gates_file "allow-$key" "$key=/tmp/forged")"
done
check "gates: an R4E_ key no gates file uses (R4E_STOP_AFTER) is refused" gates_refused "$(gates_file stop R4E_STOP_AFTER=self_det)"
check "gates: an allowed key set twice is refused" gates_refused "$(gates_file dup R4E_PIN_GATE=on R4E_PIN_GATE=off)"
# owner 2026-11-14: the level of a request that names none rides the gates file, pinned with the cut like the served length
check "gates: LAB_DEFAULT_REASONING_EFFORT is a gates key and is exported" \
    [ "$(gates_value "$(gates_file effort LAB_DEFAULT_REASONING_EFFORT=high)" LAB_DEFAULT_REASONING_EFFORT)" = high ]
inherited_effort() { LAB_DEFAULT_REASONING_EFFORT=low gates_value "$GATES" LAB_DEFAULT_REASONING_EFFORT; }
check "gates: an inherited LAB_DEFAULT_REASONING_EFFORT does not outlive the cut it was exported for" [ -z "$(inherited_effort)" ]
gates_halt_says() {  # GATES_FILE TEXT: sourcing r4e_run.sh with it fails and says TEXT
    R4E_GATES_OVERRIDE="$1" bash -c 'source "$1" test-gates B' _ "$KIT/r4e_run.sh" 2>&1 >/dev/null | grep -q -F -- "$2"
}
check "gates: the duplicate refusal names the key and both lines" gates_halt_says "$TMP/gates-dup.env" "R4E_PIN_GATE is set twice"
check "gates: r4e_run.sh reads gates through diag-r4e/r4e_gates.sh" grep -q 'source "$R4E_HERE/r4e_gates.sh"' "$KIT/r4e_run.sh"
check "gates: a function definition is refused" gates_refused "$(gates_file func 'sha256sum() { echo forged; }')"
check "gates: a command line is refused and not run" \
    bash -c '! R4E_GATES_OVERRIDE="$1" bash -c "source \"\$2\" test-gates B" _ "$1" "$3" > /dev/null 2>&1 && [ ! -e "$2" ]' \
    _ "$(gates_file cmd "touch $TMP/ran")" "$TMP/ran" "$KIT/r4e_run.sh"
check "gates: a command substitution value is refused and not run" \
    bash -c '! R4E_GATES_OVERRIDE="$1" bash -c "source \"\$2\" test-gates B" _ "$1" "$3" > /dev/null 2>&1 && [ ! -e "$2" ]' \
    _ "$(gates_file subst "R4E_PIN_GATE=\$(touch $TMP/ran2)")" "$TMP/ran2" "$KIT/r4e_run.sh"
check "gates: a double-quoted value is refused" gates_refused "$(gates_file dq 'R4E_PIN_GATE="off"')"
check "gates: a CRLF line is refused" gates_refused "$(gates_file crlf $'R4E_PIN_GATE=off\r')"
check "gates: an indented line is refused" gates_refused "$(gates_file indent '  R4E_PIN_GATE=off')"
ARM_OK="$(gates_file arm "R4E_ARM_ENV='/mnt/data1/declane-out/c1d-r4f/r4f-arm.env'" "R4E_ARM_ENV_SHA256=$(printf 'a%.0s' {1..64})" 'R4E_FORBIDDEN_KEYS=EXTRA_KEY')"
check "gates: R4E_ARM_ENV, R4E_ARM_ENV_SHA256 and R4E_FORBIDDEN_KEYS stay settable" \
    [ "$(gates_value "$ARM_OK" R4E_ARM_ENV)" = /mnt/data1/declane-out/c1d-r4f/r4f-arm.env ]
check "gates: a single-quoted regex value is kept byte for byte" \
    [ "$(gates_value "$GATES" PIN_EVICT_RE)" = "$(sed -n "s/^PIN_EVICT_RE='\(.*\)'$/\1/p" "$GATES")" ]
check "gates: the values are exported (set -a semantics)" \
    [ "$(R4E_GATES_OVERRIDE="$GATES" bash -c 'source "$1" test-gates B > /dev/null 2>&1; bash -c "printf %s \"\$R4E_PIN_GATE\""' _ "$KIT/r4e_run.sh")" = off ]
last_line_only() {  # GATES_FILE OUT: the file with each key's last line only (what bash's source kept)
    tac "$1" | awk -F= '/^[A-Z_][A-Z0-9_]*=/ { if (seen[$1]++) next } { print }' | tac > "$2"
}
n_fixture=0
for gates_fixture in /mnt/data1/graded-out/queue/*/*gates.override.env "$KIT/r4e-gates.override.env"; do
    [ -f "$gates_fixture" ] || continue
    n_fixture=$(( n_fixture + 1 ))
    if [ -n "$(grep -oE '^[A-Z_][A-Z0-9_]*=' "$gates_fixture" | sort | uniq -d)" ]; then   # the r4f / r4ecf runs' gates (run_r4f.sh appended)
        check "gates: $gates_fixture sets a key twice and is refused for that alone" gates_halt_says "$gates_fixture" "is set twice"
        last_line_only "$gates_fixture" "$TMP/dedup-$n_fixture.env"
        gates_fixture="$TMP/dedup-$n_fixture.env"
    fi
    check "gates: the served gates file $gates_fixture reads (every key allowed, every line data)" negate gates_refused "$gates_fixture"
done
check "gates: the queue's gates files were all read (20 at install)" [ "$n_fixture" -ge 20 ]
export R4E_GATES_OVERRIDE=none
check "no override: the routed gate stays on" [ "$(gates R4E_ROUTED_GATE)" = on ]
check "no override: the pin gate stays on" [ "$(gates R4E_PIN_GATE)" = on ]
saved_live="$PIN_LIVE_RE"; saved_hit="$PIN_LOCAL_HIT_RE"; saved_evict="$PIN_EVICT_RE"
set -a; source "$GATES"; set +a
R4E_PIN_GATE=on   # the pin's own line formats stay judged with its gate on (the r4f pin fix runs that way)
# Synthetic boot log in the pin's real print formats (glm53_attn_pin.py emit_armed / maybe_log / the invalid-value refusal)
pin_log="$TMP/pin-container.log"
counter() {  # LOCAL_HIT TRIM PRESSURE CAP: one maybe_log line
    echo "2026-11-09T03:00:0${5:-0}.000000000Z GLM53_ATTN_PIN pinned_sessions=3 pinned_blocks=96 held_plans=1 local_hit_tokens=$1 evicted_adopt_trim=$2 evicted_pressure=$3 evicted_cap=$4 connector_fallbacks=0 adopted=2 lookups=9 pinned=3 errors=0"
}
{
    echo "2026-11-09T03:00:00.000000000Z GLM53_ATTN_PIN armed HPU_GLM53_ATTN_PIN=1 groups=(0,) block_size=128 max_pins=64 log_s=30 k_rule=hit_blocks-1 recurrent=host_sidecar pid=7"
    counter 0 0 0 0 1; counter 4096 0 0 0 2; counter 8192 3 0 0 3; counter 8192 3 2 1 4
} > "$pin_log"
check "pin format: the regex fields are the module's own counter names" \
    bash -c 'for f in pinned_sessions= local_hit_tokens= evicted_adopt_trim= evicted_pressure= evicted_cap= held_plans=; do grep -q -F "$f" "$1" || exit 1; done' _ "$PIN_PY"
check "PIN_LIVE_RE: counter lines with local hits only (3), not the armed line or a zero-hit line" \
    [ "$(grep -a -c -E "$PIN_LIVE_RE" "$pin_log")" = 3 ]
printf '%s\n' 'LOADPATH_PROOF loaded_gb=4 external_hit_pct=60.0' 'LOADPATH_PROOF loaded_gb=40 external_hit_pct=26.2' > "$report"
pin_counters "$pin_log" "$report"; check "pin PASS on the attention-only pin's counter lines" [ "$PIN" = PASS ]
head -n 2 "$pin_log" > "$TMP/pin-zero.log"
pin_counters "$TMP/pin-zero.log" "$report"; check "pin FAIL when the pin never hits locally (armed + zero counters)" [ "$PIN" = FAIL ]
check "PIN_LOCAL_HIT_RE reads local_hit_tokens (group 1) from each counter line" \
    [ "$(python3 -I -c 'import re,sys; print(",".join(m.group(1) for l in open(sys.argv[2]) for m in [re.search(sys.argv[1], l)] if m))' "$PIN_LOCAL_HIT_RE" "$pin_log")" = 0,4096,8192,8192 ]
check "PIN_EVICT_RE matches counter lines with a non-zero eviction only (2)" \
    [ "$(python3 -I -c 'import re,sys; print(sum(bool(re.search(sys.argv[1], l)) for l in open(sys.argv[2])))' "$PIN_EVICT_RE" "$pin_log")" = 2 ]
routed_counters /dev/null; check "routed gate off: ROUTED=OFF, not FAIL" [ "$ROUTED" = OFF ]
SELF_DET=PASS ORACLE=PASS SPEED=PASS PIN=PASS; say_verdict
check "verdict PASS with the routed gate off" bash -c 'tail -n 1 "$1" | grep -q "^.*VERDICT r4e PASS .*routed=OFF"' _ "$STATUS"
R4E_ROUTED_GATE=on; ROUTED=OFF; say_verdict
check "verdict FAIL for ROUTED=OFF while the routed gate is on" bash -c 'tail -n 1 "$1" | grep -q "VERDICT r4e FAIL"' _ "$STATUS"
R4E_ROUTED_GATE=maybe; saved_keys="$R4E_REQUIRED_KEYS"; R4E_REQUIRED_KEYS=""
check "require_levers refuses a routed gate other than on/off" negate require_levers
check "the refusal names R4E_ROUTED_GATE" bash -c 'tail -n 1 "$1" | grep -q "HALT.*R4E_ROUTED_GATE=maybe"' _ "$STATUS"
R4E_REQUIRED_KEYS="$saved_keys"
# pin gate off (coordinator 03:0x): no pin in the cut; PIN=OFF passes only with the gate off, and any pin line is a FAIL
R4E_PIN_GATE=off; pin_counters /dev/null /dev/null; check "pin gate off: PIN=OFF without pin lines" [ "$PIN" = OFF ]
check "pin gate off says why" bash -c 'tail -n 1 "$1" | grep -q "PIN arm=r4e OFF (R4E_PIN_GATE=off"' _ "$STATUS"
pin_counters "$pin_log" "$report"; check "pin gate off: a pin line in the log is a FAIL" [ "$PIN" = FAIL ]
R4E_ROUTED_GATE=off; ROUTED=OFF; SELF_DET=PASS ORACLE=PASS SPEED=PASS PIN=OFF; say_verdict
check "verdict PASS with both gates off" bash -c 'tail -n 1 "$1" | grep -q "VERDICT r4e PASS .*routed=OFF pin=OFF"' _ "$STATUS"
R4E_PIN_GATE=on; say_verdict
check "verdict FAIL for PIN=OFF while the pin gate is on" bash -c 'tail -n 1 "$1" | grep -q "VERDICT r4e FAIL"' _ "$STATUS"
R4E_PIN_GATE=maybe; saved_keys="$R4E_REQUIRED_KEYS"; R4E_REQUIRED_KEYS=""
check "require_levers refuses a pin gate other than on/off" negate require_levers
check "the refusal names R4E_PIN_GATE" bash -c 'tail -n 1 "$1" | grep -q "HALT.*R4E_PIN_GATE=maybe"' _ "$STATUS"
R4E_PIN_GATE=off; saved_args=("${ARGS[@]}"); ARGS+=(--env HPU_GLM53_ATTN_PIN=1)
check "require_levers refuses the pin flag while the pin gate is off" negate require_levers
check "that refusal names HPU_GLM53_ATTN_PIN" bash -c 'tail -n 1 "$1" | grep -q "HALT.*HPU_GLM53_ATTN_PIN.*R4E_PIN_GATE=off"' _ "$STATUS"
ARGS=("${saved_args[@]}"); R4E_REQUIRED_KEYS="$saved_keys"
R4E_PIN_GATE=on; R4E_ROUTED_GATE=on; PIN_LIVE_RE="$saved_live"; PIN_LOCAL_HIT_RE="$saved_hit"; PIN_EVICT_RE="$saved_evict"

# graphs-only holds with the routed gate off too: a lazy fallback line is a FAIL, not OFF (owner rule: graphs only, no lazy path)
printf '%s\n' 'x GLM53_GRAPHS_ONLY keys=1024:8192' 'x lazy fallback taken' > "$TMP/lazy.log"
R4E_ROUTED_GATE=off routed_counters "$TMP/lazy.log"; check "routed gate off still FAILs on a lazy fallback line" [ "$ROUTED" = FAIL ]
check "routed gate off says the lazy line count" bash -c 'tail -n 1 "$1" | grep -q "ROUTED arm=r4e FAIL (R4E_ROUTED_GATE=off.*lazy_lines=1"' _ "$STATUS"
# routed RCA 05:4x: the graph-counter dicts carry 'lazy_fallbacks': 0 on every line; only a nonzero count (or a lazy line) is lazy
printf '%s\n' 'x GLM53_GRAPHS_ONLY keys=1024:8192' "x GLM53_GS counters {'captures': 12, 'lazy_fallbacks': 0, 'replays': 900}" \
    'x GLM53_GS lazy_fallbacks=0 replays=40' > "$TMP/lazy0.log"
R4E_ROUTED_GATE=off routed_counters "$TMP/lazy0.log"; check "routed gate off: lazy_fallbacks 0 counters are not lazy lines (OFF)" [ "$ROUTED" = OFF ]
echo "x GLM53_GS counters {'captures': 12, 'lazy_fallbacks': 3, 'replays': 900}" >> "$TMP/lazy0.log"
R4E_ROUTED_GATE=off routed_counters "$TMP/lazy0.log"; check "routed gate off: a nonzero 'lazy_fallbacks': 3 FAILs" [ "$ROUTED" = FAIL ]
printf '%s\n' 'x GLM53_GS lazy_fallbacks=12 replays=40' > "$TMP/lazy12.log"
R4E_ROUTED_GATE=off routed_counters "$TMP/lazy12.log"; check "routed gate off: lazy_fallbacks=12 FAILs" [ "$ROUTED" = FAIL ]
check "LAZY_RE default is the RCA pattern" [ "$LAZY_RE" = 'lazy fallback|lazy_fallbacks.?[:=] *[1-9]|LAZY_FALLBACK' ]

# key256 (coordinator 00:4x): mixed-step/key256_gate.py on the run, the given arguments, its rc and last line said; reported, not gating
key256_report
check "key256 skipped without the arm's report.txt" bash -c 'tail -n 1 "$1" | grep -q "KEY256 arm=r4e skipped: no graded/r4e/report.txt"' _ "$STATUS"
mkdir -p "$TMP/k256/mixed-step" "$RUN/medium"; echo report > "$RUN/report.txt"
printf '%s\n' 'import sys' 'print("args", " ".join(sys.argv[1:]))' 'print("KEY256_GATE GO stub")' 'sys.exit(3)' > "$TMP/k256/mixed-step/key256_gate.py"
# r4e-A2 05:25 (rc=1 FileNotFoundError): the medium profile writes no medium/report.txt, so only inputs that exist are passed
L="$TMP/k256" key256_report
check "key256 without medium/report.txt passes the log and report only" bash -c 'tail -n 1 "$1" | grep -q "KEY256 arm=r4e rc=3 KEY256_GATE GO stub" && grep -qx "args $2/container.log $2/report.txt --baseline-report $3/report.txt" "$2/key256_gate.txt"' _ "$STATUS" "$RUN" "$BASELINE_RUN"
echo medium > "$RUN/medium/report.txt"
L="$TMP/k256" key256_report
check "key256 runs the given command and says rc + last line" bash -c 'tail -n 1 "$1" | grep -q "KEY256 arm=r4e rc=3 KEY256_GATE GO stub" && grep -qx "args $2/container.log $2/report.txt $2/medium/report.txt --baseline-report $3/report.txt" "$2/key256_gate.txt"' _ "$STATUS" "$RUN" "$BASELINE_RUN"
mv "$RUN/medium/report.txt" "$RUN/medium/report.txt.k256-fixture"
mv "$RUN/report.txt" "$RUN/report.txt.k256-fixture"
check "r4e_main calls key256_report after the reports" bash -c 'sed -n "/^r4e_main() {/,/^}/p" "$1/r4e_run.sh" | grep -A1 "reports \"\$ARM\" \"\$RUN\"" | grep -q "^    key256_report$"' _ "$KIT"

# remainder census (coordinator 02:1x; MIXED_STEP_R4E_PLAN.md section 8): 256 vs 384 for r4f, reported, not gating
mv "$RUN/container.log" "$RUN/container.log.rem-saved" 2> /dev/null
remainder_report
check "remainder census skipped without the arm's container.log" bash -c 'tail -n 1 "$1" | grep -q "REMAINDER_HIST arm=r4e skipped: no graded/r4e/container.log"' _ "$STATUS"
echo log > "$RUN/container.log"; echo '{}' > "$RUN/medium/turns.jsonl"
printf '%s\n' 'import sys' 'print("REMAINDER_HIST stub", " ".join(sys.argv[1:]))' 'sys.exit(1)' > "$TMP/rem_stub.py"
REMAINDER_PY="$TMP/rem_stub.py" remainder_report
check "remainder census runs the whole log and the medium window, each line said with its rc" bash -c 'grep -q "REMAINDER_HIST arm=r4e rc=1 REMAINDER_HIST stub $2/container.log --scope $3-log$" "$1" && tail -n 1 "$1" | grep -q "REMAINDER_HIST arm=r4e rc=1 REMAINDER_HIST stub $2/container.log --turns $2/medium/turns.jsonl --scope $3-medium$"' _ "$STATUS" "$RUN" r4e
rm -f "$RUN/container.log" "$RUN/medium/turns.jsonl"; mv "$RUN/container.log.rem-saved" "$RUN/container.log" 2> /dev/null
check "r4e_main calls remainder_report right after key256_report" bash -c 'sed -n "/^r4e_main() {/,/^}/p" "$1/r4e_run.sh" | grep -A1 "^    key256_report$" | grep -q "^    remainder_report$"' _ "$KIT"
# r4f-B2 self-det bisect (coordinator 07:0x): R4E_STOP_AFTER=self_det ends the measure after self-det and its parity, before graded
measure_calls() {  # R4E_STOP_AFTER value: the phases r4e_measure reaches with every phase stubbed
    ( R4E_STOP_AFTER="$1"
      for phase in r4e_parity self_det_phase engine_alive r4e_graded_phase oracle_phase needle_phase profile_medium long_leg_phase; do
          eval "$phase() { echo $phase >> \"\$TMP/measure-calls\"; return 0; }"
      done
      : > "$TMP/measure-calls"; r4e_measure > /dev/null 2>&1; echo "rc=$?"; paste -sd' ' "$TMP/measure-calls" )   # in call order
}
check "R4E_STOP_AFTER=self_det stops before graded (rc 0)" bash -c '[[ "$1" == "rc=0"*"self_det_phase"* ]] && [[ "$1" != *r4e_graded_phase* ]] && [[ "$1" != *profile_medium* ]]' _ "$(measure_calls self_det | paste -sd' ')"
check "R4E_STOP_AFTER=self_det says so in the status" bash -c 'grep -q "STOP_AFTER arm=r4e self_det" "$1"' _ "$STATUS"
check "R4E_STOP_AFTER unset runs graded and medium" bash -c '[[ "$1" == *r4e_graded_phase*profile_medium* ]]' _ "$(measure_calls "" | paste -sd' ')"
check "R4E_STOP_AFTER refuses another value" bash -c '! R4E_STOP_AFTER=graded bash -c "source \"\$1\" test-stop A" _ "$1" > /dev/null 2>&1' _ "$KIT/r4e_run.sh"
check "remainder_hist tests pass" python3 -I "$KIT/tests/test_remainder_hist.py"
# Opus round 4 NITs: a gates value that starts with '-' is a pattern, never a grep option; R4E_REQUIRED_KEYS is split, never globbed
printf '%s\n' 'x -v y' 'plain' > "$TMP/dash.log"
pin_dash() { ( PIN_LIVE_RE='-v' R4E_PIN_GATE=off; pin_counters "$TMP/dash.log" /dev/null > /dev/null 2>&1; echo "$PIN" ) < /dev/null; }
check "PIN_LIVE_RE='-v' is a pattern (one live line, so FAIL with the pin gate off), not grep's -v" [ "$(pin_dash)" = FAIL ]
printf '%s\n' 'boot' '-ROUTED -p=5 -o=0 -r=1' > "$TMP/dash-routed.log"
routed_dash() {
    ( ROUTED_LINE_RE='-ROUTED' ROUTED_PIECES_KEY='-p' ROUTED_OVERFLOWS_KEY='-o' ROUTED_RERUNS_KEY='-r' ROUTED_MAX_OVERFLOW_PCT=0 R4E_ROUTED_GATE=on
      routed_counters "$TMP/dash-routed.log" > /dev/null 2>&1; echo "$ROUTED" ) < /dev/null
}
check "ROUTED_LINE_RE and the ROUTED_*_KEY fields starting with '-' are patterns (PASS on the dash line)" [ "$(routed_dash)" = PASS ]
check "turn_breakdown.py gets the pin regexes as --opt=VALUE (a '-' value is not an option)" \
    bash -c 'grep -q -F -- "--local-hit-re=\"\$PIN_LOCAL_HIT_RE\" --evict-re=\"\$PIN_EVICT_RE\"" "$1"' _ "$KIT/r4e_run.sh"
star_halt() { ( cd "$TMP" && R4E_REQUIRED_KEYS='*'; require_levers > /dev/null 2>&1; tail -n 1 "$STATUS" ) | grep -q -F -- ": * is not set"; }
check "R4E_REQUIRED_KEYS='*' is the literal key '*' (refused by name), not a glob of the run dir" star_halt


# 2026-11-10 (r4g-h6cF self_det FAIL, lane 11's first token, no margin on file): a FAIL at a matched schedule also gets the lane diff, and
# logprobs ride only behind R4E_DET_LOGPROBS / R4E_DET_MARGIN_ON_FAIL (off: asking compiles a logprobs sampler graph while serving)
phase_recording() {  # RC FIXTURE: phase_with, its det_pass stub also noting each call's dir and logprobs argument in $TMP/det-calls
    : > "$TMP/det-calls"; : > "$STATUS"
    det_pass() { echo "$(basename "$1") ${2:-unset}" >> "$TMP/det-calls"; mkdir -p "$1"; cp "$TMP/$(basename "$1" | sed 's/det-//')/texts.jsonl" "$1/texts.jsonl"; }
    GATE_RC="$1" GATE_OUT="$2" CELLS="$TMP/stubcells" DOCKER=echo self_det_phase
}
det_calls_are() { [ "$(paste -sd, "$TMP/det-calls")" = "$1" ]; }
mkdir -p "$TMP/margin"
python3 -I -c 'import json
lane = {"tokens": ["The", " answer", " is", " 42", ",", " so", " we", " stop."], "token_logprobs": [-0.1, -0.1, -0.1, -0.6931, -0.1, -0.1, -0.1, -0.1],
        "top_logprobs": [{}, {}, {}, {" 42": -0.6931, " 41": -0.6932, " 7": -9.0}, {}, {}, {}, {}], "text_offset": [0, 3, 10, 13, 16, 17, 20, 23]}
for cell in ("det-16x1024", "det-16x4096", "det-16x8192"):
    texts = ["The answer is 42, so we stop." if (cell, i) == ("det-16x4096", 5) else f"lane {i} text" for i in range(16)]
    print(json.dumps({"cell": cell, "salt": "s", "texts": texts, "logprobs": [lane if (cell, i) == ("det-16x4096", 5) else None for i in range(16)]}))
' > "$TMP/margin/texts.jsonl"
saved_port="${PORT[$QUAD]}"; PORT[$QUAD]=9   # a closed port: the diff never asks a live engine in a test
fail_says_diff() {  # det_gate rc 1: the phase halts (rc 1), says the lane diff and that no margin is on file, and drives nothing more
    rm -rf "$RUN/det-margin"
    phase_recording 1 "$TMP/gate-moved-same.txt" && return 1
    det_calls_are "det-ref 0,det-rep 0" && [ ! -e "$RUN/det-margin" ] \
        && grep -q "SELF_DET arm=r4e FAIL (a lane differs at a matched schedule)" "$STATUS" \
        && grep -q "SELF_DET_DIFF arm=r4e DET_DIFF det-16x4096 lane 5: first divergent char 15/29 token n/a" "$STATUS" \
        && tail -n 1 "$STATUS" | grep -q "SELF_DET_MARGIN arm=r4e DET_MARGIN det-16x4096 lane 5: n/a, no logprobs were recorded.*R4E_DET_MARGIN_ON_FAIL=5" \
        && grep -q "^DET_MARGIN det-16x4096 lane 5: n/a" "$RUN/self_det_diff.txt"
}
margin_on_fail() {  # R4E_DET_MARGIN_ON_FAIL=5: after the judged passes, one more drive with logprobs 5; its margin is said and kept
    R4E_DET_MARGIN_ON_FAIL=5 phase_recording 1 "$TMP/gate-moved-same.txt" && return 1
    det_calls_are "det-ref 0,det-rep 0,det-margin 5" \
        && grep -q "SELF_DET_MARGIN arm=r4e one more det drive with logprobs=5, not judged" "$STATUS" \
        && tail -n 1 "$STATUS" | grep -q -F 'SELF_DET_MARGIN arm=r4e DET_MARGIN det-16x4096 lane 5 pass=margin token 3/8: chose " 42" logprob=-0.693100 margin_nats=0.000100 (top1-top2) top5=[[" 42", -0.6931], [" 41", -0.6932], [" 7", -9.0]]' \
        && grep -q "^DET_MARGIN det-16x4096 lane 5 pass=margin token 3/8" "$RUN/self_det_diff.txt"
}
unjudged_never_drives_margin() { R4E_DET_MARGIN_ON_FAIL=5 phase_recording 2 "$TMP/gate-moved-diff.txt" && det_calls_are "det-ref 0,det-rep 0"; }
crash_never_drives_margin() { R4E_DET_MARGIN_ON_FAIL=5 phase_recording 3 "$TMP/gate-moved-same.txt" && return 1; det_calls_are "det-ref 0,det-rep 0"; }
judged_logprobs() {  # R4E_DET_LOGPROBS=5: both judged passes ask, and the status says what that costs
    R4E_DET_LOGPROBS=5 phase_recording 2 "$TMP/gate-moved-same.txt" || return 1
    det_calls_are "det-ref 5,det-rep 5" && grep -q "SELF_DET_LOGPROBS arm=r4e both judged passes ask for logprobs=5: the logprobs sampler graphs compile while serving" "$STATUS"
}
check "self_det_phase: a FAIL says the lane diff and that no margin was recorded; no extra drive by default" fail_says_diff
check "self_det_phase: R4E_DET_MARGIN_ON_FAIL=5 drives det-margin after a FAIL and says the margin" margin_on_fail
check "self_det_phase: UNJUDGED never drives the margin pass" unjudged_never_drives_margin
check "self_det_phase: a det_gate.py crash (rc 3) never drives the margin pass" crash_never_drives_margin
check "self_det_phase: R4E_DET_LOGPROBS=5 asks in both judged passes and says so" judged_logprobs
PORT[$QUAD]="$saved_port"
check "both logprobs switches default to off" [ "${R4E_DET_LOGPROBS:-unset}/${R4E_DET_MARGIN_ON_FAIL:-unset}" = 0/0 ]
switch_refused() {  # NAME VALUE: sourcing r4e_run.sh with the switch at VALUE fails and names it
    ( export "$1=$2"; source "$KIT/r4e_run.sh" test-r4e B ) 2>&1 | grep -q "HALT $1=.*: use 0 (off) or 1..20"
}
check "R4E_DET_LOGPROBS=on is refused" switch_refused R4E_DET_LOGPROBS on
check "R4E_DET_MARGIN_ON_FAIL=21 is refused" switch_refused R4E_DET_MARGIN_ON_FAIL 21
mkdir -p "$TMP/envcells"
printf '%s\n' 'import os' 'def main(argv):' '    seen = os.path.join(os.path.dirname(argv[argv.index("--out") + 1]), "seen")' \
    '    open(seen, "w").write(os.environ.get("DET_LOGPROBS", "unset"))' '    return 0' > "$TMP/envcells/drive_cells.py"
det_pass_env() {  # [LOGPROBS]: the real det_pass (r4e_run.sh sourced afresh) on a stub drive_cells; prints the DET_LOGPROBS the drive saw
    ( source "$KIT/r4e_run.sh" test-r4e B > /dev/null 2>&1; mkdir -p "$RUN"; CELLS="$TMP/envcells"; DET_MAX_S=60
      det_pass "$TMP/envpass" "$@" > /dev/null 2>&1; cat "$TMP/envpass/seen" 2> /dev/null )
}
check "det_pass: without a logprobs argument the drive runs with DET_LOGPROBS=0" [ "$(det_pass_env)" = 0 ]
check "det_pass: its logprobs argument reaches the drive as DET_LOGPROBS" [ "$(det_pass_env 5)" = 5 ]

# 2026-11-10 teardown: serve_compile_attr.py (the real tool) runs before the down; the recipe leaf outlives the down only when that
# attribution failed on a run that failed so far or compiled while serving. Stub docker, lab.py and hl-smi; no card, no container.
RUN="$TMP/teardown-run"; LAB="$TMP/lab"; mkdir -p "$RUN" "$LAB" "$TMP/leaf"; echo recipe > "$TMP/leaf/one.recipe"
printf '%s\n' '#!/bin/bash' 'case "$1" in' "    inspect) [ -f '$TMP/container-exists' ] ;;" "    logs) cat '$TMP/boot.log' ;;" 'esac' > "$TMP/docker-stub"
printf '%s\n' '#!/bin/bash' 'for module in 0 1 2 3 4 5 6 7; do echo "$module, 0"; done' > "$TMP/hlsmi-stub"
printf '%s\n' 'import sys' "open('$TMP/labpy-args', 'w').write(' '.join(sys.argv[1:]))" "print('LAB_DOWN stub')" > "$TMP/labpy-stub.py"
chmod +x "$TMP/docker-stub" "$TMP/hlsmi-stub"; touch "$TMP/container-exists"
printf '{"recipe_leaf": "%s"}\n' "$TMP/leaf" > "$LAB/state-$NAME.json"
boot_log() {  # N0 N1: a boot log whose ranks 0 and 1 compiled N0 and N1 graphs while serving (two windows each; the last total counts)
    printf '%s\n' "2026-11-10T04:20:00.000000000Z (Worker_TP0_EP0 pid=300) WARNING GLM53_SERVE_COMPILES n=0 ms=0.0 window_s=60 total=0 pid=300" \
        "2026-11-10T04:21:00.000000000Z (Worker_TP0_EP0 pid=300) WARNING GLM53_SERVE_COMPILES n=$1 ms=10.0 window_s=60 total=$1 pid=300" \
        "2026-11-10T04:21:00.000000000Z (Worker_TP1_EP1 pid=301) WARNING GLM53_SERVE_COMPILES n=$2 ms=10.0 window_s=60 total=$2 pid=301" > "$TMP/boot.log"
}
served() { echo 'LAB_READY {"started_utc": "2026-11-10T04:00:00+00:00", "startup_ready_s": 900}' > "$RUN/up.log"; }
teardown_case() {  # MEASURED_RC SELF_DET ORACLE: attribute_serve on the stubs; prints LAB_KEEP_LEAF (0 = empty)
    : > "$STATUS"; rm -f "$RUN/serve_compile_attr.txt"
    SELF_DET="$2" ORACLE="$3" DOCKER="$TMP/docker-stub" attribute_serve "$1"
    echo "${LAB_KEEP_LEAF:-0}"
}
boot_log 4 3
check "serve_compile_total sums each rank's last total" [ "$(serve_compile_total "$TMP/boot.log")" = 7 ]
check "serve_compile_total says nothing for a log it cannot read" [ -z "$(serve_compile_total "$TMP/no-such.log")" ]
served
check "teardown: attributed with serve compiles, the leaf goes (keep_leaf=0)" [ "$(teardown_case 0 PASS PASS)" = 0 ]
check "teardown: the attribution is stored in the run dir and said" bash -c 'tail -n 1 "$1/serve_compile_attr.txt" | grep -qx "SERVE_COMPILE_ATTR ranks=2 serve_compiles_total=7 attributed=0 medium=no" && grep -q "SERVE_ATTR arm=r4e rc=0 serve_compiles_total=7 failed_so_far=no keep_leaf=0 SERVE_COMPILE_ATTR ranks=2 serve_compiles_total=7" "$2" && grep -qx "serve_attr rc=0" "$1/rc"' _ "$RUN" "$STATUS"
check "teardown: attributed after a self-det FAIL, the leaf goes" [ "$(teardown_case 1 FAIL none)" = 0 ]
rm -f "$RUN/up.log"   # the attribution cannot run: no LAB_READY
check "teardown: not attributed with serve compiles, the leaf stays" [ "$(teardown_case 0 PASS PASS)" = 1 ]
check "teardown: that is said with the tool's reason" grep -q "SERVE_ATTR arm=r4e rc=1 serve_compiles_total=7 failed_so_far=no keep_leaf=1 serve_compile_attr: .*up.log has no LAB_READY line" "$STATUS"
boot_log 0 0
check "teardown: not attributed, no compiles, self-det FAIL: the leaf stays" [ "$(teardown_case 1 FAIL none)" = 1 ]
check "teardown: not attributed, no compiles, oracle FAIL: the leaf stays" [ "$(teardown_case 0 PASS FAIL)" = 1 ]
check "teardown: not attributed, no compiles, the engine died (measure rc 1): the leaf stays" [ "$(teardown_case 1 PASS PASS)" = 1 ]
check "teardown: not attributed, no compiles, nothing failed: the leaf goes" [ "$(teardown_case 0 PASS PASS)" = 0 ]
: > "$TMP/boot.log"
check "teardown: not attributed and no compile counter in the log (0 counted), nothing failed: the leaf goes" [ "$(teardown_case 0 UNJUDGED PASS)" = 0 ]
rm -f "$TMP/container-exists"; boot_log 4 3
check "teardown: no container (nothing served): no attribution, the leaf goes" bash -c '[ "$1" = 0 ] && [ ! -s "$2" ] && [ ! -e "$3/serve_compile_attr.txt" ]' _ "$(teardown_case 1 FAIL none)" "$STATUS" "$RUN"
touch "$TMP/container-exists"
finish_case() {  # MEASURED_RC: r4e_finish on the stubs (lab dir $LAB, stub lab.py and hl-smi); its rc
    : > "$STATUS"; rm -f "$TMP/labpy-args"
    L="$LAB" LABPY="$TMP/labpy-stub.py" HLSMI="$TMP/hlsmi-stub" DOCKER="$TMP/docker-stub" HOLD="" WATCH="" SELF_DET=FAIL ORACLE=none r4e_finish "$1"
}
kept_through_the_down() {  # not attributed: lab.py down --keep-leaf, the leaf's path and size said after RELEASED, the flag cleared
    finish_case 1 || return 1
    [ "$(cat "$TMP/labpy-args")" = "down --name $NAME --keep-leaf" ] && [ -z "$LAB_KEEP_LEAF" ] && [ -f "$TMP/leaf/one.recipe" ] \
        && grep -q "RELEASED quad=$QUAD down_rc=0" "$STATUS" \
        && tail -n 1 "$STATUS" | grep -q "LEAF_KEPT arm=r4e path=$TMP/leaf size=[0-9.]*[KMG] serve_compile_attr.py did not run .*then delete the leaf directory"
}
removed_by_the_down() {  # attributed: lab.py down as before (no --keep-leaf), no LEAF_KEPT line, the attribution on file
    served; finish_case 1 || return 1
    [ "$(cat "$TMP/labpy-args")" = "down --name $NAME" ] && negate grep -q LEAF_KEPT "$STATUS" \
        && grep -q "^SERVE_COMPILE_ATTR ranks=2 serve_compiles_total=7" "$RUN/serve_compile_attr.txt" && grep -qx "down rc=0" "$RUN/rc"
}
check "r4e_finish: a run that could not be attributed keeps its leaf through the down and says path and size" kept_through_the_down
check "r4e_finish: an attributed run's leaf is removed by the down as before" removed_by_the_down
plain_down() { LAB_KEEP_LEAF="" L="$LAB" LABPY="$TMP/labpy-stub.py" HLSMI="$TMP/hlsmi-stub" DOCKER="$TMP/docker-stub" WATCH="" bring_down "$NAME" "$QUAD" "$RUN" && [ "$(cat "$TMP/labpy-args")" = "down --name $NAME" ]; }
check "bring_down without LAB_KEEP_LEAF is lab.py down as before" plain_down
# Fable review 2026-11-10 (1): no pre-down step may hold the down and the quad release; each is cut after R4E_TEARDOWN_STEP_S
printf '%s\n' 'import time' 'time.sleep(600)' > "$TMP/attr-hangs.py"
printf '%s\n' '#!/bin/bash' 'case "$1" in' "    inspect) [ -f '$TMP/container-exists' ] ;;" \
    "    logs) [ -e '$TMP/logs-hung' ] || { touch '$TMP/logs-hung'; exec sleep 600; }; cat '$TMP/boot.log' ;;" 'esac' > "$TMP/docker-logs-hang"   # only its first logs call hangs: bring_down's own is not under test
chmod +x "$TMP/docker-logs-hang"
hung_finish() {  # DOCKER_STUB ATTR_PY SAID: r4e_finish with 1 s steps ends within 15 s, still downs and releases, and says which step was cut
    local began=$SECONDS
    : > "$STATUS"; rm -f "$TMP/labpy-args"
    R4E_TEARDOWN_STEP_S=1 ATTR_PY="$2" L="$LAB" LABPY="$TMP/labpy-stub.py" HLSMI="$TMP/hlsmi-stub" DOCKER="$1" HOLD="" WATCH="" SELF_DET=FAIL ORACLE=none r4e_finish 1 || return 1
    [ $(( SECONDS - began )) -le 15 ] && grep -q "^down --name $NAME" "$TMP/labpy-args" && grep -q "RELEASED quad=$QUAD down_rc=0" "$STATUS" \
        && grep -q "TEARDOWN_TIMEOUT arm=r4e $3.* did not end within 1s (R4E_TEARDOWN_STEP_S) and was cut: the down and the quad release go ahead" "$STATUS"
}
hung_attribution() { hung_finish "$TMP/docker-stub" "$TMP/attr-hangs.py" "serve_compile_attr.py" && grep -q -- "--keep-leaf" "$TMP/labpy-args"; }
check "r4e_finish: an attribution that never ends is cut, said, and the down goes ahead (leaf kept: not attributed)" hung_attribution
check "r4e_finish: a docker logs that never ends is cut, said, and the down goes ahead" hung_finish "$TMP/docker-logs-hang" "$ATTR_PY" "docker logs $NAME"
check "R4E_TEARDOWN_STEP_S defaults to 120" [ "$R4E_TEARDOWN_STEP_S" = 120 ]
check "R4E_TEARDOWN_STEP_S=0 is refused" bash -c '( export R4E_TEARDOWN_STEP_S=0; source "$1/r4e_run.sh" test-r4e B ) 2>&1 | grep -q "HALT R4E_TEARDOWN_STEP_S=.*: use whole seconds above 0"' _ "$KIT"
check "r4e_main tears down through r4e_finish, on the measured rc and on TERM" bash -c 'grep -q "^    r4e_finish \"\$measured\"$" "$1/r4e_run.sh" && grep -q "bringing the boot down\"; r4e_finish 143; say" "$1/r4e_run.sh"' _ "$KIT"
echo "r4e_args: $(( total - fails ))/$total ok"
[ "$fails" = 0 ]
