#!/bin/bash
# diag-r4e/r4e_run.sh: the r4e candidate arm on quad B, one boot (coordinator 2026-11-08 22:1x; owner: the routing fix and the re-read fix are in
# the 08:00 champion). r4e = r4d + routed MoE (ROUTED_MOE_C1D_PLAN S9, ragged two-tier prompt MoE) + the session pin (HPU_GLM53_REUSE_V2_FOR).
# Reuses diag-final/final_run.sh (config, static guard, quad take, boot, parity, graded, down) and diag-prefill/proof_prefill.sh MEDIUM=1
# (profile_medium, reports), both sourced with their NO_MAIN switches. Gates, each said to the status file the moment it lands:
#   CONFIG     final_args (final-config.env verbatim + the V2 block) + every line of R4E_LEVERS_ENV; no overlay mount (r4e bakes glm53_loadpath.py
#              23036c88, checked in the image); the image id must start with R4E_IMAGE_SHA; every R4E_REQUIRED_KEYS key set; forbid() holds
#   SELF_DET   drive_cells.py --plan det twice on this boot (det-ref, det-rep) and det_gate.py det-ref det-rep: bitwise across its own replays
#              (not vs c1d-all: routed MoE changes numerics by design). det_drive.py seeds prompts apart from the cache salt and salts every set
#              fresh, so pass 2 replays the same prompts with no cache hit. det_texts.py keeps the lane texts beside cells.jsonl.
#              Coordinator 23:0x, lever 1's 8 s wall-clock prefill wait can move the lineup between passes: det_gate rc 0 -> PASS;
#              rc 2 with every set 16/16 equal -> PASS ("schedule differed, outputs identical"); rc 2 with lane diffs -> UNJUDGED
#              (batch-composition numerics), the run continues, the diffs (lanes, first divergent token) are said and graded decides;
#              rc 1 (a lane differs at a matched schedule) or a failed drive -> FAIL -> HALT, down, release. A FAIL also gets the
#              lane diff (self_det_diff.txt: DET_DIFF + DET_MARGIN lines). Logprobs are NOT host-side on this server (det_texts.py header:
#              the sampler runs log_softmax + top-k on the device logits, a sampler graph no warmup covers), so both switches are off
#              unless set: R4E_DET_LOGPROBS=N asks in both judged passes; R4E_DET_MARGIN_ON_FAIL=N asks only in one more drive after
#              a FAIL at a matched schedule (det-margin/, not judged: the judged passes and their graphs stay as they are).
#   TEARDOWN   r4e_finish: before the down (lab.py down removes the boot's recipe leaf) serve_compile_attr.py runs on the boot log into
#              graded/ARM/serve_compile_attr.txt. The leaf is removed as before unless that attribution failed on a run that failed so
#              far or compiled while serving: then bring_down keeps it (LAB_KEEP_LEAF) and LEAF_KEPT says its path and size.
#              Each step before the down (docker inspect, docker logs, the attribution) is cut after R4E_TEARDOWN_STEP_S (120 s):
#              a hang is said and the down and the quad release go ahead.
#   ORACLE     graded.py run on run2's calibrated tasks, greedy only, the tasks the eager anchor has (--samples 0 --max-task-tokens 9000: every
#              eager task is <= 8,205 tokens; coordinator 22:2x), 60 min cap in all (a --resume pass gets what is left); graded.py compare
#              --greedy-only against each ORACLE_REFS entry (run2 eager, 182 greedy): PASS = NO_LOSS (exit 0) against every one (NO_LOSS needs
#              >= 100 paired greedy tasks, gstats MIN_PAIRED_GREEDY); ORACLE_INFO_REFS (tonight's c1d-final) are compared and reported, not gating
#   MEDIUM     proof_prefill.sh profile_medium + reports against BASELINE_RUN (tonight's medium baseline on B)
#   SPEED      lane_report.py ratios vs that baseline: decode p50 >= SPEED_MIN_DECODE, TTFT p50 <= SPEED_MAX_TTFT, prefill >= SPEED_MIN_PREFILL
#   COUNTERS   routed MoE: the last ROUTED_LINE_RE line has pieces > 0 and overflows <= ROUTED_MAX_OVERFLOW_PCT % of pieces, 0 LAZY_RE lines;
#              session pin: PIN_LIVE_RE lines > 0 and the medium run's external hit % above the baseline's (loadpath_report LOADPATH_PROOF)
#   GAIN       each lever against its own counter, vs the baseline: routed MoE = prompt replay median ms per key (arm_report REPLAY);
#              session pin = external hit %, loaded GB, TTFT p50 (loadpath_report LOADPATH_PROOF). Reported, not gating.
#   VERDICT    r4e PASS only when ORACLE, SPEED, ROUTED and PIN all PASS and SELF_DET is PASS (or UNJUDGED, which the graded greedy
#              NO_LOSS then decides); R4E_ROUTED_GATE=off makes ROUTED=OFF (0 lazy lines) count as passed; the quad is released either way
#   KEY256     mixed-step/key256_gate.py on this run vs the medium baseline (coordinator 00:4x): its rc and last line, reported, not gating
# Output: /mnt/data1/graded-out/queue/NAME/{status,static-guard.txt,graded/r4e/{rc,up.log,container.log,armed-parity-*.txt,det-ref/,det-rep/,
#         self_det.txt,graded/results.json,medium/,loadpath_report.txt,lane_report.txt,steps_during_loads.txt},graded/compare-r4e-vs-*.txt}
# usage: R4E_IMAGE=claude-flash-dual-graphs-c1d:<tag> R4E_IMAGE_SHA=sha256:<12+ hex> \
#          setsid nohup bash r4e_run.sh r4e-B B < /dev/null > /mnt/data1/graded-out/queue/r4e-B.console 2>&1 &
#        tail -f /mnt/data1/graded-out/queue/r4e-B/status          STOP: kill -TERM <START pid> (down + release, DONE verdict=STOPPED)
#        R4E_RUN_NO_MAIN=1 source r4e_run.sh NAME [B]   (tests: functions only)
set -u
R4E_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The cut's gate settings (base r4e: r4e-gates.override.env), read before every default below so its values win; none = defaults.
# The file is data, never code: diag-r4e/r4e_gates.sh's r4e_read_gates (KEY=VALUE lines, an allowlist of keys, each key once).
# The served length rides the gates file only (reviewer B 1a): an inherited LAB_MAX_MODEL_LEN would outlive the cut it was
# exported for and rewrite this boot's --max-model-len beside another cut's key list. Unset here, set again by the gates read.
# The same holds for the level of a request that names none (LAB_DEFAULT_REASONING_EFFORT, owner 2026-11-14).
unset LAB_MAX_MODEL_LEN LAB_DEFAULT_REASONING_EFFORT
# shellcheck source=r4e_gates.sh
source "$R4E_HERE/r4e_gates.sh"
R4E_GATES_OVERRIDE="${R4E_GATES_OVERRIDE:-$R4E_HERE/r4e-gates.override.env}"
if [ "$R4E_GATES_OVERRIDE" != none ] && [ -f "$R4E_GATES_OVERRIDE" ]; then
    r4e_read_gates "$R4E_GATES_OVERRIDE" || return 2 2> /dev/null || exit 2
fi
R4E_NAME_ARG="${1:?usage: r4e_run.sh NAME [B|A]}"
R4E_QUAD_ARG="${2:-B}"
L="${L:-$HOME/.codex/tmp/flash-spikes-20261104/team-20261106-dual}"
R4E_OUT="${OUT_ROOT:-/mnt/data1/graded-out/queue}/$R4E_NAME_ARG"
PROOF_PREFILL_SH="${PROOF_PREFILL_SH:-$L/diag-prefill/proof_prefill.sh}"
FINAL_RUN_SH="${FINAL_RUN_SH:-$L/diag-final/final_run.sh}"
export FREE_FOR_S="${FREE_FOR_S:-240}"

# proof_prefill.sh's settings the medium profile and its reports read (the baseline run's own values, prefill-proof1 MEDIUM=1 22:10)
BASELINE_RUN="${BASELINE_RUN:-/mnt/data1/claude-flash-cells/prefill-proof1/medium}"
BASELINE_LABEL="${BASELINE_LABEL:-medium-baseline-2210}"
PROFILE_MAX_S="${PROFILE_MAX_S:-2400}"
PROFILE_DEADLINE="${PROFILE_DEADLINE:-}"   # HH:MM: cut the medium profile so the quad is down by then; empty = PROFILE_MAX_S only
PIECES=ours,loadpath                       # reports() then says the V2_LOAD_PROOF line (the r4d V2 path rides in r4e)
grep -q PROOF_PREFILL_NO_MAIN "$PROOF_PREFILL_SH" 2> /dev/null \
    || { echo "HALT $PROOF_PREFILL_SH has no PROOF_PREFILL_NO_MAIN switch: sourcing it would run a proof; install proof_prefill.sh.new first" >&2; exit 2; }
# shellcheck source=/dev/null
PROOF_PREFILL_NO_MAIN=1 source "$PROOF_PREFILL_SH" "$R4E_OUT"

# final_run.sh second: its names win where both set one (STATUS = R4E_OUT/status in both, IMAGE, ARGS, QUAD, HERE = diag-final)
export C1D_IMAGE="${R4E_IMAGE:?set R4E_IMAGE to the r4e tag (r4d + routed MoE + session pin)}"
R4E_IMAGE_SHA="${R4E_IMAGE_SHA:?set R4E_IMAGE_SHA to the r4e image id prefix, e.g. sha256:0123456789ab}"
# shellcheck source=/dev/null
FINAL_RUN_NO_MAIN=1 source "$FINAL_RUN_SH" "$R4E_NAME_ARG" "$R4E_QUAD_ARG"
ARM="${R4E_ARM:-r4e}"   # r4f (coordinator 04:3x): two arms of one image at once each need their own run dir and container name
[[ "$ARM" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "HALT R4E_ARM=${ARM@Q}: use lower-case letters, digits and '-' (it names the container)" >&2; return 2 2> /dev/null || exit 2; }
RUN="$OUT/graded/$ARM"
NAME="claude-flash-launch-final-$ARM"
# A cut's override only ADDS forbidden keys (KEY|KEY...; Opus 16:5x BLOCKER: replacing the list bypassed the ARM check). A forbidden
# key is admitted only through final_run.sh's forbid(): the exact line in a baked ARM file (R4E_ARM_ENV, its sha in BAKED_ARM_SHA256S)
R4E_STOP_AFTER="${R4E_STOP_AFTER:-}"   # self_det: end after self-det (a bisect arm); unset: the whole run
case "$R4E_STOP_AFTER" in ''|self_det) ;; *) echo "HALT R4E_STOP_AFTER=${R4E_STOP_AFTER@Q}: use self_det or leave it unset" >&2; return 2 2> /dev/null || exit 2 ;; esac
FORBIDDEN_KEYS="$FORBIDDEN_KEYS${R4E_FORBIDDEN_KEYS:+|$R4E_FORBIDDEN_KEYS}"
[[ "$FORBIDDEN_KEYS" =~ ^[A-Z0-9_]+(\|[A-Z0-9_]+)*$ ]] || { echo "HALT R4E_FORBIDDEN_KEYS=${FORBIDDEN_KEYS@Q}: use KEY|KEY|... (env names)" >&2; return 2 2> /dev/null || exit 2; }

R4E_LEVERS_ENV="${R4E_LEVERS_ENV:-$R4E_HERE/r4e-levers.env}"
R4E_REQUIRED_KEYS="${R4E_REQUIRED_KEYS:-VLLM_GAUDI_MIXED_PROMPT_MOE HPU_GLM53_RAGGED_GRAPH_MOE HPU_GLM53_RAGGED_GRAPH_TABLE HPU_GLM53_REUSE_V2_FOR}"
R4E_REQUIRED_LINES="${R4E_REQUIRED_LINES:-VLLM_DECODE_PRIORITY=phases VLLM_DECODE_PRIORITY_MIXED=1 VLLM_DECODE_PRIORITY_MIN_DECODE_STEPS=8 VLLM_DECODE_PRIORITY_MAX_PREFILL_WAIT_S=8}"
PARITY_REF_LOG="${PARITY_REF_LOG:-/mnt/data1/graded-out/queue/final-r4c/graded/c1d-final/container.log}"   # the r4d bytes, same env
ORACLE_REFS="${ORACLE_REFS:-eager=$REF_QUEUE/graded/eager/graded/results.json}"   # gating (coordinator 22:2x: the eager anchor is tonight's reference)
ORACLE_INFO_REFS="${ORACLE_INFO_REFS:-c1d-final=/mnt/data1/graded-out/queue/final-r4c/graded/c1d-final/graded/results.json}"   # reported only
ORACLE_COMPARE_ARGS="${ORACLE_COMPARE_ARGS:---greedy-only}"
R4E_GRADED_MAX_S="${R4E_GRADED_MAX_S:-3600}"
R4E_GRADED_ARGS="${R4E_GRADED_ARGS:---samples 0 --max-task-tokens 9000}"   # extra graded.py run args: greedy only, the eager anchor's tasks
# Lever 1 (decode priority, tuned) rides in this arm (coordinator 22:5x), so the speed bar is that lever's own rule (proof_prefill.sh
# say_lever_decision): decode p50 >= 1.5x, TTFT p50 <= 1.3x, prefill >= 0.7x the medium baseline
SPEED_MIN_DECODE="${SPEED_MIN_DECODE:-1.5}"; SPEED_MAX_TTFT="${SPEED_MAX_TTFT:-1.3}"; SPEED_MIN_PREFILL="${SPEED_MIN_PREFILL:-0.7}"
# r4h7 (run_r4h7.sh sets them): SPEED_BAR_PY judges SPEED in place of the ratios above (speed_bar.py: the measured S7b run and
# its band; SPEED_BAR_MEASURED = which built-in measured run (s7b, s12b: the arm's own); SPEED_BAR_REFERENCE = another measured
# run's turns.jsonl). Empty = the ratio gate, as every earlier cut ran.
SPEED_BAR_PY="${SPEED_BAR_PY:-}"; SPEED_BAR_REFERENCE="${SPEED_BAR_REFERENCE:-}"; SPEED_BAR_MEASURED="${SPEED_BAR_MEASURED:-}"
# r4h7 legs, each off while its variable is empty (the verdict then reads OFF, which passes): the cold/warm leg (ledger C840;
# COLD_WARM_REFERENCE = the reference image's cold-warm folder of the same salt) and the live-proof list
COLD_WARM_PY="${COLD_WARM_PY:-}"; COLD_WARM_REFERENCE="${COLD_WARM_REFERENCE:-}"; COLD_WARM_SALT="${COLD_WARM_SALT:-cw1}"
# COLD_WARM_WHOLE_REPLY=1 judges past the forced end of thinking too: only once the reserve skip is proven live (REFERENCES.txt)
COLD_WARM_MAX_S="${COLD_WARM_MAX_S:-1800}"; COLD_WARM_WHOLE_REPLY="${COLD_WARM_WHOLE_REPLY:-0}"
LIVE_PROOF_PY="${LIVE_PROOF_PY:-}"; LIVE_PROOF_LIST="${LIVE_PROOF_LIST:-}"
ROUTED_LINE_RE="${ROUTED_LINE_RE:-ragged_pieces=}"     # S6: census counters ragged_pieces/overflows/reruns in the GLM53_GRAPHS_ONLY line
ROUTED_PIECES_KEY="${ROUTED_PIECES_KEY:-ragged_pieces}"; ROUTED_OVERFLOWS_KEY="${ROUTED_OVERFLOWS_KEY:-ragged_overflows}"
ROUTED_RERUNS_KEY="${ROUTED_RERUNS_KEY:-ragged_reruns}"; ROUTED_MAX_OVERFLOW_PCT="${ROUTED_MAX_OVERFLOW_PCT:-2}"
R4E_ROUTED_GATE="${R4E_ROUTED_GATE:-on}"   # off = the cut has no routed MoE (base r4e): ROUTED=OFF, which the verdict accepts only then
R4E_PIN_GATE="${R4E_PIN_GATE:-on}"         # off = the cut has no session pin (r4e-A 03:0x): PIN=OFF, accepted only then; the pin flag is refused
LAZY_RE="${LAZY_RE:-lazy fallback|lazy_fallbacks.?[:=] *[1-9]|LAZY_FALLBACK}"   # routed RCA 05:4x: graph counters carry 'lazy_fallbacks': 0
PIN_LIVE_RE="${PIN_LIVE_RE:-REUSE_V2}"
PIN_LOCAL_HIT_RE="${PIN_LOCAL_HIT_RE:-local_hit_tokens=([0-9]+)}"   # the pin's own lines; names confirmed at the 02:15 handoff
PIN_EVICT_RE="${PIN_EVICT_RE:-REUSE_V2\S*.*evict}"
# Ground-truth 33K + 150K needles after graded (coordinator 22:4x; 150K added 02:1x: the long-context row folds into r4e), only while the
# clock allows (reported, not gating: no paired reference)
R4E_NEEDLE_BY="${R4E_NEEDLE_BY:-04:40}"; R4E_NEEDLE_MAX_S="${R4E_NEEDLE_MAX_S:-1200}"; R4E_NEEDLE_LENGTHS="${R4E_NEEDLE_LENGTHS:-33k,150k}"
REMAINDER_PY="${REMAINDER_PY:-$R4E_HERE/remainder_hist.py}"   # the remainder census (MIXED_STEP_R4E_PLAN.md section 8): 256 vs 384 for r4f
# The long-context leg (coordinator 22:5x: the pin's speed case): LONG_LANES at LONG_START_MIN-MAX after the medium, cut by LONG_DEADLINE,
# compared with run4's per-lane bands (490K and 100-130K lanes; run4 had no lane at 250-300K)
LONG_LANES="${LONG_LANES:-4}"; LONG_START_MIN="${LONG_START_MIN:-250000}"; LONG_START_MAX="${LONG_START_MAX:-300000}"
LONG_DURATION_S="${LONG_DURATION_S:-600}"; LONG_MIN_TURNS="${LONG_MIN_TURNS:-2}"; LONG_MAX_WALL_S="${LONG_MAX_WALL_S:-2400}"
LONG_DEADLINE="${LONG_DEADLINE:-05:30}"; LONG_MAX_S="${LONG_MAX_S:-2700}"
# r4h7 (run_r4h7.sh): none = no clock cut for the long legs (run_r4g.sh exports 00:00, a closed window: its runs took the long
# leg from post_medium_legs.sh). Each long leg is still cut at LONG_MAX_S.
[ "$LONG_DEADLINE" != none ] || LONG_DEADLINE=""
RUN4_DIR="${RUN4_DIR:-/mnt/data1/claude-long/run4-c1d/c1d-final}"
RUN4_BANDS="${RUN4_BANDS:-450000:520000 95000:140000}"
# r4h7: a second long leg near the served ceiling (LONG2_LANES=0: off), judged by LONG_LEG_PY. MEDIUM_MAX_MODEL_LEN is the gates'
# served length: proof_prefill.sh profile_medium hands it to the harness, which ends a lane before a turn would pass it.
LONG2_LANES="${LONG2_LANES:-0}"; LONG2_START_MIN="${LONG2_START_MIN:-450000}"; LONG2_START_MAX="${LONG2_START_MAX:-500000}"
LONG_LEG_PY="${LONG_LEG_PY:-$R4E_HERE/long_leg_check.py}"
# r4h7: REFERENCE_SH STAGE PORT DIR ARM records references on this boot (run_r4h7.sh: the kit's record_refs.sh): early = right
# after self-det, late = after the last leg; a stage is cut at REFERENCE_MAX_S. Recorded, never judged. Empty = off.
REFERENCE_SH="${REFERENCE_SH:-}"; REFERENCE_MAX_S="${REFERENCE_MAX_S:-3600}"
MEDIUM_MAX_MODEL_LEN="${LAB_MAX_MODEL_LEN:-}"
[[ "$LONG2_LANES" =~ ^[0-9]+$ ]] || { echo "HALT LONG2_LANES=${LONG2_LANES@Q}: use a lane count, 0 = no second long leg" >&2; return 2 2> /dev/null || exit 2; }
# The det drive's key widths (drive_cells.py reads DET_WIDTHS): every width the served length spans. Without LAB_MAX_MODEL_LEN
# the boot serves the source unit's own length and all three run, as every earlier cut did; a 524,288 gates file leaves
# 1024,4096 (the 8192 set's 530,000-token anchor is refused by such a server). run_r4h7.sh passes the arm's decode widths.
if [ -z "${DET_WIDTHS:-}" ]; then
    DET_WIDTHS=""
    for det_width in 1024 4096 8192; do
        [ -n "${LAB_MAX_MODEL_LEN:-}" ] && [ $(( det_width * 128 )) -gt "$LAB_MAX_MODEL_LEN" ] || DET_WIDTHS+="${DET_WIDTHS:+,}$det_width"
    done
fi
[[ "$DET_WIDTHS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "HALT DET_WIDTHS=${DET_WIDTHS@Q} (LAB_MAX_MODEL_LEN=${LAB_MAX_MODEL_LEN:-unset}): use key widths in blocks, e.g. 1024,4096" >&2; return 2 2> /dev/null || exit 2; }
export DET_WIDTHS
DET_SETS="${DET_SETS:-$(tr ',' '\n' <<< "$DET_WIDTHS" | grep -c .)}"   # one det set per width
R4E_DET_LOGPROBS="${R4E_DET_LOGPROBS:-0}"               # N: both judged det passes ask for the top N logprobs per token (compiles while serving)
R4E_DET_MARGIN_ON_FAIL="${R4E_DET_MARGIN_ON_FAIL:-0}"   # N: after a FAIL at a matched schedule, one more det drive (not judged) asks for them
for r4e_switch in R4E_DET_LOGPROBS R4E_DET_MARGIN_ON_FAIL; do
    [[ "${!r4e_switch}" =~ ^([0-9]|1[0-9]|20)$ ]] \
        || { echo "HALT $r4e_switch=${!r4e_switch@Q}: use 0 (off) or 1..20, the top alternatives kept per token" >&2; return 2 2> /dev/null || exit 2; }
done
ATTR_PY="${ATTR_PY:-$L/diag-w1/fold/tools/serve_compile_attr.py}"   # the serve-compile attribution r4e_finish runs before the down
LAB_KEEP_LEAF=""   # qlib.sh bring_down: non-empty keeps the boot's recipe leaf (attribute_serve sets it)
R4E_TEARDOWN_STEP_S="${R4E_TEARDOWN_STEP_S:-120}"   # the most each pre-down step (docker inspect, docker logs, the attribution) may take
[[ "$R4E_TEARDOWN_STEP_S" =~ ^[1-9][0-9]*$ ]] \
    || { echo "HALT R4E_TEARDOWN_STEP_S=${R4E_TEARDOWN_STEP_S@Q}: use whole seconds above 0 (120 by default)" >&2; return 2 2> /dev/null || exit 2; }
SELF_DET=none; SELF_DET_NOTE=""; ORACLE=none; SPEED=none; ROUTED=none; PIN=none
COLD_WARM=OFF; LIVE_PROOF=OFF; LONG2=OFF   # the r4h7 legs: OFF until their phase runs

r4e_args() {  # final_args, then every line of the lever env file (later wins), then the overlay mount out: r4e serves its own baked bytes
    final_args || return 1
    [ -s "$R4E_LEVERS_ENV" ] || { say "HALT arm=$ARM: the lever env file $R4E_LEVERS_ENV is missing or empty"; return 1; }
    collapse_env   # what docker serves: the base sets VLLM_GAUDI_MIXED_PROMPT_KEYS twice (graphs-dual.env, then c1d's r4d keys)
    apply_levers "$R4E_LEVERS_ENV" || return 1
    drop_mount "$LOADPATH_DIR/glm53_loadpath.py:$SITE/glm53_loadpath.py:ro"
    local conflicts
    conflicts="$(env_conflicts | paste -sd' ')"
    [ -z "$conflicts" ] || { say "HALT arm=$ARM: --env keys set twice to different values: $conflicts (docker keeps the last); set each once"; return 1; }
}

apply_levers() {  # FILE: every lever line replaces EVERY --env entry of its key (c1d_args sets some keys twice; set_env swaps only the first)
    local line
    while IFS= read -r line; do
        case "$line" in ''|'#'*) continue ;; esac
        [[ "$line" == [A-Za-z_]*=* ]] \
            || { say "HALT arm=$ARM: $1 line ${line@Q} is not KEY=VALUE (no '=' passes the host's value as --env KEY, unchecked)"; return 1; }
        set_env_every "${line%%=*}" "$line"
    done < "$1"
}

set_env_every() {  # KEY LINE: every --env KEY=... pair leaves ARGS, then LINE is appended once
    local kept=() i
    for (( i = 0; i < ${#ARGS[@]}; i++ )); do
        if [ "${ARGS[i]}" = --env ] && [ "${ARGS[i+1]:-}" != "${ARGS[i+1]#"$1="}" ]; then i=$(( i + 1 )); continue; fi
        kept+=("${ARGS[i]}")
    done
    ARGS=(${kept[@]+"${kept[@]}"} --env "$2")
}

collapse_env() {  # docker's rule made explicit: of a key's --env entries only the last stays, so ARGS, CONFIG and the guard show the served value
    local -A last=()
    local kept=() i
    for (( i = 1; i < ${#ARGS[@]}; i++ )); do [ "${ARGS[i-1]}" = --env ] && last["${ARGS[i]%%=*}"]=$i; done
    for (( i = 0; i < ${#ARGS[@]}; i++ )); do
        if [ "${ARGS[i]}" = --env ] && [ "${last["${ARGS[i+1]%%=*}"]:-}" != $(( i + 1 )) ]; then i=$(( i + 1 )); continue; fi
        kept+=("${ARGS[i]}")
    done
    ARGS=(${kept[@]+"${kept[@]}"})
}

env_conflicts() {  # the --env keys ARGS sets more than once with different values, one per line (r4e-A2 served the r4d prompt keys so)
    local i
    for (( i = 1; i < ${#ARGS[@]}; i++ )); do [ "${ARGS[i-1]}" = --env ] && printf '%s\n' "${ARGS[i]}"; done \
        | sort -u | cut -d= -f1 | uniq -d
}

drop_mount() {  # SPEC: the --mount SPEC pair leaves ARGS; every other item stays in order
    local kept=() i
    for (( i = 0; i < ${#ARGS[@]}; i++ )); do
        if [ "${ARGS[i]}" = --mount ] && [ "${ARGS[i+1]:-}" = "$1" ]; then i=$(( i + 1 )); continue; fi
        kept+=("${ARGS[i]}")
    done
    ARGS=("${kept[@]}")
}

require_levers() {  # every routed-MoE and session-pin key is set: r4e without either lever is r4d, not the candidate
    local key
    local line
    case "$R4E_ROUTED_GATE" in on|off) ;; *) say "HALT arm=$ARM: R4E_ROUTED_GATE=$R4E_ROUTED_GATE; use on, or off for a cut without routed MoE"; return 1 ;; esac
    case "$R4E_PIN_GATE" in on|off) ;; *) say "HALT arm=$ARM: R4E_PIN_GATE=$R4E_PIN_GATE; use on, or off for a cut without the session pin"; return 1 ;; esac
    if [ "$R4E_PIN_GATE" = off ] && has_env HPU_GLM53_ATTN_PIN; then
        say "HALT arm=$ARM: HPU_GLM53_ATTN_PIN is set while R4E_PIN_GATE=off; drop the line from $R4E_LEVERS_ENV or turn the pin gate on"; return 1
    fi
    local required_keys required_lines
    read -ra required_keys <<< "$R4E_REQUIRED_KEYS"   # split on blanks, never globbed (Opus round 4)
    read -ra required_lines <<< "$R4E_REQUIRED_LINES"
    for key in ${required_keys[@]+"${required_keys[@]}"}; do
        has_env "$key" || { say "HALT arm=$ARM: $key is not set; add it to $R4E_LEVERS_ENV (r4e = r4d + routed MoE + session pin)"; return 1; }
    done
    for line in ${required_lines[@]+"${required_lines[@]}"}; do
        printf '%s\n' "${ARGS[@]}" | grep -qx -- "$line" \
            || { say "HALT arm=$ARM: $line is not in the boot env; add it to $R4E_LEVERS_ENV (lever 1 tuned, coordinator 22:5x)"; return 1; }
    done
    [ -f "$TASKS/tasks.json" ] || { say "HALT arm=$ARM: $TASKS/tasks.json is missing; graded needs run2's calibrated tasks"; return 1; }
}

baked_sha() { $DOCKER run --rm --entrypoint sha256sum "$IMAGE" "$SITE/$1" 2> /dev/null | cut -c1-8; }   # FILE: its sha8 inside the image

image_bytes_ok() {  # the image is the pinned id and bakes the approved load path (no overlay mount covers it any more)
    local id baked
    id="$($DOCKER image inspect -f '{{.Id}}' "$IMAGE" 2> /dev/null)"
    [ -n "$id" ] && [[ "$id" == "$R4E_IMAGE_SHA"* ]] \
        || { say "HALT arm=$ARM: $IMAGE resolves to ${id:-nothing on this host}, not the pinned $R4E_IMAGE_SHA; re-pin or retag"; return 1; }
    baked="$(baked_sha glm53_loadpath.py)"
    [ "$baked" = "$LOADPATH_SHA" ] \
        || { say "HALT arm=$ARM: $IMAGE bakes glm53_loadpath.py ${baked:-none}, not $LOADPATH_SHA (r4e must carry r4d's V2 fix baked)"; return 1; }
}

lever_summary() {  # KEY=VALUE of every line the lever env file sets, as booted
    local previous="" item keys
    keys="$(grep -v -E '^(#|$)' "$R4E_LEVERS_ENV" | cut -d= -f1 | paste -sd'|')"
    [ -n "$keys" ] || return 0
    for item in "${ARGS[@]}"; do
        [ "$previous" = --env ] && [[ "$item" =~ ^($keys)= ]] && printf '%s ' "$item"
        previous="$item"
    done
}

record_levers() { cp "$R4E_LEVERS_ENV" "$RUN/r4e-levers.env"; sha256sum "$RUN/r4e-levers.env" | cut -c1-8; }
mount_count() { printf '%s\n' "${ARGS[@]}" | grep -c -x -- --mount; }

r4e_parity() {  # STAGE: this boot's armed lines vs the r4d-bytes boot (tonight's c1d-final, same env): the extras are the two levers' lines
    $DOCKER logs -t "$NAME" > "$RUN/container.log" 2>&1
    bash "$HERE/armed_parity.sh" "$RUN/container.log" "$PARITY_REF_LOG" "$RUN/armed-parity-$1.txt"
    say "$(head -n 1 "$RUN/armed-parity-$1.txt" | cut -d'(' -f1)stage=$1 vs=r4d-c1d-final routed_lines=$(grep -a -c -E -e "$ROUTED_LINE_RE" "$RUN/container.log") pin_lines=$(grep -a -c -E -e "$PIN_LIVE_RE" "$RUN/container.log") install_lines=$(grep -a -c GLM53_LOADPATH_INSTALL "$RUN/container.log") (graded/$ARM/armed-parity-$1.txt)"
}

det_pass() {  # DIR [LOGPROBS]: one det drive (drive_cells.py --plan det, 3 sets of 16 lanes) into DIR/cells.jsonl, its lane texts into DIR/texts.jsonl
    local rc   # LOGPROBS (default 0 = det_drive's own request): the top N per token, kept beside the texts (det_texts.py DET_LOGPROBS)
    mkdir -p "$1"
    (cd "$L" && DET_LOGPROBS="${2:-0}" timeout "$DET_MAX_S" python3 "$R4E_HERE/det_texts.py" drive "$CELLS" "$1/texts.jsonl" -- \
        --port "${PORT[$QUAD]}" --out "$1/cells.jsonl" --plan det --abort-file "$RUN/ABORT") > "$1/drive.log" 2>&1
    rc=$?
    echo "drive $(basename "$1") rc=$rc" >> "$RUN/rc"
    [ "$rc" = 0 ] || say "SELF_DET arm=$ARM drive $(basename "$1") rc=$rc: $(tail -n 1 "$1/drive.log" | cut -c1-240)"
    return "$rc"
}

outputs_identical() {  # FILE (det_gate.py output): 0 when DET_SETS or more sets are listed and every one is IDENTICAL or "lanes equal 16/16"
    local sets equal
    sets=$(grep -c -E '^det-16x[0-9]+: ' "$1")
    equal=$(grep -c -E '^det-16x[0-9]+: steps matched [0-9]+/[0-9]+ (IDENTICAL$|UNJUDGED .*; lanes equal 16/16 )' "$1")
    [ "$sets" -ge "$DET_SETS" ] && [ "$sets" = "$equal" ]
}

self_det_verdict() {  # RC FILE: SELF_DET and SELF_DET_NOTE from det_gate.py's exit code and lines (coordinator 23:0x)
    case "$1" in
        0) SELF_DET=PASS; SELF_DET_NOTE="schedule matched, outputs identical" ;;
        1) SELF_DET=FAIL; SELF_DET_NOTE="a lane differs at a matched schedule" ;;
        2) if outputs_identical "$2"; then SELF_DET=PASS; SELF_DET_NOTE="schedule differed, outputs identical"
           else SELF_DET=UNJUDGED; SELF_DET_NOTE="batch-composition numerics: lanes differ where the lineup diverged; graded greedy decides"; fi ;;
        *) SELF_DET=FAIL; SELF_DET_NOTE="det_gate.py rc=$1" ;;
    esac
}

say_det_diffs() {  # the differing lanes: each one's first divergent token and, per pass that kept logprobs, its margin (det_texts.py diff)
    (cd "$L" && python3 "$R4E_HERE/det_texts.py" diff "$RUN/det-ref" "$RUN/det-rep" --port "${PORT[$QUAD]}" --margin-dir "$RUN/det-margin") \
        > "$RUN/self_det_diff.txt" 2>&1   # every line stays in graded/ARM/self_det_diff.txt; the status gets up to 8 lanes
    say "SELF_DET_DIFF arm=$ARM $(grep -a '^DET_DIFF_SUMMARY' "$RUN/self_det_diff.txt" | cut -c1-300) (graded/$ARM/self_det_diff.txt)"
    grep -a '^DET_DIFF ' "$RUN/self_det_diff.txt" | head -n 8 | cut -c1-240 | while IFS= read -r line; do say "SELF_DET_DIFF arm=$ARM $line"; done
    grep -a '^DET_MARGIN ' "$RUN/self_det_diff.txt" | head -n 16 | cut -c1-400 | while IFS= read -r line; do say "SELF_DET_MARGIN arm=$ARM $line"; done
}

margin_pass() {  # after a FAIL at a matched schedule, when R4E_DET_MARGIN_ON_FAIL asks: one more det drive with logprobs, never judged
    [ "$R4E_DET_MARGIN_ON_FAIL" != 0 ] || return 0
    say "SELF_DET_MARGIN arm=$ARM one more det drive with logprobs=$R4E_DET_MARGIN_ON_FAIL, not judged: its sampler graphs compile while serving, so serve-time compiles after this line are this drive's (graded/$ARM/det-margin)"
    det_pass "$RUN/det-margin" "$R4E_DET_MARGIN_ON_FAIL" || say "SELF_DET_MARGIN arm=$ARM the margin drive failed: the diff says n/a for the margin (graded/$ARM/det-margin/drive.log)"
}

self_det_phase() {  # two det drives on this boot, then det_gate.py det-ref det-rep on the boot log; 1 only on FAIL (the run halts)
    local rc
    [ "$R4E_DET_LOGPROBS" = 0 ] || say "SELF_DET_LOGPROBS arm=$ARM both judged passes ask for logprobs=$R4E_DET_LOGPROBS: the logprobs sampler graphs compile while serving (the serve gate counts them)"
    det_pass "$RUN/det-ref" "$R4E_DET_LOGPROBS" && det_pass "$RUN/det-rep" "$R4E_DET_LOGPROBS" || { SELF_DET=FAIL; SELF_DET_NOTE="a det drive failed"; return 1; }
    $DOCKER logs -t "$NAME" > "$RUN/det-ref/container.log" 2>&1
    cp "$RUN/det-ref/container.log" "$RUN/det-rep/container.log"
    (cd "$CELLS" && python3 det_gate.py "$RUN/det-ref" "$RUN/det-rep") > "$RUN/self_det.txt" 2>&1
    rc=$?
    echo "self_det rc=$rc" >> "$RUN/rc"
    self_det_verdict "$rc" "$RUN/self_det.txt"
    say "SELF_DET arm=$ARM $SELF_DET ($SELF_DET_NOTE) rc=$rc sets_identical=$(grep -c ' IDENTICAL$' "$RUN/self_det.txt")/$DET_SETS $(grep -a 'steps matched' "$RUN/self_det.txt" | tail -n 1 | cut -c1-80) (graded/$ARM/self_det.txt)"
    case "$SELF_DET" in
        UNJUDGED) say_det_diffs ;;
        FAIL) [ "$rc" != 1 ] || margin_pass; say_det_diffs ;;
    esac
    [ "$SELF_DET" != FAIL ]
}

r4e_graded_phase() {  # graded.py run (R4E_GRADED_ARGS) within R4E_GRADED_MAX_S in all; errors (rc 1) get one --resume pass in the time left
    local rc extra left ends=$(( $(date +%s) + R4E_GRADED_MAX_S ))
    read -r -a extra <<< "$R4E_GRADED_ARGS"
    graded_run "$R4E_GRADED_MAX_S" "${extra[@]}" > "$RUN/graded.log" 2>&1
    rc=$?
    left=$(( ends - $(date +%s) ))
    if [ "$rc" = 1 ] && [ "$left" -gt 60 ]; then
        say "GRADED arm=$ARM rc=1, one --resume pass for the tasks that errored (${left}s left of ${R4E_GRADED_MAX_S}s)"
        graded_run "$left" "${extra[@]}" --resume >> "$RUN/graded.log" 2>&1
        rc=$?
    fi
    echo "graded rc=$rc" >> "$RUN/rc"
    say "GRADED arm=$ARM rc=$rc (124 = the ${R4E_GRADED_MAX_S}s cap) args=${R4E_GRADED_ARGS:-none} results=$(result_count "$RUN/graded/results.json") $(grep -a 'attempted=' "$RUN/graded.log" | tail -n 1 | cut -c1-200)"
}

graded_run() {  # SECONDS [EXTRA ARGS]: one graded.py run pass against this boot, cut after SECONDS
    local seconds="$1"; shift
    timeout "$seconds" python3 -I "$G/graded.py" run --base-url "$URL" --tasks "$TASKS/tasks.json" --label "$ARM" --out-dir "$RUN/graded" \
        --lanes 16 --long-lanes 1 "$@"
}

result_count() { python3 -I -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["results"]))' "$1" 2> /dev/null || echo 0; }   # RESULTS_JSON

compare_one() {  # NAME REFERENCE_JSON: graded.py compare (ORACLE_COMPARE_ARGS) of this arm against REFERENCE; prints NO_LOSS|LOSS|INCONCLUSIVE|...
    local out="$OUT/graded/compare-r4e-vs-$1" rc word extra
    if [ ! -f "$2" ] || [ ! -f "$RUN/graded/results.json" ]; then
        say "COMPARE r4e-vs-$1 skipped: no results at $2 or $RUN/graded/results.json"; echo MISSING; return
    fi
    read -r -a extra <<< "$ORACLE_COMPARE_ARGS"
    python3 -I "$G/graded.py" compare --reference "$2" --candidate "$RUN/graded/results.json" --out-dir "$out" "${extra[@]}" > "$out.txt" 2>&1
    rc=$?
    case "$rc" in 0) word=NO_LOSS ;; 1) word=LOSS ;; 2) word=INCONCLUSIVE ;; *) word="ERROR_$rc" ;; esac
    say "COMPARE r4e-vs-$1 $word rc=$rc $(grep -a 'tasks attempted=' "$out.txt" | cut -c8-90) $(grep -a 'GRADED reason' "$out.txt" | head -n 1 | cut -c1-160)"
    echo "$word"
}

oracle_phase() {  # compare_one against every ORACLE_REFS name=results.json (PASS = NO_LOSS against each), then each ORACLE_INFO_REFS (reported only)
    local entry
    ORACLE=PASS
    for entry in $ORACLE_REFS; do
        [ "$(compare_one "${entry%%=*}" "${entry#*=}")" = NO_LOSS ] || ORACLE=FAIL
    done
    for entry in ${ORACLE_INFO_REFS:-}; do compare_one "${entry%%=*}" "${entry#*=}" > /dev/null; done
    say "ORACLE arm=$ARM $ORACLE (NO_LOSS against every one of: $(for entry in $ORACLE_REFS; do printf '%s ' "${entry%%=*}"; done)compare args ${ORACLE_COMPARE_ARGS:-none})"
}

before() { [ "$(date +%s)" -lt "$(date -d "today $1" +%s)" ]; }   # HH:MM: now is earlier than HH:MM today

needle_phase() {  # graded's R4E_NEEDLE_LENGTHS needles, greedy, right/wrong against the known answer, when the clock is before R4E_NEEDLE_BY
    before "$R4E_NEEDLE_BY" || { say "LONG_NEEDLE arm=$ARM skipped: past $R4E_NEEDLE_BY (the r4e window goes to the speed legs)"; return 0; }
    PROBE_LENGTHS="$R4E_NEEDLE_LENGTHS" PROBE_LANES=9 PROBE_MAX_S="$R4E_NEEDLE_MAX_S" bash "$R4E_HERE/long_needle_probe.sh" "$RUN/long-needle" "r4e=$URL" \
        > "$RUN/long-needle.log" 2>&1
    say "$(grep '^LONG_NEEDLE all' "$RUN/long-needle/summary.txt" 2> /dev/null | sed "s/^LONG_NEEDLE all/LONG_NEEDLE arm=$ARM/" | grep . || echo "LONG_NEEDLE arm=$ARM no summary, see $RUN/long-needle.log") (cap ${R4E_NEEDLE_MAX_S}s; reported, not gating)"
}

long_leg_phase() {  # the medium harness at LONG_LANES x LONG_START_MIN-MAX into RUN/long/medium, cut so the quad is down by LONG_DEADLINE
    mkdir -p "$RUN/long"
    MEDIUM_LANES="$LONG_LANES" MEDIUM_START_MIN="$LONG_START_MIN" MEDIUM_START_MAX="$LONG_START_MAX" MEDIUM_DURATION_S="$LONG_DURATION_S" \
        MEDIUM_MIN_TURNS="$LONG_MIN_TURNS" MEDIUM_MAX_WALL_S="$LONG_MAX_WALL_S" PROFILE_DEADLINE="$LONG_DEADLINE" PROFILE_MAX_S="$LONG_MAX_S" \
        profile_medium "$ARM-long" "$QUAD" "$RUN/long" "$NAME"
}

long2_leg_phase() {  # r4h7: the same harness at LONG2_LANES x LONG2_START_MIN-MAX into RUN/long2/medium, judged by long_leg_check.py; off at 0 lanes
    [ "$LONG2_LANES" != 0 ] || return 0
    local rc limit=()
    mkdir -p "$RUN/long2"
    MEDIUM_LANES="$LONG2_LANES" MEDIUM_START_MIN="$LONG2_START_MIN" MEDIUM_START_MAX="$LONG2_START_MAX" MEDIUM_DURATION_S="$LONG_DURATION_S" \
        MEDIUM_MIN_TURNS="$LONG_MIN_TURNS" MEDIUM_MAX_WALL_S="$LONG_MAX_WALL_S" PROFILE_DEADLINE="$LONG_DEADLINE" PROFILE_MAX_S="$LONG_MAX_S" \
        profile_medium "$ARM-long2" "$QUAD" "$RUN/long2" "$NAME"
    [ -z "$MEDIUM_MAX_MODEL_LEN" ] || limit=(--max-model-len "$MEDIUM_MAX_MODEL_LEN")
    python3 -I "$LONG_LEG_PY" "$RUN/long2/medium/turns.jsonl" --lanes "$LONG2_LANES" --min-turns "$LONG_MIN_TURNS" --min-prompt "$LONG2_START_MIN" \
        ${limit[@]+"${limit[@]}"} > "$RUN/long2_leg.txt" 2>&1
    rc=$?
    case "$rc" in 0) LONG2=PASS ;; 1) LONG2=FAIL ;; *) LONG2=UNJUDGED ;; esac
    say "LONG2 arm=$ARM $LONG2 rc=$rc starts=${LONG2_START_MIN}-${LONG2_START_MAX} $(tail -n 1 "$RUN/long2_leg.txt" | cut -c1-240) (graded/$ARM/long2_leg.txt)"
}

cold_warm_phase() {  # r4h7: cold_warm_leg.py run then judge on this boot, nothing else in flight (ledger C840); off while COLD_WARM_PY is empty
    [ -n "$COLD_WARM_PY" ] || return 0
    local rc reference=()
    [ -z "$COLD_WARM_REFERENCE" ] || reference=(--reference "$COLD_WARM_REFERENCE")
    [ "$COLD_WARM_WHOLE_REPLY" != 1 ] || reference+=(--whole-reply)
    timeout "$COLD_WARM_MAX_S" python3 -I "$COLD_WARM_PY" run --port "${PORT[$QUAD]}" --out "$RUN/cold-warm" --salt "$COLD_WARM_SALT" > "$RUN/cold-warm.log" 2>&1 \
        && python3 -I "$COLD_WARM_PY" judge "$RUN/cold-warm" ${reference[@]+"${reference[@]}"} >> "$RUN/cold-warm.log" 2>&1
    rc=$?
    case "$rc" in 0) COLD_WARM=PASS ;; 1) COLD_WARM=FAIL ;; *) COLD_WARM=UNJUDGED ;; esac
    echo "cold_warm rc=$rc" >> "$RUN/rc"
    say "COLD_WARM arm=$ARM $COLD_WARM rc=$rc $(tail -n 1 "$RUN/cold-warm.log" | cut -c1-240) (graded/$ARM/cold-warm.log)"
}

live_proof() {  # LOG: r4h7: every lever's armed line and live counter in the final boot log (live_proof.py's list); off while LIVE_PROOF_LIST is empty
    [ -n "$LIVE_PROOF_LIST" ] || return 0
    local rc
    python3 -I "$LIVE_PROOF_PY" "$1" --list "$LIVE_PROOF_LIST" --levers "$R4E_LEVERS_ENV" > "$RUN/live_proof.txt" 2>&1
    rc=$?
    case "$rc" in 0) LIVE_PROOF=PASS ;; 1) LIVE_PROOF=FAIL ;; *) LIVE_PROOF=NOT-VERIFIED ;; esac
    say "LIVE_PROOF arm=$ARM $LIVE_PROOF rc=$rc $(tail -n 1 "$RUN/live_proof.txt" | cut -c1-240) (graded/$ARM/live_proof.txt)"
}

reference_phase() {  # STAGE: r4h7's reference recording on this boot (REFERENCE_SH STAGE PORT DIR ARM; early = after self-det, late = after the
    # last leg), cut at REFERENCE_MAX_S. Recorded, never judged: no verdict word reads it. Off while REFERENCE_SH is empty.
    [ -n "$REFERENCE_SH" ] || return 0
    local rc
    mkdir -p "$RUN/refs"
    timeout -k 10 "$REFERENCE_MAX_S" bash "$REFERENCE_SH" "$1" "${PORT[$QUAD]}" "$RUN/refs" "$ARM" > "$RUN/refs-$1.console" 2>&1
    rc=$?
    say "REFERENCES arm=$ARM stage=$1 rc=$rc $(tail -n 1 "$RUN/refs-$1.console" | cut -c1-240) (graded/$ARM/refs-$1.console; recorded, not part of the verdict)"
}

speed_bar_gate() {  # TURNS: r4h7's SPEED: speed_bar.py against the measured reference inside its band (exit 0 PASS, 1 FAIL, else UNJUDGED)
    local rc reference=()
    [ -z "$SPEED_BAR_MEASURED" ] || reference=(--measured "$SPEED_BAR_MEASURED")
    [ -z "$SPEED_BAR_REFERENCE" ] || reference+=(--reference "$SPEED_BAR_REFERENCE")
    python3 -I "$SPEED_BAR_PY" "$1" ${reference[@]+"${reference[@]}"} > "$RUN/speed_bar.txt" 2>&1
    rc=$?
    case "$rc" in 0) SPEED=PASS ;; 1) SPEED=FAIL ;; *) SPEED=UNJUDGED ;; esac
    say "SPEED arm=$ARM $SPEED rc=$rc $(tail -n 1 "$RUN/speed_bar.txt" | cut -c1-260) (graded/$ARM/speed_bar.txt; the baseline ratios are not judged)"
}

breakdown() {  # LABEL TURNS LOG [MIN MAX]: turn_breakdown.py's lines, each said with LABEL
    local band=()
    [ -z "${4:-}" ] || band=(--min-prompt "$4" --max-prompt "$5")
    [ -s "$2" ] || { say "BREAKDOWN $1: no turns at $2"; return 0; }
    python3 -I "$R4E_HERE/turn_breakdown.py" "$(dirname "$2")" --turns "$2" --log "$3" "${band[@]}" --local-hit-re="$PIN_LOCAL_HIT_RE" --evict-re="$PIN_EVICT_RE" 2>&1 | while IFS= read -r line; do
        say "BREAKDOWN $1 ${line:0:300}"
    done
}

speed_breakdowns() {  # medium: this arm vs the baseline; long leg: this arm vs run4's per-lane bands (fence, GB/turn, TTFT, decode per lane)
    local band
    breakdown "$ARM-medium" "$RUN/medium/turns.jsonl" "$RUN/container.log"
    breakdown "$BASELINE_LABEL" "$BASELINE_RUN/medium/turns.jsonl" "$BASELINE_RUN/container.log"
    breakdown "$ARM-long" "$RUN/long/medium/turns.jsonl" "$RUN/container.log"
    for band in $RUN4_BANDS; do breakdown "run4-${band%%:*}-CONTEXT-ONLY" "$RUN4_DIR/profileA/turns.jsonl" "$RUN4_DIR/container.log" "${band%%:*}" "${band#*:}"; done
}

engine_alive() {  # PHASE: the boot still runs and answers /health; else ENGINE_DEAD with the first worker error line
    is_running "$NAME" && [ "$(curl -s -m 15 -o /dev/null -w '%{http_code}' "$URL/health")" = 200 ] && return 0
    say "ENGINE_DEAD arm=$ARM after $1: $($DOCKER logs "$NAME" 2>&1 | grep -a -m 1 -E 'RuntimeError: GLM53_|EngineDeadError|RuntimeError: ' | cut -c1-300)"
    return 1
}

field() { printf '%s\n' "$1" | grep -o -e "$2=[0-9.]*" | tail -n 1 | cut -d= -f2; }   # LINE KEY: the last KEY=<number> value in LINE

routed_counters() {  # LOG: the routed-MoE path counters (the last ROUTED_LINE_RE line) and the lazy-fallback line count
    local line pieces overflows reruns lazy
    lazy="$(grep -a -c -E -e "$LAZY_RE" "$1")"
    if [ "$R4E_ROUTED_GATE" = off ]; then   # no routed counters to judge, but graphs-only still holds: a lazy line is a FAIL
        ROUTED=OFF; [ "$lazy" = 0 ] || ROUTED=FAIL
        say "ROUTED arm=$ARM $ROUTED (R4E_ROUTED_GATE=off: this cut has no routed MoE; lazy_lines=$lazy, want 0)"; return 0
    fi
    line="$(grep -a -E -e "$ROUTED_LINE_RE" "$1" | tail -n 1)"
    pieces="$(field "$line" "$ROUTED_PIECES_KEY")"; overflows="$(field "$line" "$ROUTED_OVERFLOWS_KEY")"; reruns="$(field "$line" "$ROUTED_RERUNS_KEY")"
    ROUTED=FAIL
    if [ -n "$pieces" ] && [ -n "$overflows" ] && awk -v p="$pieces" -v o="$overflows" -v m="$ROUTED_MAX_OVERFLOW_PCT" -v z="$lazy" \
        'BEGIN {exit !(p > 0 && o * 100 <= p * m && z == 0)}'; then ROUTED=PASS; fi
    say "ROUTED arm=$ARM $ROUTED pieces=${pieces:-none} overflows=${overflows:-none} reruns=${reruns:-none} lazy_lines=$lazy (want pieces > 0, overflows <= ${ROUTED_MAX_OVERFLOW_PCT}% of pieces, 0 lazy; line /$ROUTED_LINE_RE/)"
}

pin_counters() {  # LOG REPORT: session-pin live lines in the boot log and the external hit % of this run vs the baseline (LOADPATH_PROOF lines 1, 2)
    local live arm base
    live="$(grep -a -c -E -e "$PIN_LIVE_RE" "$1")"
    if [ "$R4E_PIN_GATE" = off ]; then
        PIN=OFF; [ "$live" = 0 ] || PIN=FAIL
        say "PIN arm=$ARM $PIN (R4E_PIN_GATE=off: this cut has no session pin; live_lines=$live, want 0)"; return 0
    fi
    arm="$(grep '^LOADPATH_PROOF' "$2" 2> /dev/null | sed -n 1p | grep -o 'external_hit_pct=[0-9.]*' | cut -d= -f2)"
    base="$(grep '^LOADPATH_PROOF' "$2" 2> /dev/null | sed -n 2p | grep -o 'external_hit_pct=[0-9.]*' | cut -d= -f2)"
    PIN=FAIL
    if [ -n "$arm" ] && [ -n "$base" ] && awk -v l="$live" -v a="$arm" -v b="$base" 'BEGIN {exit !(l > 0 && a > b)}'; then PIN=PASS; fi
    say "PIN arm=$ARM $PIN live_lines=$live external_hit_pct=${arm:-n/a} baseline=${base:-n/a} (want live lines > 0 and the hit % above $BASELINE_LABEL's; line /$PIN_LIVE_RE/)"
}

speed_gate() {  # REPORT: lane_report.py's ratios vs the medium baseline (proof_prefill.sh ratio_of); UNJUDGED without them
    local line decode ttft prefill
    if [ -n "$SPEED_BAR_PY" ]; then speed_bar_gate "$RUN/medium/turns.jsonl"; return; fi
    line="$(tail -n 1 "$1" 2> /dev/null)"
    decode="$(ratio_of "$line" decode_tok_s)"; ttft="$(ratio_of "$line" ttft_s)"; prefill="$(ratio_of "$line" prefill_tok_s)"
    if [ -z "$decode" ] || [ -z "$ttft" ] || [ -z "$prefill" ]; then
        SPEED=UNJUDGED; say "SPEED arm=$ARM UNJUDGED: no baseline ratios in $1 (baseline $BASELINE_RUN)"; return
    fi
    SPEED=FAIL
    if awk -v d="$decode" -v t="$ttft" -v p="$prefill" -v md="$SPEED_MIN_DECODE" -v mt="$SPEED_MAX_TTFT" -v mp="$SPEED_MIN_PREFILL" \
        'BEGIN {exit !(d >= md && t <= mt && p >= mp)}'; then SPEED=PASS; fi
    say "SPEED arm=$ARM $SPEED vs $BASELINE_LABEL: decode_p50=${decode}x (want >= $SPEED_MIN_DECODE) ttft_p50=${ttft}x (<= $SPEED_MAX_TTFT) prefill=${prefill}x (>= $SPEED_MIN_PREFILL)"
}

replay_ms() { grep -a "prompt replay $2:" "$1" 2> /dev/null | head -n 1 | grep -o 'median=[0-9.]*' | cut -d= -f2; }   # REPORT KEY: median ms

proof_field() { grep '^LOADPATH_PROOF' "$1" 2> /dev/null | sed -n "${2}p" | grep -o "$3=[0-9.]*" | cut -d= -f2; }   # REPORT LINE FIELD

ratio() { awk -v a="$1" -v b="$2" 'BEGIN {if (a == "" || b == "" || b + 0 == 0) print "n/a"; else printf "%.2fx", a / b}'; }   # ARM BASE

lever_gains() {  # RUN_REPORT BASE_REPORT LOADPATH_REPORT: each lever against its own counter, this arm vs the baseline (reported, not gating)
    local key line="" arm base field
    local served_keys
    served_keys="$(sed -n 's/^VLLM_GAUDI_MIXED_PROMPT_KEYS=//p' "$R4E_LEVERS_ENV" 2> /dev/null | tail -n 1 | tr ',' ' ')"   # this cut's own keys
    for key in ${served_keys:-1024:8192 1024:1024 128:8192 128:1024}; do
        arm="$(replay_ms "$1" "$key")"; base="$(replay_ms "$2" "$key")"
        line+="$key=${arm:-n/a}/${base:-n/a}ms($(ratio "$arm" "$base")) "
    done
    say "GAIN routed arm=$ARM prompt replay median arm/baseline: $line(routed MoE targets the 1024 keys; < 1.00x is faster)"
    line=""
    for field in external_hit_pct loaded_gb ttft_p50_s pure_decode_wall_p50_ms mixed_wall_p50_ms; do
        arm="$(proof_field "$3" 1 "$field")"; base="$(proof_field "$3" 2 "$field")"
        line+="$field=${arm:-n/a}/${base:-n/a}($(ratio "$arm" "$base")) "
    done
    say "GAIN pin arm=$ARM arm/baseline: $line(the pin raises the external hit %, the re-read it saves shows in TTFT)"
}

key256_report() {  # the mixed-step key swap gate on this run (coordinator 00:4x): its exit code and last line said; reported, not gating
    local rc inputs=("$RUN/container.log" "$RUN/report.txt")
    [ -s "$RUN/report.txt" ] || { say "KEY256 arm=$ARM skipped: no graded/$ARM/report.txt (the medium profile did not run)"; return 0; }
    [ -f "$RUN/medium/report.txt" ] && inputs+=("$RUN/medium/report.txt")   # the medium profile writes none (r4e-A2 rc=1 FileNotFoundError)
    python3 -I "$L/mixed-step/key256_gate.py" "${inputs[@]}" \
        --baseline-report "$BASELINE_RUN/report.txt" > "$RUN/key256_gate.txt" 2>&1
    rc=$?
    say "KEY256 arm=$ARM rc=$rc $(tail -n 1 "$RUN/key256_gate.txt" | cut -c1-240) (graded/$ARM/key256_gate.txt)"
}

remainder_report() {  # the prompt-piece remainder census on the whole log and the medium window: its rc and line said; reported, not gating
    local scope rc
    [ -s "$RUN/container.log" ] || { say "REMAINDER_HIST arm=$ARM skipped: no graded/$ARM/container.log"; return 0; }
    for scope in log medium; do
        local window=()
        if [ "$scope" = medium ]; then
            [ -s "$RUN/medium/turns.jsonl" ] || { say "REMAINDER_HIST arm=$ARM medium skipped: no graded/$ARM/medium/turns.jsonl"; continue; }
            window=(--turns "$RUN/medium/turns.jsonl")
        fi
        python3 -I "$REMAINDER_PY" "$RUN/container.log" "${window[@]}" --scope "$ARM-$scope" > "$RUN/remainder_hist-$scope.txt" 2>&1
        rc=$?
        say "REMAINDER_HIST arm=$ARM rc=$rc $(tail -n 1 "$RUN/remainder_hist-$scope.txt" | cut -c1-240)"
    done
}

say_verdict() {  # one closing verdict line: PASS only when every gate passed; self-det UNJUDGED counts only beside an oracle NO_LOSS
    local word=FAIL self_ok=no routed_ok=$ROUTED pin_ok=$PIN
    case "$SELF_DET" in PASS) self_ok=yes ;; UNJUDGED) [ "$ORACLE" = PASS ] && self_ok=yes ;; esac
    [ "$ROUTED$R4E_ROUTED_GATE" != OFFoff ] || routed_ok=PASS   # OFF passes only when the cut's gate is off
    [ "$PIN$R4E_PIN_GATE" != OFFoff ] || pin_ok=PASS
    local leg legs_ok=yes   # the r4h7 legs: each PASS, or OFF because the cut does not run it
    for leg in "$COLD_WARM" "$LIVE_PROOF" "$LONG2"; do case "$leg" in PASS|OFF) ;; *) legs_ok=no ;; esac; done
    [ "$self_ok$ORACLE$SPEED$routed_ok$pin_ok$legs_ok" = yesPASSPASSPASSPASSyes ] && word=PASS
    say "VERDICT r4e $word self_det=$SELF_DET${SELF_DET_NOTE:+ ($SELF_DET_NOTE)} oracle=$ORACLE speed=$SPEED routed=$ROUTED pin=$PIN cold_warm=$COLD_WARM live_proof=$LIVE_PROOF long2=$LONG2 image=$IMAGE levers=$(sha256sum "$R4E_LEVERS_ENV" | cut -c1-8)"
}

serve_compile_total() {  # LOG: its serve-time compiles, each rank's last GLM53_SERVE_COMPILES total summed (boot_budget.py serve's count); empty = unreadable
    awk '/GLM53_SERVE_COMPILES / && match($0, /\(Worker_TP[0-9]+/) { rank = substr($0, RSTART, RLENGTH)
             if (match($0, / total=[0-9]+/)) last[rank] = substr($0, RSTART + 7, RLENGTH - 7) }
         END { for (rank in last) sum += last[rank]; print sum + 0 }' "$1" 2> /dev/null
}

teardown_step() {  # WHAT CMD...: CMD cut after R4E_TEARDOWN_STEP_S (KILL 10 s later); a cut is said, and its rc 124 (137 killed) returned
    local what="$1" rc
    shift
    timeout -k 10 "$R4E_TEARDOWN_STEP_S" "$@"
    rc=$?
    case "$rc" in 124|137) say "TEARDOWN_TIMEOUT arm=$ARM $what did not end within ${R4E_TEARDOWN_STEP_S}s (R4E_TEARDOWN_STEP_S) and was cut: the down and the quad release go ahead without it" ;; esac
    return "$rc"
}

attribute_serve() {  # MEASURED_RC: before the down, serve_compile_attr.py on the boot log into graded/ARM/serve_compile_attr.txt. Sets LAB_KEEP_LEAF
    local compiles failed=no rc   # only when that attribution failed on a run that failed so far or compiled while serving
    LAB_KEEP_LEAF=""
    teardown_step "docker inspect $NAME" $DOCKER inspect "$NAME" > /dev/null 2>&1 || return 0   # no container (or no answer): the leaf goes as before
    teardown_step "docker logs $NAME (the saved boot log may be partial)" $DOCKER logs -t "$NAME" > "$RUN/container.log" 2>&1
    compiles="$(serve_compile_total "$RUN/container.log")"
    [ "$1" = 0 ] && [ "$SELF_DET" != FAIL ] && [ "$ORACLE" != FAIL ] || failed=yes
    teardown_step "serve_compile_attr.py (not attributed)" python3 -I "$ATTR_PY" "$RUN" > "$RUN/serve_compile_attr.txt" 2>&1
    rc=$?
    echo "serve_attr rc=$rc" >> "$RUN/rc"
    if [ "$rc" != 0 ] && { [ "$failed" = yes ] || [ "${compiles:-unknown}" != 0 ]; }; then LAB_KEEP_LEAF=1; fi
    say "SERVE_ATTR arm=$ARM rc=$rc serve_compiles_total=${compiles:-unknown} failed_so_far=$failed keep_leaf=${LAB_KEEP_LEAF:-0} $(tail -n 1 "$RUN/serve_compile_attr.txt" | cut -c1-200) (graded/$ARM/serve_compile_attr.txt)"
}

say_kept_leaf() {  # after the down: the recipe leaf bring_down kept, its path and size on disk, and what frees it
    [ -n "$LAB_KEEP_LEAF" ] || return 0
    local leaf
    leaf="$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["recipe_leaf"])' "$L/state-$NAME.json" 2> /dev/null)"
    say "LEAF_KEPT arm=$ARM path=${leaf:-unknown: no recipe_leaf in $L/state-$NAME.json} size=$(timeout 60 du -sh "${leaf:-/nonexistent}" 2> /dev/null | cut -f1) serve_compile_attr.py did not run (graded/$ARM/serve_compile_attr.txt says why): fix that, rerun python3 $ATTR_PY $RUN, then delete the leaf directory"
}

r4e_finish() {  # MEASURED_RC: the attribution, then final_run.sh's finish (down + release); the leaf outlives the down only when attribution could not run
    attribute_serve "$1"
    finish
    local rc=$?
    say_kept_leaf
    LAB_KEEP_LEAF=""
    return "$rc"
}

r4e_prepare() {  # config, admission and the quad: every refusal before any card time
    mkdir -p "$RUN/trace" "$RUN/hl"
    chmod 777 "$RUN/trace" "$RUN/hl"
    : > "$RUN/rc"
    say "START pid=$$ arm=$ARM quad=$QUAD image=$IMAGE pinned=$R4E_IMAGE_SHA levers=$R4E_LEVERS_ENV tasks=$TASKS medium_baseline=$BASELINE_RUN graded_max_s=$R4E_GRADED_MAX_S graded_args=${R4E_GRADED_ARGS:-none} medium_deadline=${PROFILE_DEADLINE:-none} gates=$R4E_GATES_OVERRIDE$([ -f "$R4E_GATES_OVERRIDE" ] && echo "@$(sha256sum "$R4E_GATES_OVERRIDE" | cut -c1-8)") routed_gate=$R4E_ROUTED_GATE pin_gate=$R4E_PIN_GATE"
    r4e_args && forbid && require_levers && image_bytes_ok || return 1
    say "CONFIG final_env=$FINAL_ENV sha=$(record_env) levers_sha=$(record_levers) $(env_summary)$(lever_summary)thinking=on mounts=$(mount_count)"
    static_guard || return 1
    say "WAITING r4e: quad $QUAD once free ${FREE_FOR_S}s in a row (B replica down 02:40, S8 one-card cell ends ~03:25); the next WAITING line is final_run.sh's own wording"
    take_placed_quad
}

r4e_measure() {  # on the booted engine: self-det (only a FAIL stops here), graded + oracle, the medium profile; 1 when the engine died or self-det failed
    r4e_parity booted
    self_det_phase || { say "HALT arm=$ARM: SELF_DET $SELF_DET; graded and medium are not run"; return 1; }
    engine_alive self-det || return 1
    r4e_parity det
    if [ "$R4E_STOP_AFTER" = self_det ]; then   # a bisect arm (r4f-B2, coordinator 07:0x): the self-det answer only
        say "STOP_AFTER arm=$ARM self_det: graded, oracle, needles, medium and the long leg are not run (R4E_STOP_AFTER=self_det)"; return 0
    fi
    reference_phase early
    [ -z "$REFERENCE_SH" ] || engine_alive references-early || return 1
    r4e_graded_phase
    oracle_phase
    engine_alive graded || return 1
    needle_phase
    cold_warm_phase
    [ -z "$COLD_WARM_PY" ] || engine_alive cold-warm || return 1
    profile_medium "$ARM" "$QUAD" "$RUN" "$NAME"
    engine_alive medium || return 1
    long_leg_phase
    engine_alive long-leg || return 1
    long2_leg_phase
    [ "$LONG2_LANES" = 0 ] || engine_alive long2-leg || return 1
    reference_phase late
    [ -z "$REFERENCE_SH" ] || engine_alive references-late
}

r4e_main() {
    r4e_prepare || { say "DONE verdict=HALT"; return 1; }
    trap 'say "TERM: bringing the boot down"; r4e_finish 143; say "DONE verdict=STOPPED"; exit 143' TERM INT
    start_watch "$NAME" "$QUAD" "$RUN"
    say "BOOT arm=$ARM quad=$QUAD image=$IMAGE name=$NAME"
    if ! boot "$NAME" "$QUAD" "$IMAGE" "$RUN" "$DEADLINE"; then
        say "BOOT_FAILED arm=$ARM $(tail -n 1 "$RUN/rc") $(grep -a -m1 'refuses\|Error' "$RUN/up.log" | cut -c1-160)"
        finish; say "DONE verdict=BOOT_FAILED"; return 1
    fi
    say "BOOTED arm=$ARM $(tail -n 1 "$RUN/rc")"
    r4e_measure
    local measured=$?
    r4e_finish "$measured"
    [ ! -s "$RUN/medium/turns.jsonl" ] || reports "$ARM" "$RUN"
    key256_report
    remainder_report
    routed_counters "$RUN/container.log"
    pin_counters "$RUN/container.log" "$RUN/loadpath_report.txt"
    live_proof "$RUN/container.log"
    speed_gate "$RUN/lane_report.txt"
    lever_gains "$RUN/report.txt" "$BASELINE_RUN/report.txt" "$RUN/loadpath_report.txt"
    speed_breakdowns
    say_verdict
    if [ "$measured" = 0 ]; then say "DONE"
    elif [ "$SELF_DET" = FAIL ]; then say "DONE verdict=SELF_DET_FAIL"
    else say "DONE verdict=ENGINE_DEAD"
    fi
}

[ -n "${R4E_RUN_NO_MAIN:-}" ] || r4e_main
