#!/bin/bash
# diag-r4e/r4e_gates.sh: the one reader of an r4e gates file (r4e_run.sh, run_r4f.sh; run_r4g.sh should source it too). Sourced, defines
# r4e_read_gates only. A gates file is data, never code (coordinator 2026-11-09 17:1x: a sourced gates file could re-point FINAL_RUN_SH
# and with it the ARM allowlist): KEY=VALUE lines (VALUE bare [A-Za-z0-9_./:,=@+%-]* or in single quotes), '#' comments and blank lines.
# Keys are an allowlist (Opus round 3: a denylist let GCONV_PATH / NODE_OPTIONS / DOCKER_HOST through to child processes): exactly the
# keys the 20 gates files in the queue and the run_r4f.sh / run_r4g.sh writers use. Each key once; each value exported.
R4E_GATE_KEYS_ALLOWED='^(R4E_(REQUIRED_KEYS|PIN_GATE|ROUTED_GATE|FORBIDDEN_KEYS|ARM_ENV|ARM_ENV_SHA256)|PIN_(LIVE|LOCAL_HIT|EVICT)_RE|ROUTED_(LINE_RE|PIECES_KEY|OVERFLOWS_KEY|RERUNS_KEY|MAX_OVERFLOW_PCT)|LAB_MAX_MODEL_LEN|LAB_DEFAULT_REASONING_EFFORT)$'

r4e_read_gates() {  # FILE: export its KEY=VALUE lines; HALT (stderr, return 1) on any other line, a key not allowed or a key set twice
    local line key value n=0
    local -A first_line=()
    local quoted_re="^([A-Z_][A-Z0-9_]*)='([^'"$'\r'"]*)'\$" bare_re='^([A-Z_][A-Z0-9_]*)=([A-Za-z0-9_./:,=@+%-]*)$'
    [ -f "$1" ] || { echo "HALT gates file $1 is missing" >&2; return 1; }
    while IFS= read -r line || [ -n "$line" ]; do
        n=$(( n + 1 ))
        [[ "$line" =~ ^(#.*)?$ ]] && continue
        [[ "$line" =~ $quoted_re ]] || [[ "$line" =~ $bare_re ]] \
            || { echo "HALT $1:$n is not a KEY=VALUE data line (bare or single-quoted value): ${line@Q}; gates files are data, not code" >&2; return 1; }
        key="${BASH_REMATCH[1]}" value="${BASH_REMATCH[2]}"
        [[ "$key" =~ $R4E_GATE_KEYS_ALLOWED ]] \
            || { echo "HALT $1:$n sets $key: not a gates key (R4E_GATE_KEYS_ALLOWED in r4e_gates.sh); set anything else in the launcher's environment" >&2; return 1; }
        [ -z "${first_line[$key]:-}" ] \
            || { echo "HALT $1:$n: $key is set twice (lines ${first_line[$key]} and $n); keep one line per key" >&2; return 1; }
        first_line[$key]=$n
        printf -v "$key" '%s' "$value" && export "$key"
    done < "$1"
}
