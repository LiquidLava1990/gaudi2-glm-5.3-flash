# ATTN-PIN overlay for r4e (c1d graphs-only image) — install notes for the c1d engineer

Attention-only session pin: a finished turn keeps its full-attention KV blocks in HBM (its own graph-written bytes), the
next turn of the same session reuses them locally and loads only the last hit block + the recurrent row through today's
connector + LOADPATH_V2 path. Graph keys unchanged; equality with the load path is gated by det/oracle (the CPU store
dedups by block hash, so it is not bitwise by construction). Design and the C588
comparison: `prebuild.md`. Default OFF; OFF is byte-identical (no file of the image changes; the two new files import
nothing and patch nothing while the flag is unset).

## r5 (2026-11-09): why r4f never hit, what changed, the one-node cell

**Root cause.** The pin was wrapped on `KVCacheManager.free` only. Serving runs async scheduling with a kv_both
connector, so `defer_block_free` is on (scheduler.py:521-527). A stop-token finish while the next step is in flight frees
through `KVCacheManager.pop_blocks_for_free` (scheduler.py:3743-3752), and `free` never runs, so no real chat turn was
ever pinned. Only length-capped requests (the det drive's fixed output) freed through `free`. Evidence, r4f-A on
2026-11-09: the det drive pinned 81, all from prompts that are never reused. The graded run then finished 239 requests,
all with `finish_reason=stop` (15 of them needle prompts at or above 8,192 tokens), and pinned 0: no counter line after
06:58:57. The det pins (64 sessions, 10,408 blocks) stayed held at 31.3% KV usage with nothing to hit. r4f-B never
reached graded: it HALTed at SELF_DET, and its det cells show `external_prefix_cache_hits=0.0`, so `hit and load_async`
was false on every call. The 917 ms / 1.73 GB / 2490 ms figures are the `run4-95000-CONTEXT-ONLY` reference rows that
every status prints, not r4f-B numbers.

**Changes (default OFF unchanged).**
1. `KVCacheManager.pop_blocks_for_free` is wrapped like `free` (11 patch points). The pinned blocks sit below
   `num_computed - num_in_flight`, so the in-flight step never writes them, and they are left off the deferred list.
2. Idle bound: `HPU_GLM53_ATTN_PIN_IDLE_TURNS=N` (default 32). A pin that sees N later finished requests (requests that
   held KV) without a hit goes back to the pool and counts as `evicted_idle`, so single-turn traffic never holds KV for
   nothing.
3. `plan_adopt` emits the counter line on its cadence, so `lookups`/`connector_fallbacks` show without a pin event.

The connector only answers hits for prompts of at least `cold_reuse_min_tokens=8192`. Shorter chat turns run 0 lookups
by design, so the proof is the medium profile (about 70K-token prompts).

**Tests (CPU, no card):** host 53 OK, craft OK, real r4f Scheduler 7/7 OK (new: stop finish on the deferred path pins,
and turn 2 adopts the pinned prefix; fresh single-turn prompts age the pin out), image 9/9 OK (11 points wrapped). RED
before the fix, on the same real Scheduler: `no pin for a deferred free: pinned=1`.

**One-node cell on r4g (lab lead runs it; quad of your choice).** `run_tests.sh` takes the image as its required first
argument (no default; it refuses a missing tag). r4g did not exist on the host at kit time, so `R4G` is a parameter.

```
L=$HOME/.codex/tmp/flash-spikes-20261104/team-20261106-dual; K=$L/pin-kit
R4G=claude-flash-dual-graphs-c1d:<r4g tag>; TAG=$R4G-pin5; SITE=/usr/local/lib/python3.12/dist-packages
(cd $K && sha256sum -c SHA256SUMS)                                       # kit bytes (glm53_attn_pin.py 3dcd55ea)
cid=$(docker create $R4G true)                                           # r4g + the one pin file (the .pth is unchanged)
docker cp $cid:$SITE/glm53_attn_pin.py - | tar -xO | sha256sum           # 3dcd55ea... already: skip the bake, TAG=$R4G
docker cp $K/glm53_attn_pin.py $cid:$SITE/glm53_attn_pin.py
docker commit $cid $TAG && docker rm $cid; SHA=$(docker image inspect -f '{{.Id}}' $TAG)
bash $K/run_tests.sh $TAG                                                # three suites end OK (CPU, --network none)
R4F_PLAN=1 R4F_NAME=r4g-pin5 R4E_IMAGE=$TAG R4E_IMAGE_SHA=$SHA [R4F_ARM_ENV=<r4g arm.env>] bash $L/diag-r4f/run_r4f.sh A <QUAD>
R4F_NAME=r4g-pin5 R4E_IMAGE=$TAG R4E_IMAGE_SHA=$SHA [R4F_ARM_ENV=<r4g arm.env>] [R4E_NEEDLE_BY=..] \
  setsid nohup bash $L/diag-r4f/run_r4f.sh A <QUAD> < /dev/null > /mnt/data1/graded-out/queue/r4g-pin5.console 2>&1 &
```

Arm A = r4e-A2's levers (dd0e7627) plus the ARM lines, including `HPU_GLM53_ATTN_PIN=1`, and no copy-free. r4f-B, with
copy-free, FAILed SELF_DET; r4f-A, with the pin, PASSed. `R4F_ARM_ENV` points `check_inputs` at r4g's ARM file when r4g
has its own; the plan line must list `HPU_GLM53_ATTN_PIN=1`. Leave graded and medium at the defaults, so the medium
profile (about 70K-token prompts, at or above `cold_reuse_min_tokens=8192`) runs after graded. PASS needs all of:
- `SELF_DET arm=r4g-pin5 PASS (schedule matched, outputs identical)` with `sets_identical=3/3` (a FAIL HALTs before
  medium);
- the last `GLM53_ATTN_PIN pinned_sessions=` line in graded/r4g-pin5/container.log with `lookups>0`, `adopted>0`,
  `errors=0` (check with `grep -a 'GLM53_ATTN_PIN pinned_sessions' container.log | tail -1`), plus `PIN arm=r4g-pin5 PASS`;
- `BREAKDOWN r4g-pin5-medium` present (medium ran), with `LOADS window: fence p50` at or below 56.1 ms (r4e-A2).

`PIN_EVICT_RE` in r4e-gates.override.env lists adopt_trim|pressure|cap; add `evicted_idle` there if the eviction count
should include the idle releases (it is reporting only, not a pass gate).

## Files (pin-kit/, lab host, $HOME/.codex/tmp/flash-spikes-20261104/team-20261106-dual/pin-kit/)

| file | goes to | role |
|---|---|---|
| `glm53_attn_pin.py` | `/usr/local/lib/python3.12/dist-packages/glm53_attn_pin.py` | the lever (one module, twelve wrapped methods) |
| `glm53_attn_pin.pth` | `/usr/local/lib/python3.12/dist-packages/glm53_attn_pin.pth` | `import glm53_attn_pin; glm53_attn_pin.install_from_env()` |
| `tests/` | not installed | host suite on the fake KV stack (`python3 -m unittest discover -s tests`) |
| `tests_image/` | not installed | real-import suite inside the image (`run_tests.sh`) |
| `orig/` + `orig/SHA8.txt` | not installed | the r4d bytes the design was read against (read-only) |

sha8 of each shipped file: `SHA8.txt` next to this file (regenerate with `sha256sum glm53_attn_pin.py glm53_attn_pin.pth`).

## Bake into r4e

1. `COPY glm53_attn_pin.py glm53_attn_pin.pth /usr/local/lib/python3.12/dist-packages/` on top of r4d (or `docker cp` into a
   container of r4d and `docker commit`). Nothing else in dist-packages changes; `final-config.env` (40f987bd) is NOT edited.
2. Flag-OFF static guard: `FINAL=<env> bash /mnt/data1/declane-out/c1d-work/r4d_static_guard.sh <r4e> <out>` must match the
   r4d output line for line (the overlay adds two files and touches none).
3. The arm is a NEW env file (copy of 40f987bd plus one line), never an edit of 40f987bd:

   ```
   HPU_GLM53_ATTN_PIN=1
   ```

   Optional: `HPU_GLM53_ATTN_PIN_MAX=64` (pinned sessions kept, LRU beyond; positive int), `HPU_GLM53_ATTN_PIN_LOG_S=30`
   (seconds between counter lines; 0 = every event), `HPU_GLM53_ATTN_PIN_IDLE_TURNS=32` (later finished requests a pin
   may wait for its hit). A bad value refuses the boot at scheduler init.
4. Re-run `bash pin-kit/run_tests.sh <image tag>` (required argument): host suite + real-import suite must both end `OK`.

## Boot refusals (scheduler init raises; the engine does not come up half-armed)

prefix caching ON (`--no-enable-prefix-caching` is the c1d contract) · no OffloadingConnector · no FullAttention KV group at
the hash block size · connector `blocks_per_chunk != 1` · `hash_block_size != block_size` · a pinned group the connector
does not offload (`HPU_GLM53_ATTENTION_ONLY_OFFLOAD` must keep the attention group) · bad `_MAX` / `_LOG_S` value.

## Live counters (M1 proof; REREAD_C1D_PLAN gate "a live path counter")

Boot, once, from the scheduler process:

```
GLM53_ATTN_PIN invalid value for HPU_GLM53_ATTN_PIN=<v>, refused   (only when the flag is not "0"/"1"/unset; stays off)
GLM53_ATTN_PIN armed HPU_GLM53_ATTN_PIN=1 groups=(0,) block_size=128 max_pins=64 log_s=30 k_rule=hit_blocks-1 recurrent=host_sidecar pid=<pid>
```

Then every `_LOG_S` seconds while pins/adoptions happen:

```
GLM53_ATTN_PIN pinned_sessions=<n> pinned_blocks=<n> held_plans=<n> local_hit_tokens=<n> evicted_adopt_trim=<n> evicted_pressure=<n> evicted_cap=<n> evicted_idle=<n> connector_fallbacks=<n> adopted=<n> lookups=<n> pinned=<n> errors=<n>
```

* `pinned` — finished requests whose attention blocks were kept; `pinned_sessions`/`pinned_blocks` — currently held.
* `lookups` — connector hits examined; `adopted` — hits served from a pin; `local_hit_tokens` — tokens NOT reloaded over PCIe
  (the saving: the load path moves 128 tokens x per-token KV bytes per block less). `connector_fallbacks` — hits with no
  matching pin (different session, or the hit was a single block).
* `evicted_adopt_trim` — pinned blocks beyond the adopted k, returned at injection.
* `evicted_pressure` — allocation retry, make-room, dropped pending plans, Scheduler.reset_prefix_cache, reset_cache and the latch release.
* `evicted_cap` — LRU beyond `HPU_GLM53_ATTN_PIN_MAX`.
* `evicted_idle` — pins that saw `HPU_GLM53_ATTN_PIN_IDLE_TURNS` later finished requests without a hit.
* `held_plans` — plans holding a pin for a request not allocated yet (a gauge; a large steady value means waiters
  keep exiting the waiting loop after the lookup). One final counter line is flushed at `Scheduler.shutdown`.
* `errors` — any exception inside the pin: it latches OFF (`GLM53_ATTN_PIN disabled hook=... error=...`), releases every pin
  and the stock path continues. **errors>0 makes the arm INVALID** for the gates below: fix, re-bake, re-run.

PASS needs `adopted>0` and `local_hit_tokens>0` on a multi-turn session, with `errors=0`.

## Gates for r4e (from REREAD_C1D_PLAN.md; nothing here ran on a card yet)

1. det/oracle bitwise at `/mnt/data1/claude-flash-cells/c1d-gate2/c1d-all`: pin ON vs r4d, same seeds — identical. The pinned
   blocks are this session's own bytes. On this harness, with the shared prefix seeded once, output is expected
   identical; in real traffic a CPU-dedup chunk from another writer may differ from them, so the oracle decides.
2. Graded + self-det as for r4d.
3. Medium-speed gate on quad B: decode tok/s at the seat count NOT worse than r4d (C588 guard: this lever adds no per-step
   work and no recurrent copies; a decode loss means HBM pressure from pins is shrinking the batch -> lower `_MAX`, or the
   bake is wrong). TTFT on second turns should drop with `local_hit_tokens`.
4. Flag OFF on the r4e image: `GLM53_ATTN_PIN` must not appear in the boot log; static guard identical to r4d.

## Behaviour under pressure (so a decode-batch regression can be read correctly)

Pinned blocks are free-pool blocks that yield first: `allocate_slots` -> None evicts the LRU pin and retries, and once no
pin is left drops the oldest pending plan (its request loads the stock way). While in-flight prefills hold a reservation,
`schedule()` first computes the gate need over every scanned WAITING request: `reserved + max remaining(r)` over the
waiters stock (`free + held`) would admit at `_glm53_leaves_partials_room`; it then evicts pins (non-matching first, then
matching, then pending plans) until `free` covers that need, so every waiter stock would admit still passes the gate.
`reset_cache` releases all; `_MAX` caps the count. At
16 x ~378K-token sessions the pool (33,280 blocks = 4.26M tokens) cannot hold every session's attention KV beside the
running set, so the hit rate is capacity-bound: expect `evicted_pressure` to climb and only the most recent sessions to hit.

## Uninstall

Remove the two files (or leave them: flag unset = nothing imported, nothing patched). No state outside the scheduler process.
