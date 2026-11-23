# ATTN-PIN prebuild: attention-only session pin for c1d (Fable, 2026-11-08 22:5x EST)

Target image: `claude-flash-dual-graphs-c1d:20261108-r4d` (9e46c471). Work dir (only writer: Fable):
`$HOME/.codex/tmp/flash-spikes-20261104/team-20261106-dual/pin-kit/`. `orig/` holds the r4d bytes read-only
(docker cp from a created container; sha8 in `orig/SHA8.txt`). Nothing in the r4d image, `final-config.env` 40f987bd
or a running script is edited.

## 1. Facts verified on the r4d bytes (file:line in orig/)

| fact | where |
|---|---|
| c1d levers are top-level `glm53_*.py` modules installed by a `.pth` calling `install_from_env()`; unset flag = the module imports nothing and patches nothing | `glm53_loadpath.pth`, `glm53_restore_fence.pth`, `glm53_hbm_margin.pth`; restore_fence.py:140-152 |
| The scheduler asks the connector with a block-aligned local hit of 0 (prefix caching off), then `allocate_slots(..., num_external_computed_tokens=L, delay_cache_blocks=True)`, then `update_state_after_alloc(request, get_blocks(rid), L)` with the LIVE block lists | sched/scheduler.py:1961-1983, 2266-2298 |
| Stock `update_state_after_alloc` treats the leading allocated blocks whose `block_hash is not None` as locally computed: it loads only the chunks from the first unhashed block and still schedules the recurrent sidecar row at `num_cached_tokens` | offloading/scheduler.py:1913-1960, 1967-1984 |
| A load job must keep its attention transfer (`raise "load job cannot omit its attention transfer"`), so an all-local turn cannot be sidecar-only without a worker patch | offloading/worker.py:405-406 |
| First-time allocation with external tokens runs `add_local_computed_blocks` (asserts the request has NO blocks yet, then `num_cached_block[rid]=0`) and then `allocate_external_computed_blocks` (`get_new_blocks(cdiv(total,B) - len(req_blocks))`) | single_type:233-330, coordinator:222-266 |
| `get_num_blocks_to_allocate` counts `len(req_to_blocks[rid])` as already owned | single_type:193-214 |
| Normal finish frees through `KVCacheManager.free(request)` (`last_sched_seq <= processed_step_seq` always true for a normal finish); deferred frees (`pop_blocks_for_free`) only for an abort with a step in flight | sched/scheduler.py:3739-3752 |
| `connector.request_finished` always returns `(False, None)` on c1d; the finished request's store jobs are built next step from `finished_req_ids`; a pending store on a re-allocated block is flushed via `_current_batch_allocated_block_ids` | offloading/scheduler.py:2987-3058, 2752-2764, 1906-1908 |
| Lookup of a hit depends on the finished stores: HIT_PENDING / RETRY defer the request, so when L is known every chunk below L is stored (quiescence by construction) | offloading/scheduler.py:1741-1748, 1230-1237 |
| With caching off: `get_new_blocks` never evicts hashes, `free_blocks` puts unhashed blocks LIFO, `cache_blocks` is a no-op, coordinator is `KVCacheCoordinatorNoPrefixCache` | block_pool.py:671-677, 719-743; kv_cache_manager.py:771; coordinator:420 |
| Groups: MLA full attention (`FullAttentionManager`, block 128), `KpoolTailManager` (1 block/request, circular, never prefix-cached), Mamba/KDA (`MambaSpec`, recurrent row on the host sidecar) | single_type:684, 1114-1239; kv_cache_interface |
| HBM: pool 33,280 x 128 = 4.26M tokens; pins live inside it (no new HBM); recurrent copy 563 MiB/rank does not fit 511 MiB headroom | REREAD_C1D_PLAN.md §0, §2 |
| Live config: `--no-enable-prefix-caching`, `request_hash_block_size 128`, `cold_reuse_min_tokens 8192`, `VLLM_RECURRENT_SIDECAR_PROMPT_END_ONLY=1`, `GLM53_LOADPATH_V2=1`, `HPU_GLM53_ATTENTION_ONLY_OFFLOAD=1` set in both live containers | REREAD_C1D_PLAN.md §0; diag-twonode/final-config.env |

## 2. REUSE vs NEW

| piece | status | source | decision |
|---|---|---|---|
| `.pth` + `install_from_env()` + meta-path `_PatchOnImport` finder (patch on import, no vllm import when off) | LIVE in r4d | glm53_restore_fence.py 723d2d12, glm53_loadpath.py 23036c88 | REUSE the pattern verbatim (multi-module finder) |
| stock "hashed leading blocks are local" split in `update_state_after_alloc` | LIVE in r4d | offloading/scheduler.py:1913-1960 | REUSE as-is: the connector split is the stock code; the pin only supplies hashed blocks |
| stock recurrent-sidecar load at `num_cached_tokens` + LOADPATH_V2 worker | LIVE | offloading/scheduler.py:1967-1984, glm53_loadpath.py | REUSE untouched (the recurrent row keeps today's path) |
| stock "blocks already owned" accounting (`len(req_to_blocks)` in alloc) | LIVE | single_type:193-214, 323-329 | REUSE: injected blocks are owned blocks |
| w-pin-rc2 `test_flag_off_identity.py` (AST identity of patched files) | BUILT (Full RC2) | work-in/w-pin-rc2/tests | REUSE the idea; here it is trivial: NO image file is modified, identity = function objects unchanged (test asserts it) |
| w-pin-rc2 `uncached_per_turn.py` (10/10) | BUILT | w-pin-rc2 | REUSE later as the M1 reader (needs its adopt-line regex re-pointed at `GLM53_ATTN_PIN`); not needed for the bake |
| w-pin-rc2 LRU evict order, quiescence gate, boot refusal list (D4), counters-per-turn log | BUILT | session_pin_adopt.py 8ad436a4 | REUSE the rules, re-implemented in ~1/4 the code (no slot ops, no snapshot, no detach) |
| w-pin-rc2 hpu_session_pin.py slot copy, snapshot/detach/adopt ops, 13-file overlay | BUILT | w-pin-rc2 patched/ | DROP: recurrent copy does not fit HBM; c1d bases differ (decode_priority, LOADPATH_V2, compact pool) |
| PIN-ADOPT-PLAN A4 connector bookkeeping (`num_locally_computed_tokens=L`) | designed | KB 21280-21286 | NOT NEEDED: the lookup still runs with local=0 and L comes from the stock connector |
| FULL53-FUSED-KEYSPACE-PLAN (async KV load), FULL53-DUAL-STEP-PLAN | Full plans | KB D.29/D.30 | NOT PORTED: Full-only key space and dual step; c1d has its own graphs/dual stack and LOADPATH_V2 already does async loads |
| overlay test shape (`tests/*.py` unittest, run in the image with `--network none`, `HABANA_VISIBLE_DEVICES=none`) | LIVE | team overlay/run_tests.sh, check_flags_off.py | REUSE: pin-kit `run_tests.sh` mirrors it against r4d |
| attention-only pin registry (pin on free, adopt on hit, LRU make_room, counters, refusals) | NEW | pin-kit/glm53_attn_pin.py | the lever |

## 3. C588 comparison (the Full L4 pin lost: decode 4.2 -> 3.4 at 16 seats, -4.5% uncached, 168 fence waits)

| C588 mechanism | ATTN-PIN |
|---|---|
| decode-end adoption copied/moved compact recurrent slots per turn (snapshot + detach ops at the step head, fence waits stalled steps) | no recurrent op at all: the recurrent row keeps today's sidecar load; the pin only keeps attention blocks already in HBM |
| per-step `session_pin_ops` drained on EVERY `schedule()` and applied in the worker before `_update_states` | zero per-step work on the decode path: hooks fire only at request finish, at the connector hit of a NEW request, and when `allocate_slots` returns None |
| the saving was 4.5% of uncached tokens (recompute) | the saving is the reload: ~1.64 GB/turn + ~0.9 s restore fence at long context (REREAD §1); recompute is already ~99 tokens/turn |
| prompt-end adoption never fired (slot ledger 16+4) | no slot involved; the recurrent slot ledger path is untouched |
| HBM pressure risk | pins are pool blocks that yield first (LRU) on any `allocate_slots` None and ahead of the mixed-step admission gate; cap `HPU_GLM53_ATTN_PIN_MAX` |
| gate | r4e medium-speed gate must show decode tok/s not worse than r4d at 16 lanes; an arm with `errors>0` is INVALID |

## 4. Mechanics (how a turn runs with `HPU_GLM53_ATTN_PIN=1`)

1. R1 finishes normally -> `KVCacheManager.free(R1)` wrapper: `n = min(len(R1.block_hashes), processed_tokens // 128)`;
   for every FullAttention (non-KpoolTail) group pop `req_to_blocks[R1][:n]` into a `PinEntry(hashes[:n], blocks per group)`;
   the stock free releases the remainder (partial last block, decode blocks, KpoolTail, Mamba). Nothing is copied. The
   pinned blocks keep ref_cnt 1 and carry no hash.
2. R2 arrives -> stock `get_num_new_matched_tokens(R2, 0)` returns L (CPU-store lookup, sidecar gate, as today). The
   wrapper then finds the pin with the longest hash-prefix match `k_match` against `R2.block_hashes`, caps
   `k = min(k_match, L // 128 - 1)` (one block always stays on the connector so the load job keeps its attention
   transfer, worker.py:405), consumes the pin into an `AdoptPlan`, and returns (L, async) unchanged. Graph keys are
   unchanged: same L, same suffix.
3. `allocate_slots`: `get_num_blocks_to_allocate` is discounted by k per pinned group; after the stock
   `add_local_computed_blocks` (asserts empty, sets `num_cached_block[R2]=0`) the hook appends the k pinned blocks to
   `req_to_blocks[R2]`; the stock `allocate_external_computed_blocks` then allocates `cdiv(L,128) - k` fresh blocks.
4. `update_state_after_alloc(R2, blocks, L)` wrapper: sets `block._block_hash` on the first `min(k, cdiv(L,128)-1)`
   blocks of each pinned group, calls stock (it loads chunks [k, n) + the recurrent row at L, exactly the bytes the
   store copied), then resets the marker hashes directly in its finally. Marking is all-or-nothing: every scanned
   block must be unhashed first (with caching off get_new_blocks never clears a hash, block_pool.py:671-677), and any
   failure resets every marker before the latch. Not bitwise by construction: the adopted blocks are this session's
   own bytes, while the CPU store dedups by block hash, so the load path may read a chunk first stored by another
   request with the same hash chain. Equality with the load path is the det/oracle gate's to prove.
5. Pressure: `allocate_slots` None -> evict the LRU pin (`block_pool.free_blocks`) and retry, until success or no
   pins, then drop the oldest pending (not yet injected) plan: plans outlive the waiting-loop exits at scheduler.py
   1985/2075 (continue) and 2261-2263/2280 (break), and a dropped plan's request reads no discount and loads stock.
   `schedule()` entry: while a reservation is held, evict pins (non-matching first, then matching, then pending
   plans) until `free - reserved >= max remaining(r)` over the scanned waiters that stock (`free + held`) would admit.
   `allocate_slots` entry settles a pending plan to the external tokens it really loads: the hybrid fail-closed
   path (scheduler.py:1999-2010) or a kept local tail zeroes them, and the pin goes back with no discount. A plan whose
   request is freed before allocation returns its pin. `reset_cache` releases every pin. Cap `HPU_GLM53_ATTN_PIN_MAX` (default 64) evicts LRU.
6. Counters: `GLM53_ATTN_PIN armed ...` at scheduler init; then at most every `HPU_GLM53_ATTN_PIN_LOG_S` (30) s on an
   event, and once more at `Scheduler.shutdown`:
   `GLM53_ATTN_PIN pinned_sessions= pinned_blocks= held_plans= local_hit_tokens= evicted_adopt_trim= evicted_pressure= evicted_cap= connector_fallbacks= adopted= lookups= pinned= errors=`.
7. Refusals at scheduler init (actionable RuntimeError): prefix caching on; no OffloadingConnector; no FullAttention
   group with block size == hash block size; `blocks_per_chunk != 1`; bad env values.

## 5. Blast radius (flag ON only; flag OFF = no finder, no import, every function object identical)

Patched by wrapper (originals kept under `_glm53_attn_pin_originals`): `Scheduler.__init__` (post: build registry),
`Scheduler.schedule` (pre: make_room), `Scheduler.shutdown` (pre: final counter line), `KVCacheManager.free`,
`KVCacheManager.allocate_slots` (pre: settle the plan; retry after evict),
`KVCacheCoordinator.get_num_blocks_to_allocate` (discount), `SingleTypeKVCacheManager.add_local_computed_blocks`
(post: inject), `OffloadingConnectorScheduler.get_num_new_matched_tokens` (post: plan), `.update_state_after_alloc`
(mark/unmark), `.reset_cache` (release all). Scheduler process only; the worker processes get the same import hook but
never instantiate these objects. No file in dist-packages changes: the overlay adds `glm53_attn_pin.py` + `.pth`.

## 6. Open risks

- R1 (coordinator): the 08:00 base bytes (r4c/r4b live, r4d built).
- The `_current_batch_allocated_block_ids` flush: R2's adopted blocks enter that set (stock line 1906), so any
  pending store job of R1 that still names them is flushed (synchronous wait) on R2's admission. By the lookup's
  pending-store gate those stores are complete when L is known, so the flush is a no-op; a real wait would show as
  `GLM53_ATTN_PIN` adopted turns with longer TTFT, not as wrong bytes.
- Partial-tail hits (`supports_partial_tail`) keep `cdiv(L,128)` blocks; the cap `k <= cdiv(L,128)-1` holds.
- Not exercised on a card tonight (owner rule: no boots). The r4e gates: static guard flag OFF line-for-line, boot
  line `GLM53_ATTN_PIN armed`, det/oracle bitwise vs load path, medium speed gate decode not worse.

## 7. Build notes (22:5x ET, after RED -> GREEN on the host)

- RED logged with the module absent (`logs/red-host.log`), GREEN 25/25 then 26/26 after the fixes below (`logs/green.log`).
- make_room policy settled against r4d bytes: the only pre-allocation gate that a pin can block is
  `_glm53_leaves_partials_room` (scheduler.py:1115-1122, `free - reserved >= _request_remaining_blocks(waiter)`), live only
  while in-flight prefills hold a reservation and run BEFORE the connector lookup. So the pre-hook uses exactly that
  formula for every scanned WAITING request (the mixed slot pick runs it per waiter, scheduler.py:1104), evicts pins
  until each waiter stock would admit passes, and evicts the pins a waiter matches last (lost only when keeping them
  would starve a waiter: no livelock). Review round 1 (Opus, eefd6499) found the first-waiter-only version blocked a
  second waiter; fixed with the probes in `tests/test_review_probes.py`. With no
  reservation the hook is idle and `allocate_slots` -> None -> evict LRU -> retry frees exactly what the allocation needs.
- Pinnable = exactly `FullAttentionManager` (type name), not its subclasses: KpoolTail (one circular block), RSWA/Sink
  (windows) recycle blocks. KpoolTail overrides `add_local_computed_blocks` (single_type:1206); harmless, it is never pinned.
- Adoption arithmetic: a pin of n blocks adopted with hit L = n*B gives k = n-1 local blocks, 1 fresh block for the last hit
  chunk (worker.py:405 keeps the attention transfer), and the n-th pinned block goes back to the pool (counted in
  `evicted_adopt_trim`); net HBM held grows by 1 block + the KpoolTail block.
- Capacity bound (C588 reading): 16 x ~378K tokens = 6.0M tokens > the 4.26M-token pool, so pins cannot hold every session
  beside the running set; LRU keeps the most recent. The r4e speed gate reads decode first (must not drop), TTFT second.
