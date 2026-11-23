# GLM-5.3-Flash on Intel Gaudi 2 with vLLM (FP8, 524K Context)

`gaudi2-glm-5.3-flash` · [all Gaudi 2 recipes](https://github.com/LiquidLava1990)

Build and serving recipes for GLM-5.3-Flash (FP8, abliterated variant) on 8x Intel Gaudi 2, run as two TP4 vLLM replicas. This is the production setup: 524,288-token context, host-RAM KV cache offload, and an attention pin that keeps a finished turn's leading KV blocks on the card for the next turn. The previous champion (2026-11-12) measured 10.76 tok/s per agent with 16 concurrent agents per replica. It is a working example of serving a large mixture-of-experts LLM on Habana Gaudi HPUs.

## Status

- Current production champion: cut `r4h8` with the attention pin and card-headroom levers (build `pin11`), serving since 2026-11-22 19:54 EST.
- Image `claude-flash-dual-graphs-c1d:20261122-pin11hr2-r4h8` (also tagged `...:20261122-r4h8-pin11hr2`), id `sha256:0e5df6b75e7a75c9389b0f6dee3c0ed5f909d3cd0b16e9753559c182b7388414`. It is the 2026-11-12 champion image `claude-flash-dual-graphs-c1d:20261112-r4h8` (id `sha256:8bf1872d93c1...`) plus four changed Python files in `/usr/local/lib/python3.12/dist-packages`. Nothing else in the image changed. The files are in `overlay-pin11/`.
- The champion is pinned by three files in `champion-inputs/` plus that image. `final-config.env` is unchanged. `levers.env` adds six lines, and `gates.override.env` turns the pin gate on. The 2026-11-12 inputs are in `champion-inputs/prev-20261112-r4h8/`.
- Tuning of the base config stopped on 2026-11-12. This repo is a record of what runs, not an active tuning branch.
- Model weights are not in this repo. Download them separately.
- This is a snapshot of scripts as they ran on one lab host. They contain host paths you must change (see [Paths to change](#paths-to-change)) and rely on a few things that are not in the repo (see [Known issues](#known-issues)).

## Results

Topology of the champion (cut r4h8): 2 replicas, each on one 4-card group (TP4), 16 concurrent sequences per replica (32 total), 524,288-token context.

### Current champion (r4h8-pin11, measured 2026-11-22)

| Test | Result |
|---|---|
| Attention pin, 16 conversations x 6 turns | Outputs identical to the 2026-11-12 champion on 96/96 turns. RAM-to-card KV reload per run: 31.8 GB -> 0.046 GB. |
| Overflow: 8 conversations of 444-493K tokens grown to the 524K limit, one replica | 342 turns, 0 errors. Peak card KV use 86.9%. 330,507 pinned blocks released under pressure. First token p50 10.8 s (about 5 s when everything fits on the card). |
| Stress: 4 conversations of 250-300K tokens | 430 turns, no errors, first token p50 3.9 s |
| Stress: 4 conversations of 450-500K tokens | 170 turns, no errors, first token p50 5.3 s |
| 16 lanes, quality-gate workload | First token p50 about 19.8 s, decode p50 15.3 tok/s (see [Known issues](#known-issues)) |
| Card KV capacity per replica | 30,880 blocks of 128 tokens (about 3.95M tokens); was 33,280 (about 4.26M) |
| Seeded boot | about 545-590 s |

### 2026-11-12 champion (r4h8, history)

These numbers are for the previous champion image `claude-flash-dual-graphs-c1d:20261112-r4h8` and its inputs (`champion-inputs/prev-20261112-r4h8/`). They were not re-measured on the pin11 build.

Measured on the champion (cut r4h8, 2026-11-14), 16 concurrent agents on one replica:

| Metric (r4h8) | Value |
|---|---|
| Decode speed per agent, 16 agents on one replica | 10.76 tok/s |
| First token for a short reply queued behind a long prompt | 4.4 s |
| Cold 150k-token prompt | 305 s |
| Hot KV capacity per replica | 33,280 blocks of 128 tokens (about 4.26M tokens) |
| Quality gate (graded set, both live replicas) | no loss |

How it got there (same 16-agent test, per replica):

| Cut | Decode per agent | Notes |
|---|---|---|
| r4h6e-s7b | 9.11 tok/s | first-token time 16.4 s |
| r4h7 | 9.92 tok/s | aggregate 78.3 tok/s per replica |
| r4h8 | 10.76 tok/s | decode-priority gap 8 (was 16); cold 150k prompt 305 s (was 415 s) |

The goal for this model was at least 10 tok/s per agent with 16 agents on one 4-card replica; r4h8 cleared it.

## Hardware & versions

| Item | Value |
|---|---|
| Accelerators | 8x Gaudi 2 HL-225, 96 GB HBM each |
| Host | 2-socket Xeon, ~500 GB RAM |
| Driver | 1.24.2 |
| Card firmware | 62.6.2 |
| Base image | `vault.habana.ai/gaudi-docker/1.24.2/ubuntu24.04/habanalabs/pytorch-installer-2.13.0` |
| Serving stack | vLLM with the vllm-gaudi plugin |

Card layout: replica `r0` runs on quad A (modules 4-7, port 18274); replica `r1` runs on quad B (modules 0-3, port 18275).

## Quick start

Run on the Gaudi host, from this folder. A boot seeded from the recipe cache of an earlier serving boot of the same image takes about 545-590 s. The first boot of a new image (or any boot with no seed) compiles every graph and takes about 15 minutes per replica.

### 0. Prepare the host

1. Set the paths listed in [Paths to change](#paths-to-change).
2. In `promote-r4h/serve_champion.sh`, set `L` (line 16) to the folder that holds this repo's `lab_serve_seeded.py`, `lab_serve.py` and `diag-r4e/` (so `$L/lab_serve_seeded.py` and `$L/diag-r4e/r4e_run.sh` exist). `serve_champion.sh` boots through `LABPY=$L/lab_serve_seeded.py`.
3. `lab_serve.py` copies its settings from two existing stopped containers named `glm53flash-r0` and `glm53flash-r1` (the "source units"). They carry the vLLM arguments and offload sizing and are not in this repo. See [Known issues](#known-issues).

### 1. Build the image

```bash
./build_image.sh                 # default tag claude-flash-dual-sustain:20261106
./build_image.sh my-tag:latest   # or choose a tag
```

It checks free disk (100 GB on the Docker root and on the data disk), refuses to touch protected images, builds from `overlay/` with `install_overlay.sh`, runs the differential tests, checks that new flags default to off (`check_flags_off.py`), and prints `BUILD_RESULT: PASS` or `BUILD_RESULT: FAIL`. Log: `build_image.log`. It starts no card.

That builds the older `claude-flash-dual-sustain` lane, not the champion. The champion is the 2026-11-12 image plus `overlay-pin11/`:

```bash
./build_pin11.sh                                         # default tag claude-flash-dual-graphs-c1d:20261122-pin11-r4h8
./build_pin11.sh claude-flash-dual-graphs-c1d:<date>-<variant>-r4h8
```

It needs `claude-flash-dual-graphs-c1d:20261112-r4h8` (id `8bf1872d93c1...`) on the host. `Dockerfile.pin11` copies `overlay-pin11/` onto it with `install_overlay.sh`, which first checks that the replaced files have the hashes in `overlay-pin11/BASE_SHA256.txt`. The script then checks the installed bytes against `overlay-pin11/SHA256SUMS` and runs `overlay-pin11/run_tests.sh <tag>`: host suite, style check, and the real-import suite in a `--network none` container with no card. Log: `build_pin11.log`. It starts no card.

The dist-packages files come out identical to the production image's, but the image id does not, because the build adds `/opt/pin11-overlay` and a label. Production was made by copying the files into a container of the old image and committing it (see `overlay-pin11/INSTALL-NOTES.md`). Keep the tag ending in `-r4h8`: `promote_r4h.sh stage` takes the cut from the last `-` suffix.

Tests that run without the image (any OS with Python 3.10+):

```bash
cd overlay-pin11 && PYTHONPATH=$PWD python3 -m unittest discover -s tests -t tests    # attention pin host suite (78 tests)
python3 tests/check_craft.py glm53_attn_pin.py tests/*.py tests_image/*.py           # style check
cd .. && python3 test_lab_serve_seeded.py lab_serve_seeded.py                         # seeded launcher (9 tests)
```

`overlay-pin11/tests_headroom/` holds the tests the headroom files were developed against: the 2026-11-12 image's own suite plus `test_breadcrumb.py` and `test_workspace_release_serial.py`. They need Linux, pytest and torch. Run them from `<copy>/tests/`, where `<copy>` is a copy of the 2026-11-12 image's dist-packages with the `overlay-pin11/` files laid on top. `promote-r4h/test_*.py` need Linux (bash, coreutils). Run them from `promote-r4h/` with `python3 -I test_champion_boot.py` and `python3 -I test_promote_kit.py`.

### 2. Stage the champion

Staging pins the image, the three env files and their sha256 into `champion.staged`. `--image-sha` is the image id from `docker image inspect -f '{{.Id}}' <image>`.

```bash
cd promote-r4h
./promote_r4h.sh stage \
    --image claude-flash-dual-graphs-c1d:20261122-pin11hr2-r4h8 \
    --image-sha sha256:0e5df6b75e7a75c9389b0f6dee3c0ed5f909d3cd0b16e9753559c182b7388414 \
    --levers ../champion-inputs/levers.env \
    --gates ../champion-inputs/gates.override.env \
    --final-env ../champion-inputs/final-config.env
```

### 3. Boot the replicas

```bash
./champion_boot.sh r0 --plan     # print what a start would do
./champion_boot.sh r0
./champion_boot.sh r1
```

Or promote both in one staggered run (r1 starts only after r0 passes boot, warmup, live-parity and smoke gates; any failure rolls back). Dry run first:

```bash
./promote_r4h.sh --dry-run
setsid nohup ./promote_r4h.sh </dev/null >/dev/null 2>&1 &    # log: logs/promote_r4h-latest.log
```

Other modes: `./champion_boot.sh r0 --stop` (stay down), `--recover` (what the timer runs), `--pin-acc` (re-apply the AccThreadPool cpu pins), `--keep-seed` (copy the running replica's recipe leaf to its protected seed copy), `./serve_champion.sh r0 --check` (pre-boot checks only, ends in `CHAMPION_CHECK_OK` or a `HALT` line), `./rollback_r4f.sh [--dry-run]`.

`promote_r4h.sh` with no arguments only promotes from the older r4f install. If the replicas already serve the staged image under the cut's names, adopt them with no restart instead. `./adopt_check.sh [--smoke] [--launcher-check]` is read-only. For each running replica it checks the staged image id, every staged lever (VALUE-DRIFT 0), the kit's KV block count and `/health`, and it ends in `ADOPT_OK` or `ADOPT_REFUSED`. The full procedure, including the kit install and rollback to the 2026-11-12 champion, is in `promote-r4h/ADOPT_RUNBOOK.md`. It was written before it ran, and its paths are the original lab host's.

Each boot that serves records its recipe-cache leaf. The next boot of the same image, on either replica, is seeded from that leaf, or from the protected `recipe_cache/seed-<sha12>` copy if the leaf is gone. A boot of a new image, or one with no seed, is a clean boot. After a serving boot the kit pins the replica's 4 AccThreadPool threads to one cpu each on the replica's own socket (`ACC_PIN_DONE ... pinned=4/4`).

### 4. Check health

```bash
curl -fsS http://<SERVER_IP>:18274/health   # r0
curl -fsS http://<SERVER_IP>:18275/health   # r1
```

### 5. Send a request

The scripts contain no request example, so this uses the standard vLLM OpenAI-compatible route. Take the model name from `/v1/models`. Add the `Authorization` header only if your deployment sets a key.

```bash
curl http://<SERVER_IP>:18274/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d '{"model": "<MODEL_NAME>", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 256}'
```

## What's in this repo

| Path | What it does |
|---|---|
| `Dockerfile` | Copies `overlay/` onto the base image and runs `install_overlay.sh`. |
| `build_image.sh` | Rerunnable image build with disk check, protected-image check and test stages. |
| `Dockerfile.pin11`, `build_pin11.sh` | Build the champion image: `overlay-pin11/` onto `claude-flash-dual-graphs-c1d:20261112-r4h8`, then a byte check and the pin suites. |
| `install_overlay.sh` | Runs in `docker build`: verifies base file hashes, then installs the overlay into site-packages. Used by both Dockerfiles. |
| `check_flags_off.py` | Checks that new overlay flags default to off in the built image. |
| `overlay/` | Patched vLLM, vllm-gaudi and `glm53_*` modules; `BASE_SHA256.txt` lists the base hashes. `tests/` and `cells/` stay out of the image. |
| `overlay-pin11/` | The champion's changes to the 2026-11-12 image: `glm53_attn_pin.py` (its `.pth` is unchanged), `glm53_mixed_graph.py`, `glm53_hbm_margin.py`, `vllm_gaudi/v1/worker/hpu_worker.py`. `BASE_SHA256.txt` lists the 2026-11-12 bytes they replace, `SHA256SUMS` the installed bytes. `INSTALL-NOTES.md`, `prebuild.md` (pin design), `run_tests.sh`, `tests/`, `tests_image/` and `tests_headroom/` stay out of the image. |
| `champion-inputs/levers.env` | Serving levers for cut r4h8-pin11 (applied after final-config, later wins). |
| `champion-inputs/gates.override.env` | Gate settings, served context length, default reasoning level, pin gate. |
| `champion-inputs/final-config.env` | Base serving config, including the KV offload load-path settings. |
| `champion-inputs/prev-20261112-r4h8/` | The 2026-11-12 champion's levers and gates, for rollback. |
| `lab_serve_seeded.py` | The launcher the champion boots through: `lab_serve.py` plus `LAB_RECIPE_SEED_LEAF`, which seeds a new recipe leaf from one that served. `test_lab_serve_seeded.py` tests it. |
| `lab_serve.py` | Launcher: boots a TP4 container on quad A or B from a copy of the source unit's settings. |
| `diag-r4e/` | `r4e_run.sh` (prepare and boot), `r4e_gates.sh`, probes and tests used by the launcher. |
| `promote-r4h/promote_r4h.sh` | Stages a champion and promotes it to both replicas, with gates and automatic rollback. |
| `promote-r4h/rollback_r4f.sh` | Returns both replicas to the previous (r4f) serving install. |
| `promote-r4h/serve_champion.sh` | Boots one replica from the staged files; `--check` runs pre-boot checks only. |
| `promote-r4h/champion_boot.sh` | Keeps a replica up: start, `--plan`, `--stop`, `--recover`. |
| `promote-r4h/adopt_check.sh` | Read-only check that the running replicas are the staged champion, so it can be adopted with no restart. |
| `promote-r4h/ADOPT_RUNBOOK.md` | The no-restart adopt procedure written for the 2026-11-22 promotion, with rollback. |
| `promote-r4h/common.sh`, `champion_check.sh` | Shared constants and helpers (ports, quads, budgets, AccThreadPool pins); image-hash check that starts no container. |
| `promote-r4h/tools/docker-alias-preview.sh` | Test-only docker wrapper that previews a stage tag that does not exist yet (`GLM53_PROMOTE_TEST=1`). |
| `promote-r4h/test_*.py`, `SHA256SUMS*` | Tests and checksums for the kit. |

## Settings that matter and why

Values are the champion replicas' settings (cut r4h8, read with `docker inspect` on the 2026-11-12 champion), plus the files in `champion-inputs/`. Rows marked 2026-11-22 are the pin11 changes.

| Setting | Value | Why |
|---|---|---|
| Tensor parallelism | `--tensor-parallel-size 4` per replica | One 4-card quad each; 2 replicas use all 8 cards. |
| Context | `--max-model-len 524288` (`LAB_MAX_MODEL_LEN`) | The image refuses a larger value beside these prompt keys. |
| Concurrency | `--max-num-seqs 16` per replica (32 total) | Matches the top decode key of 16 rows. |
| Batched tokens | `--max-num-batched-tokens 8192` | Per-step token budget. |
| HBM use | `--gpu-memory-utilization 0.90` | Leaves headroom on each card. |
| Prefix caching | `--no-enable-prefix-caching` | Must stay off: it made this model incoherent. |
| Default reasoning | `LAB_DEFAULT_REASONING_EFFORT=high`, `VLLM_REASONING_AUTO_BUDGET=1`, `VLLM_REASONING_AUTO_MAX_TOKENS=4096`, `VLLM_REASONING_MAX_ANSWER_TOKENS=8192` | Applies to requests that name no level. The chat template reads only `low` and `high` by name; any other value is max. Disabling reasoning returns 0 reasoning tokens, and mixed reasoning broke attention. |
| KV blocks per replica | 30880 served (`C1D_KV_BLOCKS` in `serve_champion.sh`; the 2026-11-12 champion served 33280), block size 128 | About 3.95M tokens of card KV (was about 4.26M). The launcher refuses a source unit whose `--num-gpu-blocks-override` is not 31264, then boots at the kit's count. |
| Host offload size | `VLLM_CPU_OFFLOAD_TOTAL_BYTES=65000000000`, `CPU_POOL_BYTES=135000000000` per replica | KV and recurrent state offload to host RAM over PCIe. |
| Offload loading | `HPU_KV_OFFLOAD_ASYNC_LOAD=1`, `HPU_KV_OFFLOAD_STAGE_MB=128`, `HPU_KV_OFFLOAD_MAX_INFLIGHT=2`, `HPU_KV_OFFLOAD_PIN_STAGE=1`, `GLM53_LOADPATH_V2=1`, `GLM53_LOADPATH_THREADS=4`, `GLM53_LOADPATH_DEPTH=8`, `HPU_GLM53_OFFLOAD_ROW_BUCKETS=1` | Pinned staging moves offloaded KV back with bounded in-flight copies. |
| Recurrent state | `VLLM_RECURRENT_SIDECAR_PROMPT_END_ONLY=1`, `VLLM_RECURRENT_TAIL_FUSE=1`, `VLLM_RECURRENT_TAIL_FUSE_KEYED=1` | Levers file overrides the final-config tail-fuse value of 0. |
| Decode priority | `VLLM_DECODE_PRIORITY=phases`, mixed on, min decode steps 8, max prefill wait 8 s | Prevents long prefills from stalling decode. |
| Mixed prefill | `VLLM_GAUDI_MIXED_PREFILL_TOKENS=1536` | Prefill chunk size. |
| HBM margin | `HPU_GLM53_HBM_MARGIN_MIB=512` | Headroom on each card. |
| Attention pin (2026-11-22) | `HPU_GLM53_ATTN_PIN=1`; `R4E_PIN_GATE=on` in the gates file | Keeps a finished turn's leading KV blocks on the card. A pin serves one follower (owner-only adoption). Pins are released first under card-memory pressure. Optional `HPU_GLM53_ATTN_PIN_MAX`, `_LOG_S`, `_IDLE_TURNS`: see `overlay-pin11/INSTALL-NOTES.md`. |
| Card headroom (2026-11-22) | `PT_HPU_POOL_MEM_ACQUIRE_PERC=99`, `VLLM_GAUDI_MIXED_GRAPH_RESERVE_MIB=1024`, boot guard `HPU_GLM53_POOL_LIMIT_MAX_MIB=96400`, KV blocks 30880 | Leaves card memory outside the PT pool. The boot guard refuses a pool limit above 96,400 MiB. |
| Key workspace release (2026-11-22) | `VLLM_GAUDI_MIXED_KEY_WORKSPACE_RELEASE_MODE=serial` | One rank releases at a time instead of all ranks at once. |
| Boot breadcrumbs (2026-11-22) | `HPU_GLM53_BREADCRUMB_FILE=/recipe_cache/glm53-breadcrumbs.log` | One synced line per boot step on a persistent mount, so a host crash during boot shows the last step reached. |
| AccThreadPool pins (2026-11-22) | `ACC_CPU_FIRST` in `common.sh` (r0 from cpu 50, r1 from cpu 14) | After a serving boot, each of the 4 spinning AccThreadPool threads gets one cpu on the replica's own socket. Needs `sudo -n taskset`. |
| Recipe-cache seed (2026-11-22) | `LAB_RECIPE_SEED_LEAF` (set by `champion_boot.sh`) | Boots seed the recipe cache from a leaf of the same image. A new image or a missing seed gives a clean boot. |
| Restore fence | `GLM53_RESTORE_FENCE_MODE=sync` | Async design stays off. |
| Boot / warmup budget | 1500 s / 300 s | Gates in `common.sh`. |
| Auto-restart | 3 tries per 10800 s | `champion_boot.sh --recover`; a replica stopped by hand stays down. |

## Paths to change

These are hard-coded for the original lab host. Change each to your own path.

| Path | Where | Used for |
|---|---|---|
| `/mnt/data1` | `build_image.sh`, `promote-r4h/common.sh`, `champion_boot.sh`, `lab_serve.py`, `lab_serve_seeded.py`, `diag-r4e/r4e_run.sh` | Data disk, run/queue folders, recipe cache. In `champion_boot.sh`, `QUEUE_DIR` overrides the queue folder and `RECIPE_ROOT` the recipe cache. |
| `/var/lib/docker` | `build_image.sh` | Docker root for the disk check. |
| `$L` (a folder under `$HOME` on the original host) | `promote-r4h/serve_champion.sh` line 16 | Folder holding `lab_serve_seeded.py`, `lab_serve.py` and `diag-r4e/`. |
| `ACC_CPU_FIRST=([r0]=50 [r1]=14)` | `promote-r4h/common.sh` | First cpu of each replica's AccThreadPool pins. Pick 4 free cpus on the socket nearest each quad. |
| `$HOME/...` paths | `promote-r4h/ADOPT_RUNBOOK.md`, `overlay-pin11/INSTALL-NOTES.md`, `overlay-pin11/prebuild.md` | Where the kit and trial folders lived on the original host. |
| `/opt/glm53/buckets/*.buckets` | `champion-inputs/levers.env` | Bucket file inside the image. |

## Known issues

- **Source units are not in the repo.** `lab_serve.py` copies the arguments, environment and devices of containers `glm53flash-r0` and `glm53flash-r1`. The `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`, prefix-caching flag, offload sizes and `CPU_POOL_BYTES` listed above come from them and are not in any file here, so this repo alone does not reproduce the champion.
- **Stale comment.** The `serve_champion.sh` header says "1M context"; the champion serves 524,288. `ADOPT_RUNBOOK.md` names the trial's candidate tags (for example `pin9hr2s1`, cand5); the promoted image is pin11.
- **Build.** The Dockerfile starts `FROM` a local image id (`sha256:1481bb99...`), not directly from the vault.habana.ai image. `install_overlay.sh` aborts unless `.dockerignore` keeps the tests out of the build context, so build from this folder. `build_image.sh` produces `claude-flash-dual-sustain`. How `claude-flash-dual-graphs-c1d:20261112-r4h8` was built is not in this repo, and `build_pin11.sh` needs that image.
- **KV block count.** `serve_champion.sh` exports `C1D_KV_BLOCKS=30880`. The lab's `diag-graded/qlib.sh` turns it into `lab_serve_seeded.py --lab-kv-blocks`. That file is reached from `diag-r4e/r4e_run.sh` through `FINAL_RUN_SH` and is not in this repo.
- **Interim builds hang.** Two interim builds of 2026-11-22 (tags ending `pin9hr2` and `pin10hr2`) hang under card-memory pressure. Waiting prompts parked in the scheduler's `skipped_waiting` queue were not seen by the pin's make-room scan. pin11 fixes this: make-room uses the scheduler's own waiting-prompt list. Do not serve the earlier builds.
- **Recurrent checkpoint store.** The recurrent checkpoint RAM store holds 304 entries, one per finished turn. A conversation that sits idle for about 300 other turns is recomputed in full.
- **16-lane first token.** With 16 lanes under the gate's workload, first-token p50 is about 19.8 s on this build, because prompts queue behind one prompt per mixed step. Decode p50 is 15.3 tok/s.
- **Host machine checks.** The host this was developed on still has unexplained fatal CPU machine checks (Bank 6, IIO/PCIe) during replica boots. The headroom levers and the serial workspace release were added while chasing them. They are not proven to fix them.
- **Checksums.** `promote-r4h/SHA256SUMS` lists the lab bytes. `serve_champion.sh` differs from them only on line 16 (`L=`), where the original home path became `$HOME`.
- **`common.sh` typo.** `RC56_IMAGE` is 65 hex characters, one more than a sha256.
- **Network exposure.** Replicas bind `0.0.0.0` on ports 18274 and 18275. Firewall them or put them behind an authenticated proxy.
- **Weights.** No download script is provided.

## Related repos

- [gaudi2-common](https://github.com/LiquidLava1990/gaudi2-common): shared kernels, patches and base Dockerfiles.

## License

Apache License 2.0. See [LICENSE](LICENSE). Model weights are not included and keep their own licenses.
