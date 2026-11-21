# gaudi2-glm-5.3-flash

Build and serving recipes for GLM-5.3-Flash (FP8, abliterated variant) on 8x Intel Gaudi 2, run as two TP4 vLLM replicas.

## Status

- Current production champion (cut `r4h8`), serving since 2026-11-06.
- Tuning stopped on 2026-11-12. This repo is a record of what runs, not an active tuning branch.
- The champion is pinned by three files in `champion-inputs/` plus the image `claude-flash-dual-graphs-c1d:20261112-r4h8`.
- Model weights are not in this repo. Download them separately.
- This is a snapshot of scripts as they ran on one lab host. They contain host paths you must change (see [Paths to change](#paths-to-change)) and rely on a few things that are not in the repo (see [Known issues](#known-issues)).

## Results

Topology of the champion (cut r4h8): 2 replicas, each on one 4-card group (TP4), 16 concurrent sequences per replica (32 total), 524,288-token context.

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

The goal for this model was at least 10 tok/s per agent with 16 agents on one 4-card replica; r4h8 clears it.

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

Run on the Gaudi host, from this folder. Boot takes about 15 minutes per replica.

### 0. Prepare the host

1. Set the paths listed in [Paths to change](#paths-to-change).
2. In `promote-r4h/serve_champion.sh`, set `L` (line 16) to the folder that holds this repo's `lab_serve.py` and `diag-r4e/` (so `$L/lab_serve.py` and `$L/diag-r4e/r4e_run.sh` exist).
3. `lab_serve.py` copies its settings from two existing stopped containers named `glm53flash-r0` and `glm53flash-r1` (the "source units"). They carry the vLLM arguments and offload sizing and are not in this repo. See [Known issues](#known-issues).

### 1. Build the image

```bash
./build_image.sh                 # default tag claude-flash-dual-sustain:20261106
./build_image.sh my-tag:latest   # or choose a tag
```

It checks free disk (100 GB on the Docker root and on the data disk), refuses to touch protected images, builds from `overlay/` with `install_overlay.sh`, runs the differential tests, checks that new flags default to off (`check_flags_off.py`), and prints `BUILD_RESULT: PASS` or `BUILD_RESULT: FAIL`. Log: `build_image.log`. It starts no card.

### 2. Stage the champion

Staging pins the image, the three env files and their sha256 into `champion.staged`. `--image-sha` is the image id from `docker image inspect -f '{{.Id}}' <image>`.

```bash
cd promote-r4h
./promote_r4h.sh stage \
    --image claude-flash-dual-graphs-c1d:20261112-r4h8 \
    --image-sha <sha256:...> \
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

Other modes: `./champion_boot.sh r0 --stop` (stay down), `--recover` (what the timer runs), `./serve_champion.sh r0 --check` (pre-boot checks only, ends in `CHAMPION_CHECK_OK` or a `HALT` line), `./rollback_r4f.sh [--dry-run]`.

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
| `install_overlay.sh` | Runs in `docker build`: verifies base file hashes, then installs the overlay into site-packages. |
| `check_flags_off.py` | Checks that new overlay flags default to off in the built image. |
| `overlay/` | Patched vLLM, vllm-gaudi and `glm53_*` modules; `BASE_SHA256.txt` lists the base hashes. `tests/` and `cells/` stay out of the image. |
| `champion-inputs/levers.env` | Serving levers for cut r4h8 (applied after final-config, later wins). |
| `champion-inputs/gates.override.env` | Gate settings, served context length and default reasoning level. |
| `champion-inputs/final-config.env` | Base serving config, including the KV offload load-path settings. |
| `lab_serve.py` | Launcher: boots a TP4 container on quad A or B from a copy of the source unit's settings. |
| `diag-r4e/` | `r4e_run.sh` (prepare and boot), `r4e_gates.sh`, probes and tests used by the launcher. |
| `promote-r4h/promote_r4h.sh` | Stages a champion and promotes it to both replicas, with gates and automatic rollback. |
| `promote-r4h/rollback_r4f.sh` | Returns both replicas to the previous (r4f) serving install. |
| `promote-r4h/serve_champion.sh` | Boots one replica from the staged files; `--check` runs pre-boot checks only. |
| `promote-r4h/champion_boot.sh` | Keeps a replica up: start, `--plan`, `--stop`, `--recover`. |
| `promote-r4h/common.sh`, `champion_check.sh` | Shared constants and helpers (ports, quads, budgets); image-hash check that starts no container. |
| `promote-r4h/test_*.py`, `SHA256SUMS*` | Tests and checksums for the kit. |

## Settings that matter and why

Values are the live champion replicas' settings (cut r4h8, read with `docker inspect`), plus the files in `champion-inputs/`.

| Setting | Value | Why |
|---|---|---|
| Tensor parallelism | `--tensor-parallel-size 4` per replica | One 4-card quad each; 2 replicas use all 8 cards. |
| Context | `--max-model-len 524288` (`LAB_MAX_MODEL_LEN`) | The image refuses a larger value beside these prompt keys. |
| Concurrency | `--max-num-seqs 16` per replica (32 total) | Matches the top decode key of 16 rows. |
| Batched tokens | `--max-num-batched-tokens 8192` | Per-step token budget. |
| HBM use | `--gpu-memory-utilization 0.90` | Leaves headroom on each card. |
| Prefix caching | `--no-enable-prefix-caching` | Must stay off: it made this model incoherent. |
| Default reasoning | `LAB_DEFAULT_REASONING_EFFORT=high`, `VLLM_REASONING_AUTO_BUDGET=1`, `VLLM_REASONING_AUTO_MAX_TOKENS=4096`, `VLLM_REASONING_MAX_ANSWER_TOKENS=8192` | Applies to requests that name no level. The chat template reads only `low` and `high` by name; any other value is max. Disabling reasoning returns 0 reasoning tokens, and mixed reasoning broke attention. |
| KV blocks per replica | 31264 (`--num-gpu-blocks-override`), block size 128 | `lab_serve.py` refuses a source unit that differs. |
| Host offload size | `VLLM_CPU_OFFLOAD_TOTAL_BYTES=65000000000`, `CPU_POOL_BYTES=135000000000` per replica | KV and recurrent state offload to host RAM over PCIe. |
| Offload loading | `HPU_KV_OFFLOAD_ASYNC_LOAD=1`, `HPU_KV_OFFLOAD_STAGE_MB=128`, `HPU_KV_OFFLOAD_MAX_INFLIGHT=2`, `HPU_KV_OFFLOAD_PIN_STAGE=1`, `GLM53_LOADPATH_V2=1`, `GLM53_LOADPATH_THREADS=4`, `GLM53_LOADPATH_DEPTH=8`, `HPU_GLM53_OFFLOAD_ROW_BUCKETS=1` | Pinned staging moves offloaded KV back with bounded in-flight copies. |
| Recurrent state | `VLLM_RECURRENT_SIDECAR_PROMPT_END_ONLY=1`, `VLLM_RECURRENT_TAIL_FUSE=1`, `VLLM_RECURRENT_TAIL_FUSE_KEYED=1` | Levers file overrides the final-config tail-fuse value of 0. |
| Decode priority | `VLLM_DECODE_PRIORITY=phases`, mixed on, min decode steps 8, max prefill wait 8 s | Prevents long prefills from stalling decode. |
| Mixed prefill | `VLLM_GAUDI_MIXED_PREFILL_TOKENS=1536` | Prefill chunk size. |
| HBM margin | `HPU_GLM53_HBM_MARGIN_MIB=512` | Headroom on each card. |
| Restore fence | `GLM53_RESTORE_FENCE_MODE=sync` | Async design stays off. |
| Boot / warmup budget | 1500 s / 300 s | Gates in `common.sh`. |
| Auto-restart | 3 tries per 10800 s | `champion_boot.sh --recover`; a replica stopped by hand stays down. |

## Paths to change

These are hard-coded for the original lab host. Change each to your own path.

| Path | Where | Used for |
|---|---|---|
| `/mnt/data1` | `build_image.sh`, `promote-r4h/common.sh`, `champion_boot.sh`, `lab_serve.py`, `diag-r4e/r4e_run.sh` | Data disk, run/queue folders, recipe cache. `QUEUE_DIR` overrides the queue folder in `champion_boot.sh`. |
| `/var/lib/docker` | `build_image.sh` | Docker root for the disk check. |
| `$L` (a folder under `$HOME` on the original host) | `promote-r4h/serve_champion.sh` line 16 | Folder holding `lab_serve.py` and `diag-r4e/`. |
| `/opt/glm53/buckets/*.buckets` | `champion-inputs/levers.env` | Bucket file inside the image. |

## Known issues

- **Source units are not in the repo.** `lab_serve.py` copies the arguments, environment and devices of containers `glm53flash-r0` and `glm53flash-r1`. The `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`, prefix-caching flag, offload sizes and `CPU_POOL_BYTES` listed above come from them and are not in any file here, so this repo alone does not reproduce the champion.
- **Stale comment.** The `serve_champion.sh` header says "1M context"; the champion serves 524,288.
- **Build.** The Dockerfile starts `FROM` a local image id (`sha256:1481bb99...`), not directly from the vault.habana.ai image. It expects a `.dockerignore` (keeping `overlay/tests`, `overlay/cells` and `run_tests.sh` out) that is not in this repo, and `install_overlay.sh` aborts without it. `build_image.sh` produces `claude-flash-dual-sustain`; how `claude-flash-dual-graphs-c1d:20261112-r4h8` was built is not in this repo.
- **`common.sh` typo.** `RC56_IMAGE` is 65 hex characters, one more than a sha256.
- **Network exposure.** Replicas bind `0.0.0.0` on ports 18274 and 18275. Firewall them or put them behind an authenticated proxy.
- **Weights.** No download script is provided.

## Related repos

- [gaudi2-common](https://github.com/LiquidLava1990/gaudi2-common): shared kernels, patches and base Dockerfiles.

## License

Apache License 2.0. See [LICENSE](LICENSE). Model weights are not included and keep their own licenses.
