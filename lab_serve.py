"""Quad A/B TP4 lab launcher for the dual-sustained lane: boots an image with a replica's exact runtime settings.

    lab.py up   --quad A|B --name claude-flash-launch-X --image <ref> --port 18420 [--env K=V]... [--mount SRC:DST[:ro]]... [--dry-run]
    lab.py down --name claude-flash-launch-X [--keep-leaf] [--dry-run]

Generalized from team-20261105/launch/lab.py (quad B only). Quad A = HABANA modules 4-7, copied from the held unit
glm53flash-r0 (uverbs 0-3); quad B = modules 0-3 from glm53flash-r1 (uverbs 4-7). The held production unit is only
*inspected*; the container is a copy of its HostConfig/env/args with a private port, a private reflink copy of its
recipe-cache leaf, and no change to the production units.
The caller must already hold the quad's lock (quad_hold.sh <A|B> ...); a real `up` refuses otherwise.
--dry-run prints the plan as JSON and changes nothing: no lock needed, no leaf created, no docker run/stop/rm, no
request to any port.
"""
import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Callable
from urllib.request import urlopen

RECIPE_ROOT = Path('/mnt/data1/recipe_cache')
LANE = Path(__file__).resolve().parent
KV_BLOCKS = '31264'
ACCEL_NODE = re.compile(r'/dev/accel/accel(?:_controlD)?(\d+)$')
UVERBS_NODE = re.compile(r'/dev/infiniband/uverbs(\d+)$')
UVERBS_NAME = re.compile(r'uverbs(\d+)$')
ACCEL_SYSFS = Path('/sys/class/accel')
VERBS_SYSFS = Path('/sys/class/infiniband_verbs')
STARTUP_DEADLINE_S = 1500
REPLICA_PORTS = frozenset({18274, 18275})
MODEL_LEN_STEP = 128  # --block-size of every unit: a served max-model-len is a whole number of KV blocks
MODEL_LEN_ENV = 'LAB_MAX_MODEL_LEN'  # the staged gates file exports it (pinned by its sha); unset = the source unit's value
DEFAULT_EFFORT_ENV = 'LAB_DEFAULT_REASONING_EFFORT'  # the staged gates file exports it (pinned by its sha); unset = the source unit's level
TEMPLATE_KWARGS_FLAG = '--default-chat-template-kwargs'
TEMPLATE_EFFORTS = ('low', 'high', 'max')  # GLM-5.3 chat_template.jinja line 2: only low and high are read by name, anything else is max
PROMPT_KEYS_ENV = 'VLLM_GAUDI_MIXED_PROMPT_KEYS'  # ROWS:WIDTH,... with WIDTH in blocks of MODEL_LEN_STEP tokens
NAME_PREFIX = 'claude-flash-launch-'
OFFLOAD_ATTENTION_ENV = 'VLLM_CPU_OFFLOAD_ATTENTION_BYTES'
SHM_HEADROOM_BYTES = 4 * 2**30  # on top of the attention store file: torch/Habana shared segments, psm_* and semaphore files


@dataclass(frozen=True)
class Quad:
    name: str
    source_unit: str
    modules: str
    second_unit: str = ''  # T8 only: its devices and CPUs join the source unit's
    locks: tuple[str, ...] = ()  # T8 only: the quads whose locks a boot needs; empty = its own
    kv_blocks: str = KV_BLOCKS
    tensor_parallel: str = '4'


QUADS = {'A': Quad('A', 'glm53flash-r0', '4,5,6,7'), 'B': Quad('B', 'glm53flash-r1', '0,1,2,3'),
         'T8': Quad('T8', 'glm53flash-r0', '0,1,2,3,4,5,6,7', 'glm53flash-r1', ('A', 'B'), '62528', '8')}


def require(condition: bool, message: str) -> None:
    """Precondition check that survives `python -O` (unlike assert) and tells the operator what to fix."""
    if not condition:
        raise SystemExit(message)


def lock_holder_path(quad: Quad) -> Path:
    return Path(f'/tmp/claude-flash-quad{quad.name}.holder')


def docker(*args: str) -> str:
    return subprocess.check_output(['docker', *args], text=True)


def remove_leaf(leaf: Path) -> None:
    """Delete this run's private recipe-leaf copy; report (not raise) files the container left unremovable."""
    if not leaf.name.startswith('claude-flash-') or leaf.parent != RECIPE_ROOT:
        raise ValueError(f'refusing to remove {leaf}: not a claude-flash lab leaf under {RECIPE_ROOT}')
    failures: list[str] = []
    shutil.rmtree(leaf, onexc=lambda _function, path, _error: failures.append(path))
    print('LEAF_REMOVED', leaf, f'unremovable={len(failures)}', flush=True)


def inspect(name: str) -> dict:
    return json.loads(docker('inspect', name))[0]


def healthy(port: int) -> bool:
    try:
        with urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def combined_source(first: dict, second: dict, quad: Quad) -> dict:
    """Pure: the unit `first` widened to `quad`'s cards: both units' devices and CPUs, TP and KV of `quad`; args of `first`."""
    source = copy.deepcopy(first)
    config = source['HostConfig']
    seen = {device['PathOnHost'] for device in config['Devices']}
    config['Devices'] += [device for device in second['HostConfig']['Devices'] if device['PathOnHost'] not in seen]
    config['CpusetCpus'] = f"{second['HostConfig']['CpusetCpus']},{config['CpusetCpus']}"
    config['CpusetMems'] = ','.join(sorted({*config['CpusetMems'].split(','), *second['HostConfig']['CpusetMems'].split(',')}))
    env = dict(value.split('=', 1) for value in source['Config']['Env'])
    env.update(HABANA_VISIBLE_MODULES=quad.modules, HPU_GLM53_TP_SHARD_SIZE=quad.tensor_parallel)
    env.update(HPU_GLM53_TERMINAL_LAYER_MARKSTEP='0')  # the runner refuses it above TP4
    source['Config']['Env'] = [f'{key}={value}' for key, value in env.items()]
    argv = source['Args']
    argv[argv.index('--tensor-parallel-size') + 1] = quad.tensor_parallel
    argv[argv.index('--num-gpu-blocks-override') + 1] = quad.kv_blocks
    source['State'] = {'Running': first['State']['Running'] or second['State']['Running']}
    return source


def source_for(quad: Quad) -> dict:
    source = inspect(quad.source_unit)
    return combined_source(source, inspect(quad.second_unit), quad) if quad.second_unit else source


def require_lock_holder(quad: Quad) -> None:
    for name in quad.locks or (quad.name,):
        holder = lock_holder_path(QUADS[name])
        pid = holder.read_text().strip() if holder.is_file() else ''
        if not pid.isdigit() or not Path(f'/proc/{pid}').is_dir():
            raise SystemExit(f'quad {name} lock is not held by a live quad_hold.sh ({holder}); '
                             f'start `quad_hold.sh {name} <release-file>` first (flock is mandatory)')


def accel_indices(rows: list[str], quad: Quad) -> set[int]:
    """/dev/accel indices of the quad's four HABANA modules (hl-smi query: index == the /dev/accel/accel<N> node)."""
    wanted = {int(module) for module in quad.modules.split(',')}
    pairs = [tuple(int(part) for part in row.split(',')) for row in rows if row.strip()]
    found = {index for index, module in pairs if module in wanted}
    require(len(found) == len(wanted), f'expected {len(wanted)} accel nodes for modules {quad.modules}, got {pairs}')
    return found


def quad_accel_indices(quad: Quad) -> set[int]:
    rows = subprocess.check_output(['hl-smi', '-Q', 'index,module_id', '-f', 'csv,noheader'], text=True).splitlines()
    return accel_indices(rows, quad)


def uverbs_indices(card_addresses: dict[int, str], verbs_addresses: dict[int, str], allowed: set[int]) -> set[int]:
    """Pure: the /dev/infiniband/uverbs<N> indices of the cards behind the `allowed` accel indices, matched by PCI address."""
    wanted = {card_addresses[index] for index in allowed}
    found = {index for index, address in verbs_addresses.items() if address in wanted}
    require(len(found) == len(allowed), f'expected {len(allowed)} uverbs nodes for the cards at {sorted(wanted)}, found '
                                        f'{sorted(found)} in {sorted(verbs_addresses.items())}; is habanalabs_ib loaded (lsmod)?')
    return found


def quad_uverbs_indices(allowed: set[int]) -> set[int]:
    cards = {index: (ACCEL_SYSFS / f'accel{index}/device/pci_addr').read_text().strip() for index in allowed}
    verbs = {int(UVERBS_NAME.match(node.name).group(1)): (node / 'device').resolve().name
             for node in VERBS_SYSFS.glob('uverbs[0-9]*')}
    return uverbs_indices(cards, verbs, allowed)


def with_current_uverbs(source: dict, uverbs: set[int]) -> dict:
    """Pure: a copy of `source` whose /dev/infiniband/uverbs devices are the nodes in `uverbs`.

    The source unit recorded its list at ITS boot. Card and uverbs numbers follow the driver's probe order and change
    at a host restart or driver reload; on 2026-11-13 the copied list named another quad's node and every champion
    boot died in "ibv initialization failed".
    """
    current = copy.deepcopy(source)
    devices = current['HostConfig']['Devices']
    recorded = [device for device in devices if UVERBS_NODE.match(device['PathOnHost'])]
    permissions = recorded[0]['CgroupPermissions'] if recorded else 'rwm'
    fresh = [{'PathOnHost': path, 'PathInContainer': path, 'CgroupPermissions': permissions}
             for path in (f'/dev/infiniband/uverbs{index}' for index in sorted(uverbs))]
    current['HostConfig']['Devices'] = [device for device in devices if device not in recorded] + fresh
    return current


def scoped_devices(devices: list[dict], allowed: set[int]) -> list[dict]:
    """The source unit's devices minus every accel node that is not one of the quad's four cards."""
    kept = []
    for device in devices:
        match = ACCEL_NODE.match(device['PathOnHost'])
        if match is None or int(match.group(1)) in allowed:
            kept.append(device)
    cards = sum(1 for d in kept if '/accel/accel' in d['PathOnHost'] and 'control' not in d['PathOnHost'])
    require(cards == len(allowed) and cards in (4, 8), f'scoped device list has {cards} accel cards, expected 4 or 8; allowed indices were {sorted(allowed)}')
    return kept


def count_recipes(cache: Path) -> int:
    return sum(1 for item in cache.rglob('*') if item.is_file())


def device_spec(device: dict) -> str:
    return f"{device['PathOnHost']}:{device['PathInContainer']}:{device['CgroupPermissions']}"


def new_leaf_path(name: str) -> Path:
    return RECIPE_ROOT / f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def attention_store_bytes(env: dict[str, str], argv: list[str]) -> int | None:
    """Bytes the engine will give its attention store file under /dev/shm, or None when the server has no offload connector.

    The engine takes VLLM_CPU_OFFLOAD_ATTENTION_BYTES when it is set (it beats the JSON) and cpu_bytes_to_use of --kv-transfer-config otherwise.
    A connector without either value is refused: guessing a size would either starve the store file or leave the tmpfs unbounded."""
    if OFFLOAD_ATTENTION_ENV in env:
        require(env[OFFLOAD_ATTENTION_ENV].isdigit() and int(env[OFFLOAD_ATTENTION_ENV]) > 0,
                f'{OFFLOAD_ATTENTION_ENV}={env[OFFLOAD_ATTENTION_ENV]!r} is not a positive whole number of bytes; fix the --env or the source unit')
        return int(env[OFFLOAD_ATTENTION_ENV])
    if '--kv-transfer-config' not in argv:
        return None
    extra = json.loads(argv[argv.index('--kv-transfer-config') + 1]).get('kv_connector_extra_config', {})
    require(str(extra.get('cpu_bytes_to_use', '')).isdigit() and int(extra['cpu_bytes_to_use']) > 0,
            f'the server has --kv-transfer-config but neither {OFFLOAD_ATTENTION_ENV} nor cpu_bytes_to_use is set, so lab.py cannot size the container '
            f'/dev/shm; pass --env {OFFLOAD_ATTENTION_ENV}=<bytes>')
    return int(extra['cpu_bytes_to_use'])


def shm_tmpfs(env: dict[str, str], argv: list[str]) -> list[str]:
    """docker run arguments giving the container its own /dev/shm, so the attention store file dies with the container (a kill, a crash, a stop).
    --ipc host stays: only /dev/shm is replaced. Empty when the server has no offload connector."""
    size = attention_store_bytes(env, argv)
    return [] if size is None else ['--tmpfs', f'/dev/shm:rw,size={size + SHM_HEADROOM_BYTES}']


def set_max_model_len(argv: list[str], value: str, quad: Quad) -> None:
    """Rewrite the copied unit's --max-model-len in place (a positive multiple of the 128-token block)."""
    require(value.isdigit() and int(value) > 0 and int(value) % MODEL_LEN_STEP == 0,
            f'{MODEL_LEN_ENV}={value!r} is not a positive multiple of {MODEL_LEN_STEP}; export whole tokens, e.g. {MODEL_LEN_ENV}=524288, or unset it')
    require('--max-model-len' in argv, f'source unit {quad.source_unit} has no --max-model-len to rewrite; unset {MODEL_LEN_ENV}')
    argv[argv.index('--max-model-len') + 1] = str(int(value))


def set_default_reasoning_effort(argv: list[str], value: str, quad: Quad) -> None:
    """Rewrite reasoning_effort inside the copied unit's --default-chat-template-kwargs in place: the level of a request that names none."""
    require(value in TEMPLATE_EFFORTS,
            f'default reasoning effort {value!r} is not one of low, high, max (the chat template reads every other value as max); '
            f'set {DEFAULT_EFFORT_ENV} or --lab-default-effort to one of those, or leave it out to serve the unit\'s own level')
    require(TEMPLATE_KWARGS_FLAG in argv,
            f'source unit {quad.source_unit} has no {TEMPLATE_KWARGS_FLAG} to rewrite; leave {DEFAULT_EFFORT_ENV} and --lab-default-effort out')
    index = argv.index(TEMPLATE_KWARGS_FLAG) + 1
    argv[index] = json.dumps({**json.loads(argv[index]), 'reasoning_effort': value}, separators=(',', ':'))


def require_keys_span_model_len(env: dict[str, str], argv: list[str]) -> None:
    """A graphs-only key list serves exactly its widest key: refuse a --max-model-len that is not that key's width x 128 tokens."""
    keys = env.get(PROMPT_KEYS_ENV, '')
    if not keys or '--max-model-len' not in argv:
        return
    widths = [item.rpartition(':')[2].strip() for item in keys.split(',') if item.strip()]
    require(bool(widths) and all(width.isdigit() for width in widths),
            f'{PROMPT_KEYS_ENV}={keys!r} is not ROWS:WIDTH,...; fix that levers line')
    widest = max(int(width) for width in widths) * MODEL_LEN_STEP
    served = argv[argv.index('--max-model-len') + 1]
    require(served == str(widest),
            f'--max-model-len {served} beside {PROMPT_KEYS_ENV}={keys}: the widest key holds {widest} tokens. Put '
            f'LAB_MAX_MODEL_LEN={widest} in the gates file (r4e_run.sh unsets an inherited value), or serve keys that span {served}')


def plan_run(source: dict, args: argparse.Namespace, quad: Quad, allowed: set[int],
             leaf: Path) -> tuple[list[str], dict[str, str], list[str]]:
    """Pure: the docker run command, env and server argv for this lab. Creates nothing on disk."""
    config = source['HostConfig']
    env = dict(value.split('=', 1) for value in source['Config']['Env'])
    require(env['HABANA_VISIBLE_MODULES'] == quad.modules, f'source unit {quad.source_unit} is not quad {quad.name}')
    env['PT_HPU_RECIPE_CACHE_CONFIG'] = f'/recipe_cache/{leaf.name},false,8192'
    for item in args.env:
        key, value = item.split('=', 1)
        env[key] = value
    argv = source['Args'][:]
    for key, value in (('--host', os.environ.get('LAB_HOST', '127.0.0.1')), ('--port', str(args.port))):
        argv[argv.index(key) + 1] = value
    require(argv[argv.index('--num-gpu-blocks-override') + 1] == quad.kv_blocks,
            f'source unit {quad.source_unit} does not use --num-gpu-blocks-override {quad.kv_blocks}; the lab must match production KV')
    if args.lab_kv_blocks:
        argv[argv.index('--num-gpu-blocks-override') + 1] = args.lab_kv_blocks  # after the check above: production KV stays the default
    if os.environ.get(MODEL_LEN_ENV, ''):
        set_max_model_len(argv, os.environ[MODEL_LEN_ENV], quad)
    if os.environ.get(DEFAULT_EFFORT_ENV, ''):
        set_default_reasoning_effort(argv, os.environ[DEFAULT_EFFORT_ENV], quad)
    require_keys_span_model_len(env, argv)
    run = ['docker', 'run', '-d', '--name', args.name, '--network', 'host', '--ipc', 'host',
           '--cpuset-cpus', config['CpusetCpus'], '--cpuset-mems', config['CpusetMems'],
           '--ulimit', 'memlock=-1:-1', '--security-opt', 'label=disable', '--entrypoint', source['Path'], *shm_tmpfs(env, argv)]
    for mount in source['Mounts']:
        require(mount['Type'] == 'bind', f"mount {mount['Destination']} is not a bind mount; the lab only copies bind mounts")
        run.extend(['-v', f"{mount['Source']}:{mount['Destination']}:{'rw' if mount['RW'] else 'ro'}"])
    for extra in args.mount:
        run.extend(['-v', extra])
    for cap in config['CapAdd']:
        run.extend(['--cap-add', cap])
    for device in scoped_devices(config['Devices'], allowed):
        run.extend(['--device', device_spec(device)])
    return run, env, argv


def seed_recipe_leaf(source: dict, leaf: Path, fresh_cache: bool) -> None:
    """Create the private leaf and (unless fresh) reflink-copy the source unit's recipe cache into it."""
    leaf.mkdir()
    if fresh_cache:
        return
    env = dict(value.split('=', 1) for value in source['Config']['Env'])
    origin = RECIPE_ROOT / Path(env['PT_HPU_RECIPE_CACHE_CONFIG'].split(',')[0]).name
    try:
        subprocess.run(['cp', '-a', '--reflink=auto', f'{origin}/.', str(leaf)], check=True)
    except subprocess.CalledProcessError:
        remove_leaf(leaf)
        raise


def validate_up(args: argparse.Namespace) -> None:
    if not args.name.startswith(NAME_PREFIX):
        raise SystemExit(f'container names must be {NAME_PREFIX}*; got {args.name!r}')
    if args.port in REPLICA_PORTS and not serve_port_allowed():
        raise SystemExit(f'port {args.port} is a production replica port ({sorted(REPLICA_PORTS)}); pick a lab port such as 18420 '
                         '(the r4f serve install sets LAB_SERVE_PROD=1, and only with glm53flash-production.target inactive)')


def serve_port_allowed() -> bool:
    """The r4f serve install (coordinator 08:2x): production is down, so its replicas take the production ports."""
    if os.environ.get('LAB_SERVE_PROD') != '1':
        return False
    state = subprocess.run(['systemctl', 'is-active', 'glm53flash-production.target'], capture_output=True, text=True)
    return state.stdout.strip() != 'active'


def dry_run_report(source: dict, args: argparse.Namespace, quad: Quad, allowed: set[int], leaf: Path) -> dict:
    run, env, argv = plan_run(source, args, quad, allowed, leaf)
    kept = {device_spec(d) for d in scoped_devices(source['HostConfig']['Devices'], allowed)}
    specs = [device_spec(d) for d in source['HostConfig']['Devices']]
    return {'dry_run': True, 'quad': quad.name, 'source_unit': quad.source_unit, 'modules': quad.modules,
            'accel_indices': sorted(allowed), 'source_running': source['State']['Running'],
            'cpuset_cpus': source['HostConfig']['CpusetCpus'], 'cpuset_mems': source['HostConfig']['CpusetMems'],
            'devices': [s for s in specs if s in kept], 'devices_dropped': [s for s in specs if s not in kept],
            'recipe_leaf': str(leaf), 'image': args.image, 'port': args.port, 'env_overrides': args.env, 'mounts': args.mount,
            'habana_visible_modules': env['HABANA_VISIBLE_MODULES'], 'docker_run': run, 'server_argv': argv,
            'note': 'dry-run: no lock needed, nothing created or started; a real `up` also needs the source unit stopped'}


def state_path(name: str) -> Path:
    return LANE / f'state-{name}.json'


def write_state(name: str, state: dict) -> None:
    state_path(name).write_text(json.dumps(state, indent=2))


CONTIGUOUS_ORDER = 9  # gaudi2_ctx_init: page allocation failure order:9, then Failed to create context -12
CONTIGUOUS_BLOCKS_PER_NODE = 16
COMPACT_COMMAND = 'echo 1 > /proc/sys/vm/compact_memory'
DROP_COMMAND = 'sync; echo 1 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'


def contiguous_blocks(buddyinfo: str) -> dict[int, int]:
    """Free memory in units of order-9 blocks, per node, from /proc/buddyinfo's Normal zones: the columns are orders 0 up."""
    blocks: dict[int, int] = {}
    for line in buddyinfo.splitlines():
        fields = line.replace(',', ' ').split()
        if len(fields) > 4 and fields[0] == 'Node' and fields[3] == 'Normal':
            counts = [int(field) for field in fields[4:]]
            blocks[int(fields[1])] = sum(count << (order - CONTIGUOUS_ORDER) for order, count in enumerate(counts) if order >= CONTIGUOUS_ORDER)
    return blocks


def short_nodes(buddyinfo: str) -> dict[int, int]:
    return {node: count for node, count in contiguous_blocks(buddyinfo).items() if count < CONTIGUOUS_BLOCKS_PER_NODE}


def run_as_root(command: str) -> None:
    subprocess.run(['sudo', '-n', 'sh', '-c', command], check=True)


def prepare_host_memory(run: Callable[[str], None] = run_as_root, read: Callable[[], str] = lambda: Path('/proc/buddyinfo').read_text()) -> None:
    """Every boot needs order-9 host pages for its device contexts, or its workers die with synStatus=8. Compact; drop the page cache only if that is not enough."""
    run(COMPACT_COMMAND)
    short = short_nodes(read())
    if not short:
        return
    print(f'HOST_MEMORY short of order-{CONTIGUOUS_ORDER} blocks after compaction {short}; dropping the page cache', flush=True)
    run(DROP_COMMAND)
    short = short_nodes(read())
    if short:
        detail = ', '.join(f'node {node} has {count} of {CONTIGUOUS_BLOCKS_PER_NODE}' for node, count in sorted(short.items()))
        raise SystemExit(f'host memory too fragmented to boot: {detail} free order-{CONTIGUOUS_ORDER} blocks, so the driver would fail with Failed to create context -12. '
                         'Nothing was started; stop what holds host memory on that node, or wait for it to end, and retry.')


def up(args: argparse.Namespace) -> None:
    validate_up(args)
    quad = QUADS[args.quad]
    allowed = quad_accel_indices(quad)
    source = with_current_uverbs(source_for(quad), quad_uverbs_indices(allowed))
    leaf = new_leaf_path(args.name)
    if args.dry_run:
        print(json.dumps(dry_run_report(source, args, quad, allowed, leaf), indent=2))
        return
    require_lock_holder(quad)
    require(not source['State']['Running'], f'{quad.source_unit} is running; quad {quad.name} is not free (stop the target first)')
    require(not healthy(args.port), f'port {args.port} already serving; pick another lab port')
    prepare_host_memory()
    seed_recipe_leaf(source, leaf, args.fresh_cache)
    state = {'name': args.name, 'quad': quad.name, 'image': args.image, 'port': args.port, 'recipe_leaf': str(leaf),
             'recipes_before': count_recipes(leaf), 'started_utc': datetime.now(timezone.utc).isoformat(),
             'env_overrides': args.env, 'mounts': args.mount}
    write_state(args.name, state)  # before the boot: a boot that fails still leaves `down` the leaf to remove
    run, env, argv = plan_run(source, args, quad, allowed, leaf)
    start = time.monotonic()
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as env_file:
        env_file.write(''.join(f'{key}={value}\n' for key, value in env.items()))
        env_file.flush()
        docker(*run[1:], '--env-file', env_file.name, args.image, *argv)
    wait_until_ready(args, start)
    state['startup_ready_s'] = round(time.monotonic() - start, 1)
    state['recipes_after_boot'] = count_recipes(leaf)
    write_state(args.name, state)
    print('LAB_READY', json.dumps(state), flush=True)


def wait_until_ready(args: argparse.Namespace, start: float) -> None:
    while not healthy(args.port):
        require(inspect(args.name)['State']['Running'], f'container {args.name} exited before ready; see docker logs {args.name}')
        require(time.monotonic() - start < args.deadline, f'startup deadline {args.deadline}s exceeded; docker logs {args.name}')
        time.sleep(10)


def down(args: argparse.Namespace) -> None:
    path = state_path(args.name)
    if args.dry_run:
        leaf = json.loads(path.read_text())['recipe_leaf'] if path.is_file() else None
        print(json.dumps({'dry_run': True, 'commands': [f'docker stop -t 60 {args.name}', f'docker logs --timestamps {args.name}',
                                                         f'docker rm {args.name}'],
                          'remove_leaf': None if args.keep_leaf else leaf}, indent=2))
        return
    subprocess.run(['docker', 'stop', '-t', '60', args.name], check=False)
    if path.is_file():
        state = json.loads(path.read_text())
        state['recipes_at_stop'] = count_recipes(Path(state['recipe_leaf']))
        path.write_text(json.dumps(state, indent=2))
    logs = subprocess.run(['docker', 'logs', '--timestamps', args.name], capture_output=True, text=True)
    (LANE / f'{args.name}.log').write_text(logs.stdout + logs.stderr)
    subprocess.run(['docker', 'rm', args.name], check=False)
    if path.is_file() and not args.keep_leaf:
        remove_leaf(Path(json.loads(path.read_text())['recipe_leaf']))
    print('LAB_DOWN', args.name, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('up', 'down'):
        item = sub.add_parser(command)
        item.add_argument('--name', required=True)
        item.add_argument('--dry-run', action='store_true', help='print the plan as JSON; change nothing')
        if command == 'up':
            item.add_argument('--quad', required=True, choices=sorted(QUADS), help='A = modules 4-7 (from r0), B = modules 0-3 (from r1)')
            item.add_argument('--image', required=True)
            item.add_argument('--port', type=int, required=True)
            item.add_argument('--env', action='append', default=[])
            item.add_argument('--mount', action='append', default=[])
            item.add_argument('--fresh-cache', action='store_true', help='empty recipe leaf (changed graphs)')
            item.add_argument('--deadline', type=int, default=STARTUP_DEADLINE_S)
            item.add_argument('--lab-kv-blocks', default='', help='measurement only: boot at this KV instead of the unit\'s')
        else:
            item.add_argument('--keep-leaf', action='store_true', help='keep the private recipe-leaf copy')
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    {'up': up, 'down': down}[args.command](args)


if __name__ == '__main__':
    main()
