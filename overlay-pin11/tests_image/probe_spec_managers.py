"""Which real manager class each r4e KV spec gets (MLA 128, KpoolTail 4, KDA Mamba 1048576); prints one JSON line."""
from __future__ import annotations

import json

import torch
from vllm.v1.core import single_type_kv_cache_manager as single
from vllm.v1.kv_cache_interface import KpoolTailSpec, MambaSpec, MLAAttentionSpec

R4E_SPECS = {
    "mla": MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16),
    "tail": KpoolTailSpec(block_size=4, num_kv_heads=1, head_size=256, head_size_v=0, dtype=torch.bfloat16,
                          sliding_window=4),
    "kda": MambaSpec(block_size=1 << 20, shapes=((3, 64), (4, 32, 32)), dtypes=(torch.bfloat16, torch.float32)),
}


def manager_names() -> dict[str, str]:
    single.register_all_kvcache_specs(None)
    return {name: single.KVCacheSpecRegistry.get_manager_class(spec).__name__ for name, spec in R4E_SPECS.items()}


if __name__ == "__main__":
    print(json.dumps(manager_names()))
