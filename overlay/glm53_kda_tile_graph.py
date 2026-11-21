"""Owned, bounded KDA tile graph; default off and independent of model graphs."""
import os
import torch
from glm53_mixed_graph import key_capture_active

_CACHE = {}
_STATS = {'captures': 0, 'replays': 0, 'fallbacks': 0}

def enabled():
    value = os.getenv('HPU_GLM53_KDA_TILE_GRAPH', '0')
    if value not in ('0', '1'):
        raise ValueError('HPU_GLM53_KDA_TILE_GRAPH must be 0 or 1')
    return value == '1'

def tile(original, q, k, v, g, beta, *, chunk_size=64, initial_state=None,
         output_final_state=True, use_qk_l2norm=True):
    kwargs = dict(chunk_size=chunk_size, initial_state=initial_state,
                  output_final_state=output_final_state, use_qk_l2norm=use_qk_l2norm)
    # A key capture is one graph: a nested HPUGraph and its sync=True fences are illegal inside it.
    if not enabled() or q.device.type != 'hpu' or tuple(q.shape[:2]) not in ((1,512),(4,128)) or key_capture_active():
        _STATS['fallbacks'] += 1
        return original(q,k,v,g,beta,**kwargs)
    if initial_state is None or initial_state.dtype != torch.float32:
        raise ValueError('KDA tile graph requires owned FP32 initial state')
    import habana_frameworks.torch.core as htcore
    sources = (q,k,v,g,beta,initial_state)
    key = (id(original), chunk_size, use_qk_l2norm,
           tuple((tuple(t.shape),t.dtype,t.device) for t in sources))
    entry = _CACHE.get(key)
    if entry is None:
        if len(_CACHE) >= 2:
            _STATS['fallbacks'] += 1
            return original(q,k,v,g,beta,**kwargs)
        buffers = tuple(t.contiguous().clone() for t in sources)
        htcore.mark_step(sync=True)
        torch.hpu.synchronize()
        graph = torch.hpu.HPUGraph()
        with torch.hpu.graph(graph, dry_run=False):
            outputs = original(*buffers[:5],chunk_size=chunk_size,
                initial_state=buffers[5], output_final_state=True,
                use_qk_l2norm=use_qk_l2norm)
        htcore.mark_step(sync=True)
        torch.hpu.synchronize()
        entry = (buffers, graph, outputs)
        _CACHE[key] = entry
        _STATS['captures'] += 1
    else:
        buffers,graph,outputs = entry
        for target,source in zip(buffers,sources):
            target.copy_(source)
        htcore.mark_step(sync=True)
        graph.replay()
        _STATS['replays'] += 1
    # Caller owns both results across later layers, tiles and cache replays.
    output,state = entry[2]
    result = output.clone(), state.clone() if output_final_state else None
    return result
