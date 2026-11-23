"""Child-interpreter probe for test_image_attn_pin: with HPU_GLM53_ATTN_PIN=1 the .pth (or, without it, install_from_env)
must leave the five real modules wrapped at every patch point. Prints one JSON line."""
from __future__ import annotations

import importlib
import json
import os
import sys

sys.path.insert(0, os.environ.get("PINKIT", "/pinkit"))
import glm53_attn_pin as pin  # noqa: E402

pin.install_from_env()  # a no-op when the .pth already ran; the hook is idempotent
wrapped: dict[str, bool] = {}
for module_name, points in pin.TARGET_MODULES.items():
    module = importlib.import_module(module_name)
    for cls_name, method in points:
        cls = getattr(module, cls_name)
        originals = vars(cls).get(pin.ORIGINALS_ATTR, {})
        current = getattr(cls, method)
        wrapped[f"{cls_name}.{method}"] = method in originals and getattr(current, "__wrapped__", None) is originals[method]
print(json.dumps({
    "finder_installed": any(isinstance(f, pin._PatchOnImport) for f in sys.meta_path),
    "module_file": pin.__file__,
    "wrapped": wrapped,
}))
