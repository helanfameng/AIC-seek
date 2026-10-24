"""Bind read/review/extraction helpers to isolated experiment-seven paths.

Only Python module globals in this process are rebound. No legacy source, model,
cache, labels, or manifest is modified. New code imports common7, never common.
"""
import importlib
import sys

import common7 as c

_bound = False


def bind():
    global _bound
    if not _bound:
        sys.path.append(str(c.LEGACY / "scripts"))
        old = importlib.import_module("common")
        old.DATA, old.EXPERIMENT, old.ROOT = c.DATA, c.EXPERIMENT, c.ROOT
        old.protocol, old.load_manifest = c.protocol, c.manifest
        old.configure_cache = c.configure_cache
        _bound = True
    return {name: importlib.import_module(name) for name in
            ("common", "features", "review_assist", "review", "learning", "evaluation")}
