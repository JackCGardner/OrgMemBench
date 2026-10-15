"""Adapter registry: name -> adapter class.

Real adapters are imported lazily/defensively so the harness still loads (and
smoke-tests) when a given system's optional deps aren't installed. The four
contestants — mem0, zep, gbrain, graphify-oss — register themselves here as they land.
"""

from __future__ import annotations

import logging

from .base import MemoryAdapter
from .reference import ReferenceAdapter

logger = logging.getLogger("orgmembench.registry")

# The public field of contestants (reference is a dev fixture, excluded).
# OSS/self-hosted and hosted variants are SEPARATE, precisely-labeled entries:
#   mem0           = mem0ai OSS (self-hosted)      mem0-platform = Mem0 hosted API
#   zep            = graphiti-core (Zep OSS engine) zep-cloud     = Zep Cloud hosted
#   gbrain (OSS)   graphify-oss (OSS)
CONTESTANTS = ("mem0", "mem0-platform", "graphiti", "zep-cloud", "gbrain", "graphify-oss")

_REGISTRY: dict[str, type[MemoryAdapter]] = {"reference": ReferenceAdapter}


def _try_register(name: str, module: str, cls: str) -> None:
    try:
        mod = __import__(f"orgmembench.adapters.{module}", fromlist=[cls])
        _REGISTRY[name] = getattr(mod, cls)
    except Exception as exc:  # missing deps / not-yet-built — non-fatal
        logger.debug("adapter %r unavailable: %s", name, exc)


# Defensive registration (no-op until each module exists).
_try_register("mempp", "mempp_adapter", "MemppAdapter")
_try_register("mem0", "mem0_adapter", "Mem0Adapter")
_try_register("mem0-platform", "mem0_platform_adapter", "Mem0PlatformAdapter")
_try_register("graphiti", "graphiti_adapter", "GraphitiAdapter")
_try_register("zep-cloud", "zep_cloud_adapter", "ZepCloudAdapter")
_try_register("gbrain", "gbrain_adapter", "GBrainAdapter")
_try_register("graphify-oss", "graphify_oss_adapter", "GraphifyOssAdapter")


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_adapter(name: str, config: dict | None = None, dry_run: bool | None = None) -> MemoryAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; available: {available()}")
    return _REGISTRY[name](config=config, dry_run=dry_run)
