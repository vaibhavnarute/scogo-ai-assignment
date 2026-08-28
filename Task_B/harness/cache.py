"""Small in-process cache for immutable observations within one run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable


@dataclass(slots=True)
class CacheEntry:
    value: dict[str, Any]
    fingerprint: Hashable
    dependencies: frozenset[Path]


class ObservationCache:
    def __init__(self) -> None:
        self._entries: dict[Hashable, CacheEntry] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable, fingerprint: Hashable) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None or entry.fingerprint != fingerprint:
            if entry is not None:
                self._entries.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        return dict(entry.value)

    def put(
        self,
        key: Hashable,
        value: dict[str, Any],
        fingerprint: Hashable,
        dependencies: set[Path],
    ) -> None:
        self._entries[key] = CacheEntry(
            dict(value), fingerprint, frozenset(path.resolve() for path in dependencies)
        )

    def invalidate_path(self, path: Path) -> int:
        target = path.resolve()
        keys = [
            key
            for key, entry in self._entries.items()
            if any(
                dependency == target
                or dependency in target.parents
                or target in dependency.parents
                for dependency in entry.dependencies
            )
        ]
        for key in keys:
            self._entries.pop(key, None)
        return len(keys)

    def __len__(self) -> int:
        return len(self._entries)

