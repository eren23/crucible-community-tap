"""Base collector interface for world model data gathering."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CollectionStats:
    """Statistics from a data collection run."""
    num_transitions: int = 0
    num_sequences: int = 0
    source: str = ""
    output_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseCollector(ABC):
    """Interface for all domain data collectors."""

    @abstractmethod
    def collect(self, source: Path, output: Path, **kwargs: Any) -> CollectionStats:
        """Collect data from source, write HDF5 to output."""
        ...

    @abstractmethod
    def validate(self, output: Path) -> bool:
        """Verify HDF5 integrity and schema."""
        ...
