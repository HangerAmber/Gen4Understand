"""Typed event schema matching Eq. (1), with source-level provenance."""

import math
from dataclasses import asdict, dataclass
from typing import Literal

LEVEL_WEIGHTS = {"atom": 1.0, "act": 0.5, "sub": 0.3, "task": 0.2}
SLOTS = ("actor", "action", "object", "post_state", "temporal_relation")


@dataclass(frozen=True)
class Event:
    event_id: str
    recording_id: str
    split: Literal["train", "val", "test"]
    actor: str
    action: str
    object: str
    pre_state: tuple[str, ...]
    motion: str
    post_state: tuple[str, ...]
    temporal_relation: str
    start_s: float
    end_s: float
    source_video: str
    scene_id: str
    object_instance_id: str
    scene_objects: tuple[str, ...] = ()
    unsupported_actions: tuple[str, ...] = ()
    unsupported_intents: tuple[str, ...] = ()

    def __post_init__(self):
        for name in (
            "event_id",
            "recording_id",
            "actor",
            "action",
            "object",
            "motion",
            "source_video",
            "scene_id",
            "object_instance_id",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if (
            not all(math.isfinite(t) for t in (self.start_s, self.end_s))
            or not 0 <= self.start_s < self.end_s
        ):
            raise ValueError("Expected finite 0 <= start_s < end_s")
        if not self.pre_state or not self.post_state:
            raise ValueError("Pre- and post-states are required")
        if self.temporal_relation.split(":", 1)[0] not in {"before", "after", "while"}:
            raise ValueError("temporal_relation must start with before, after, or while")
        if self.action in self.unsupported_actions:
            raise ValueError("An observed action cannot be a hard negative")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in (
            "pre_state",
            "post_state",
            "scene_objects",
            "unsupported_actions",
            "unsupported_intents",
        ):
            if key in value:
                if not isinstance(value[key], list | tuple) or not all(
                    isinstance(item, str) and item.strip() for item in value[key]
                ):
                    raise ValueError(f"{key} must be an array of nonempty strings")
                value[key] = tuple(value[key])
        return cls(**value)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Unit:
    unit_id: str
    level: Literal["act", "sub", "task"]
    event_ids: tuple[str, ...]

    def __post_init__(self):
        if self.level not in {"act", "sub", "task"} or not self.event_ids:
            raise ValueError("A hierarchy unit requires a supported level and children")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("Duplicate children in hierarchy unit")
