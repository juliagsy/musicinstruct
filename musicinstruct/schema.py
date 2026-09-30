"""Typed records for MIDI-Instruct items and model predictions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

GoldMode = Literal["unique", "constraint"]
InstructionType = Literal["specific", "descriptive", "stylistic"]
Split = Literal["train", "validation", "test"]
AnnotationStatus = Literal["draft", "reviewed", "adjudicated"]


class EditMask(BaseModel):
    """Region the instruction is allowed to change. Complement is the preserve set."""

    tracks: list[int] | None = None
    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)
    include_drums: bool = False

    @model_validator(mode="after")
    def _valid_time_range(self) -> EditMask:
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError("end_time must be >= start_time")
        return self


class Predicate(BaseModel):
    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """Gold structured edit for the MIDI-Reason track."""

    op: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class BenchmarkItem(BaseModel):
    item_id: str = Field(min_length=1)
    composition_id: str = Field(min_length=1)
    gold_mode: GoldMode
    op_family: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    instruction_type: InstructionType = "specific"
    midi_in: str
    gold_midi: str | None = None
    plan: Plan | None = None
    must_change: list[Predicate] = Field(default_factory=list)
    must_preserve: list[Predicate] = Field(default_factory=list)
    edit_mask: EditMask = Field(default_factory=EditMask)
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    source: str = "synthetic"
    license: str = "CC0-1.0"
    split: Split = "test"
    annotation_status: AnnotationStatus = "draft"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("must_change", "must_preserve")
    @classmethod
    def _nonempty_names(cls, predicates: list[Predicate]) -> list[Predicate]:
        if any(not p.name.strip() for p in predicates):
            raise ValueError("predicate names must be non-empty")
        return predicates


class Prediction(BaseModel):
    item_id: str
    midi_path: str | None = None
    plan: Plan | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
