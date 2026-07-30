#!/usr/bin/env python3
"""Typed PBT manifest schemas."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from training.pbt.models.config import MEMBER_NAME_RE


class ManifestSectionModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ManifestMemberState(ManifestSectionModel):
    name: str
    lr: float = Field(gt=0.0)
    parent: str | None = None

    @field_validator("name")
    @classmethod
    def validate_member_name(cls, value):
        if not MEMBER_NAME_RE.fullmatch(value):
            raise ValueError("Manifest member names must be filesystem-safe")
        return value


class PBTManifest(ManifestSectionModel):
    schema_version: Literal[1]
    experiment: str
    fingerprint: str
    status: Literal["running", "completed", "interrupted", "failed"]
    next_generation: int = Field(ge=0)
    config: dict[str, Any]
    members: dict[str, ManifestMemberState]
    generations: list[dict[str, Any]]
    best: dict[str, Any] | None = None

    @classmethod
    def parse_payload(cls, payload: Any):
        try:
            return cls.model_validate(payload)
        except ValidationError as error:
            raise ValueError(str(error)) from error

    @model_validator(mode="after")
    def validate_member_keys(self):
        for key, member in self.members.items():
            if key != member.name:
                raise ValueError("Manifest member key must match member.name")
        return self

    def to_runtime_dict(self):
        return self.model_dump(mode="json")
