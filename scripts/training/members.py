"""Internal member identity, independent of mutable learning rate.

The schema-v1 wire format remains name/lr. Names are opaque identifiers,
including historical labels such as lr_3e-6; never parse an LR from a name.
"""
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class MemberState:
    """Snapshot used at command boundaries; LR updates stay in schema-v1 state.

    Do not replace an entire legacy record with this projection: lineage and
    extension fields belong to the original manifest and must remain intact.
    """
    member_id: str
    current_lr: float

    @classmethod
    def from_legacy(cls, member: Mapping):
        return cls(member_id=member["name"], current_lr=member["lr"])

    def to_legacy(self):
        """Project identity/LR for legacy command APIs, not manifest replacement."""
        return {"name": self.member_id, "lr": self.current_lr}
