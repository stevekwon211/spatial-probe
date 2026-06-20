# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Query specification + schema validation for occupancy-retrieval experiments.

A query file (e.g. `experiments/occquery_v0/queries.yaml`) is untrusted, schema-validated input:
every query is checked at load time so an invalid one fails loudly HERE, not by accident in the
middle of an experiment. Each query declares:

- `backend`  -- occupancy (the OccQuery core) vs tracking (the box-only baseline).
- `status`   -- implemented vs baseline_only (a capability flag, NOT a syntax error: an
                unsupported query is valid YAML with status=baseline_only, never a SyntaxError).
- `scope`    -- any / all (a `predicate` over each frame) or transition (a `before`->`after`
                temporal pattern within `within_frames`).

Predicate expressions are evaluated by `probe.query_dsl.safe_eval` (an AST whitelist), never
Python `eval()`.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import yaml

__all__ = ["Query", "QuerySpecError", "load_queries", "validate_query"]

_BACKENDS = {"occupancy", "tracking"}
_STATUSES = {"implemented", "baseline_only"}
_SCOPES = {"any", "all", "transition"}
_KNOWN = {
    "id", "nl", "backend", "status", "scope", "refav_expressible", "rationale",
    "predicate", "before", "after", "within_frames",
}


class QuerySpecError(ValueError):
    """Raised when a query file violates the schema."""


@dataclass(frozen=True)
class Query:
    id: str
    nl: str
    backend: str
    status: str
    scope: str
    refav_expressible: bool
    rationale: str
    predicate: str | None = None
    before: str | None = None
    after: str | None = None
    within_frames: int | None = None

    @property
    def is_occupancy(self) -> bool:
        """True iff this query runs on the implemented occupancy core (not a baseline-only or
        non-occupancy query)."""
        return self.backend == "occupancy" and self.status == "implemented"


def validate_query(raw: dict) -> Query:
    """Validate one raw query dict against the schema, returning a Query or raising QuerySpecError."""
    qid = raw.get("id", "<no id>")
    required = ["id", "nl", "backend", "status", "scope", "refav_expressible", "rationale"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise QuerySpecError(f"query {qid}: missing fields {missing}")
    unknown = set(raw) - _KNOWN
    if unknown:
        raise QuerySpecError(f"query {qid}: unknown fields {sorted(unknown)}")
    if raw["backend"] not in _BACKENDS:
        raise QuerySpecError(f"query {qid}: backend must be one of {sorted(_BACKENDS)}")
    if raw["status"] not in _STATUSES:
        raise QuerySpecError(f"query {qid}: status must be one of {sorted(_STATUSES)}")
    if raw["scope"] not in _SCOPES:
        raise QuerySpecError(f"query {qid}: scope must be one of {sorted(_SCOPES)}")
    if not isinstance(raw["refav_expressible"], bool):
        raise QuerySpecError(f"query {qid}: refav_expressible must be a bool")

    if raw["scope"] == "transition":
        for k in ("before", "after", "within_frames"):
            if k not in raw:
                raise QuerySpecError(f"query {qid}: transition scope requires '{k}'")
        if "predicate" in raw:
            raise QuerySpecError(f"query {qid}: transition scope must not set 'predicate'")
        if not isinstance(raw["within_frames"], int) or raw["within_frames"] < 1:
            raise QuerySpecError(f"query {qid}: within_frames must be a positive int")
    else:
        if "predicate" not in raw:
            raise QuerySpecError(f"query {qid}: {raw['scope']} scope requires 'predicate'")
        if "before" in raw or "after" in raw or "within_frames" in raw:
            raise QuerySpecError(f"query {qid}: {raw['scope']} scope must not set transition fields")

    return Query(
        id=raw["id"],
        nl=raw["nl"],
        backend=raw["backend"],
        status=raw["status"],
        scope=raw["scope"],
        refav_expressible=raw["refav_expressible"],
        rationale=raw["rationale"],
        predicate=raw.get("predicate"),
        before=raw.get("before"),
        after=raw.get("after"),
        within_frames=raw.get("within_frames"),
    )


def load_queries(path: str | pathlib.Path) -> list[Query]:
    """Load and fully validate a query YAML file. Raises QuerySpecError on any violation."""
    doc = yaml.safe_load(pathlib.Path(path).read_text())
    if not isinstance(doc, dict) or "queries" not in doc or not isinstance(doc["queries"], list):
        raise QuerySpecError("query file must have a top-level 'queries' list")
    queries = [validate_query(q) for q in doc["queries"]]
    ids = [q.id for q in queries]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise QuerySpecError(f"duplicate query ids: {dupes}")
    return queries
