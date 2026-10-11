"""Plain-JSON views of models that may carry BSON values.

Explain output and query results can contain ObjectIds, dates and other BSON types, which
pydantic's JSON mode refuses. `jsonable` renders them as MongoDB Extended JSON instead.
"""

from __future__ import annotations

import json
from typing import Any

from bson import json_util
from pydantic import BaseModel


def jsonable(obj: Any, **dump_kwargs: Any) -> Any:
    """A JSON-serialisable copy of a pydantic model or plain structure."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="python", **dump_kwargs)
    return json.loads(json_util.dumps(obj))
