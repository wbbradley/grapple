from typing import Any

import numpy as np
import psycopg
from pydantic import BeforeValidator, PlainSerializer
from typing_extensions import Annotated

# TODO(wbbradley): We are using dictionaries for all cursors at the moment. Clean this up later.
Cursor = psycopg.Cursor[dict[str, Any]]


def nd_array_custom_before_validator(x):
    assert x.shape == (3072,)
    return x


def nd_array_custom_serializer(x):
    assert x.shape == (3072,)
    return str(x)


Vector = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=str),
]
