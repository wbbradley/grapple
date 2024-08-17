from typing import Any

import numpy as np
import psycopg

# TODO(wbbradley): We are using dictionaries for all cursors at the moment. Clean this up later.
Cursor = psycopg.Cursor[dict[str, Any]]
# TODO(wbbradley): Probably better to use Floats1d or np.array or some-such here.
Vector = np.ndarray
