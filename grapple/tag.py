# Copyright 2024, William Bradley, All rights reserved.
from pydantic import BaseModel


class Tag(BaseModel):
    id: int
    text: str
