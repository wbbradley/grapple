from pydantic import BaseModel


class Tag(BaseModel):
    id: int
    text: str
