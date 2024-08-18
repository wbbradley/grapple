from uuid import UUID

from pydantic import BaseModel


class Document(BaseModel):
    uuid: UUID
    filename: str
