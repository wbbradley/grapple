from pydantic import BaseModel


class Document(BaseModel):
    sha256: str
    filename: str
