from uuid import UUID

from pydantic import BaseModel

from grapple.types import Cursor


class Document(BaseModel):
    uuid: UUID
    filename: str


def upsert_document(cursor: Cursor, document: Document) -> None:
    cursor.execute(
        """
            INSERT INTO document (uuid, filename)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
        """,
        (document.uuid, document.filename),
    )
