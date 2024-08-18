import re
from typing import List
from uuid import UUID

from pydantic import BaseModel

from grapple.document import Document
from grapple.types import Cursor
from grapple.utils import str_to_uuid


class Paragraph(BaseModel):
    uuid: UUID
    text: str
    document_uuid: UUID
    span_index_start: int
    span_index_lim: int

    @staticmethod
    def make_uuid(document_uuid: UUID, start: int, lim: int) -> UUID:
        return str_to_uuid(f"{document_uuid}:{start}:{lim}")


def get_paragraph(cursor: Cursor, paragraph_uuid: UUID) -> Paragraph:
    return Paragraph.parse_obj(
        cursor.execute(
            """
                SELECT
                    uuid,
                    text,
                    document_uuid,
                    span_index_start,
                    span_index_lim
                FROM paragraph
                WHERE uuid=%s
            """,
            (paragraph_uuid,),
        ).fetchone()
    )


def get_paragraphs(document: Document, text: str) -> List[Paragraph]:
    text = text.strip().replace("\r\n", "\n").replace("\r", "")
    start = 0
    i = 0
    newline_count = 0
    paragraphs: List[Paragraph] = []

    def add_paragraph_text(paragraph_text: str, start: int, end: int) -> None:
        """Maybe append to paragraphs."""
        paragraph_text = re.sub(r"\s+", " ", paragraph_text).strip()
        if not paragraph_text:
            return

        paragraph = Paragraph(
            uuid=Paragraph.make_uuid(document.uuid, start, end),
            text=paragraph_text,
            document_uuid=document.uuid,
            span_index_start=start,
            span_index_lim=end,
        )

        paragraphs.append(paragraph)

    while i < len(text):
        if text[i] == "\n":
            newline_count += 1
            if newline_count > 1:
                add_paragraph_text(text[start:i], start, i)
                start = i + 1
                newline_count = 0
        else:
            newline_count = 0
        i += 1
    add_paragraph_text(text[start:i], start, i)
    return paragraphs
