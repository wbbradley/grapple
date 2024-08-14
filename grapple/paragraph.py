import re
from typing import List, Tuple
from uuid import UUID

from pydantic import BaseModel

from grapple.document import Document
from grapple.utils import str_to_uuid


class Paragraph(BaseModel):
    text: str
    uuid: UUID
    document: Document
    index_span: Tuple[int, int]


def get_paragraphs(document: Document, text: str) -> List[Paragraph]:
    text = text.strip()
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
            text=paragraph_text,
            uuid=str_to_uuid(paragraph_text),
            document=document,
            index_span=(start, end),
        )

        paragraphs.append(paragraph)

    while i < len(text):
        if text[i] == "\n":
            newline_count += 1
            if newline_count > 1:
                add_paragraph_text(text[start:i], start, i)
                start = i + 1
                newline_count = 0
        i += 1
    add_paragraph_text(text[start:i], start, i)
    return paragraphs
