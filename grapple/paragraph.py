import re
from typing import Iterator, List, Optional
from uuid import UUID

from pydantic import BaseModel

from grapple.document import Document
from grapple.types import Cursor
from grapple.utils import num_tokens_from_string, str_to_uuid

# Tunable parameters probably correlated with whichever LLM model.
MAX_PARA_TOKENS = 3000


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
    """Attempt to find and merge paragraphs into chunks of maximum size below a threshold."""
    paragraphs = []
    current_paragraph = None
    current_tokens = 0
    for paragraph in generate_paragraphs(document, text):
        tokens = num_tokens_from_string(paragraph.text)
        if current_tokens + tokens > MAX_PARA_TOKENS:
            if current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = paragraph
                current_tokens = tokens
                continue
            paragraphs.append(paragraph)
            assert current_paragraph is None
            assert current_tokens == 0
        else:
            if current_paragraph:
                merged_para_text = current_paragraph.text + "\n" + paragraph.text
                span_index_lim = paragraph.span_index_lim
                current_paragraph = Paragraph(
                    uuid=Paragraph.make_uuid(
                        document.uuid, current_paragraph.span_index_start, span_index_lim
                    ),
                    text=merged_para_text,
                    document_uuid=document.uuid,
                    span_index_start=current_paragraph.span_index_start,
                    span_index_lim=span_index_lim,
                )
                current_tokens += tokens
            else:
                current_paragraph = paragraph
                current_tokens = tokens
    if current_paragraph:
        paragraphs.append(current_paragraph)
    return paragraphs


def generate_paragraphs(document: Document, text: str) -> Iterator[Paragraph]:
    text = text.strip().replace("\r\n", "\n").replace("\r", "")
    start = 0
    i = 0
    newline_count = 0

    def construct_paragraph(paragraph_text: str, start: int, end: int) -> Optional[Paragraph]:
        """Maybe append to paragraphs."""
        paragraph_text = re.sub(r"\s+", " ", paragraph_text).strip()
        if not paragraph_text:
            return None

        paragraph = Paragraph(
            uuid=Paragraph.make_uuid(document.uuid, start, end),
            text=paragraph_text,
            document_uuid=document.uuid,
            span_index_start=start,
            span_index_lim=end,
        )

        return paragraph

    while i < len(text):
        if text[i] == "\n":
            newline_count += 1
            if newline_count > 1:
                if paragraph := construct_paragraph(text[start:i], start, i):
                    yield paragraph
                start = i + 1
                newline_count = 0
        else:
            newline_count = 0
        i += 1
    if paragraph := construct_paragraph(text[start:i], start, i):
        yield paragraph
