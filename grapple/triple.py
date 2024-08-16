from typing import List

from pydantic import BaseModel

from grapple.embedding import EmbeddingWithDistance
from grapple.metrics import metrics_count
from grapple.openai import openai_client
from grapple.types import Cursor


class SemanticTriple(BaseModel):
    subject: str
    predicate: str
    object: str
    summary: str


class SemanticTriples(BaseModel):
    triples: List[SemanticTriple]


def get_triples(paragraph: str) -> List[SemanticTriple]:
    model = "gpt-4o-2024-08-06"
    metrics_count(
        "beta.chat.completions.parse",
        tags={"provider": "openai", "model": model},
    )
    completion = openai_client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Please examine the following text and extract semantic triples for "
                    "all information contained therein, also include a summary "
                    "describing the found fact."
                ),
            },
            {
                "role": "user",
                "content": paragraph,
            },
        ],
        response_format=SemanticTriples,
    )
    if parsed := completion.choices[0].message.parsed:
        metrics_count("triples.parsed")
        return parsed.triples
    return []


def gather_related_triples(
    cursor: Cursor, embeddings_with_distance: List[EmbeddingWithDistance]
) -> List[SemanticTriple]:
    raise NotImplementedError()
