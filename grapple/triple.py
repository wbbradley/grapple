from pprint import pprint
from typing import List

from pydantic import BaseModel

from grapple.metrics import metrics_count
from grapple.openai import openai_client
from grapple.types import Cursor, Vector


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
    cursor: Cursor,
    query_embedding: Vector,
) -> List[SemanticTriple]:
    associated_triples = cursor.execute(
        """
        WITH nearest_embeddings AS (
          SELECT uuid, (vector <-> %s) AS distance
          FROM embedding
          ORDER BY distance
          LIMIT %s
        ), enriched_triples AS (
          SELECT
            t.id,
            t.created_at,
            p.text AS paragraph_text,
            es.text AS subject_text,
            ep.text AS predicate_text,
            eo.text AS object_text,
            esu.text AS summary_text,
            t.summary_uuid AS summary_uuid
          FROM triple t
          LEFT JOIN paragraph p ON t.paragraph_uuid = p.uuid
          LEFT JOIN embedding es ON t.subject_uuid = es.uuid
          LEFT JOIN embedding ep ON t.predicate_uuid = ep.uuid
          LEFT JOIN embedding eo ON t.object_uuid = eo.uuid
          LEFT JOIN embedding esu ON t.summary_uuid = esu.uuid
        )
        SELECT
           subject_text subject,
           predicate_text predicate,
           object_text object,
           summary_text summary,
           ne.distance distance
        FROM enriched_triples et
        JOIN nearest_embeddings ne ON et.summary_uuid = ne.uuid
        ORDER BY et.created_at
    """,
        (query_embedding, 3),
    ).fetchall()
    pprint(associated_triples)
    return []
