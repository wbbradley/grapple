import logging
import re
import time
from typing import List
from uuid import UUID

import openai
from pydantic import BaseModel

from grapple.metrics import metrics_count, metrics_timer
from grapple.openai import openai_client
from grapple.paragraph import Paragraph
from grapple.types import Cursor, Vector

RETRY_TIMEOUT_SECONDS = 30.0

TRIPLES_DESIRED_PER_PARAGRAPH = 15
TRIPLES_PROMPT = re.sub(
    r"\s+",
    " ",
    f"""
        Examine the text below. Extract up to {TRIPLES_DESIRED_PER_PARAGRAPH} semantic triples for
        all information contained in the text. Do not infer triples for subjects that are not
        mentioned within the given text.
    """,
).strip()


class Triple(BaseModel):
    paragraph_uuid: UUID
    subject: str
    predicate: str
    object: str
    summary: str


class OpenAIQueryTriple(BaseModel):
    subject: str
    predicate: str
    object: str
    summary: str


class OpenAIQueryTriples(BaseModel):
    triples: List[OpenAIQueryTriple]


def get_triples_from_text(text: str, paragraph_uuid: UUID) -> List[Triple]:
    model = "gpt-4o-2024-08-06"
    with metrics_timer("openai.request.get-triples-from-text"):
        while True:
            try:
                metrics_count(
                    f"beta.chat.completions.parse.{model}",
                )
                completion = openai_client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": TRIPLES_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": text,
                        },
                    ],
                    response_format=OpenAIQueryTriples,
                )
                break
            except openai.RateLimitError as e:
                metrics_count("openai.errors.rate-limited")
                logging.error(f"query_gpt_model: RateLimitError {e.message}: {e}")
                time.sleep(RETRY_TIMEOUT_SECONDS)
            except openai.APIError as e:
                metrics_count("openai.errors.api-error")
                logging.error(f"query_gpt_model: APIError {e.message}: {e}")
                logging.error("query_gpt_model: Retrying after 5 seconds...")
                time.sleep(5)
    if parsed := completion.choices[0].message.parsed:
        metrics_count("triples.parsed", value=len(parsed.triples))
        return [
            Triple(
                paragraph_uuid=paragraph_uuid,
                subject=x.subject,
                predicate=x.predicate,
                object=x.object,
                summary=x.summary,
            )
            for x in parsed.triples
        ]
    return []


def get_triples(paragraph: Paragraph) -> List[Triple]:
    return get_triples_from_text(paragraph.text, paragraph.uuid)


class GatheredTriple(Triple):
    id: int
    distance: float
    paragraph_uuid: UUID


#
# Notes on pgvector operators.
#
# `<->` (L2 distance):
# Measures the straight-line distance between points in Euclidean space. Unnormalized vectors
# directly affect the magnitude of this distance.
#
# `<#>` (negative inner product):
# Unnormalized vectors affect the magnitude of the product, but generally results in a larger
# negative value with larger vectors.
#
# `<=>` (cosine distance):
# Calculates the angle between vectors and is typically less sensitive to the magnitude, but
# primarily affected by the direction. For unnormalized vectors, you may need to normalize them to
# get accurate angular measurements.
#
# `<+>` (L1 distance):
# Measures the sum of the absolute differences of their coordinates. Larger vector values result in
# a proportionally larger L1 distance.
#
def gather_related_triples(
    cursor: Cursor,
    query_embedding: Vector,
) -> List[GatheredTriple]:
    return list(
        map(
            GatheredTriple.parse_obj,
            cursor.execute(
                """
                    WITH nearest_embeddings AS (
                      SELECT
                          uuid,
                          (vector <-> %s) AS distance
                      FROM embedding
                    ), enriched_triples AS (
                      SELECT
                        t.id,
                        t.created_at,
                        p.text AS paragraph_text,
                        p.uuid AS paragraph_uuid,
                        es.text AS subject_text,
                        ep.text AS predicate_text,
                        eo.text AS object_text,
                        esu.text AS summary_text,
                        t.subject_uuid AS subject_uuid,
                        t.predicate_uuid AS predicate_uuid,
                        t.object_uuid AS object_uuid,
                        t.summary_uuid AS summary_uuid
                      FROM triple t
                      LEFT JOIN paragraph p ON t.paragraph_uuid = p.uuid
                      LEFT JOIN embedding es ON t.subject_uuid = es.uuid
                      LEFT JOIN embedding ep ON t.predicate_uuid = ep.uuid
                      LEFT JOIN embedding eo ON t.object_uuid = eo.uuid
                      LEFT JOIN embedding esu ON t.summary_uuid = esu.uuid
                    )
                    SELECT
                       DISTINCT et.id as id,
                       et.paragraph_uuid,
                       subject_text subject,
                       predicate_text predicate,
                       object_text object,
                       summary_text summary,
                       ne.distance distance
                    FROM enriched_triples et
                    JOIN nearest_embeddings ne ON ne.uuid in (et.summary_uuid, et.subject_uuid,
                                                              et.predicate_uuid, et.object_uuid)
                    ORDER BY distance
                    LIMIT %s
                """,
                (query_embedding, 10),
            ),
        )
    )
