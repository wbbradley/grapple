from dataclasses import dataclass
from typing import List
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from scipy.spatial.distance import cosine  # type: ignore

from grapple.timer import Timer
from grapple.types import Cursor, Vector


class Embedding(BaseModel):
    uuid: UUID
    text: str
    model: str
    vector: Vector

    # Facilitate loading np.array into vector.
    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class EmbeddingWithDistance:
    embedding: Embedding
    distance: float


def make_embedding_with_distance(
    query: Vector, embedding: Embedding
) -> EmbeddingWithDistance:
    return EmbeddingWithDistance(
        embedding=embedding, distance=cosine(embedding.vector, query)
    )


_embeddings: List[Embedding] = []


def fetch_all_embeddings(cursor: Cursor) -> List[Embedding]:
    # Cached once per process.
    if _embeddings:
        return _embeddings

    with Timer("gather all embeddings"):
        cursor.execute(
            """
                SELECT uuid, text, model, vector
                FROM embedding
                ORDER BY created_at
            """
        )
        for row in cursor:
            _embeddings.append(Embedding.parse_obj(row))
        return _embeddings


def get_k_nearest_embeddings(
    cursor: Cursor, query: Vector, top_n: int
) -> List[EmbeddingWithDistance]:
    embeddings = fetch_all_embeddings(cursor)
    with Timer("calculating cosine distances"):
        embeddings_with_distance = [
            EmbeddingWithDistance(embedding=x, distance=cosine(x.vector, query))
            for x in embeddings
        ]
    embeddings_with_distance.sort(key=lambda x: x.distance)
    return embeddings_with_distance[:top_n]
