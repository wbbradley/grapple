import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union
from uuid import UUID

import click
import psycopg
import spacy
import tiktoken
from psycopg.rows import dict_row
from psycopg.types.json import Json
from pydantic import BaseModel
from scipy.spatial.distance import cosine  # type: ignore
from tqdm import tqdm

from grapple.document import Document
from grapple.metrics import metrics_count
from grapple.openai import openai_client
from grapple.paragraph import Paragraph, get_paragraphs
from grapple.timer import Timer
from grapple.triple import SemanticTriple, get_triples
from grapple.utils import str_sha256, str_to_uuid

DEFAULT_SPACY_MODEL = "en_core_web_lg"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

# TODO(wbbradley): We are using dictionaries for all cursors at the moment. Clean this up later.
Cursor = psycopg.Cursor[dict[str, Any]]
# TODO(wbbradley): Probably better to use Floats1d or np.array or some-such here.
Vector = List[float]

logging.basicConfig(
    filename=os.path.expanduser("~/grapple.log"),
    level=logging.INFO,
    filemode="a",
)
logging.getLogger("httpx").setLevel(logging.WARN)


@click.group()
def main() -> None:
    pass


class GraphNode(NamedTuple):
    id: str
    description: str


class GraphEdge(NamedTuple):
    id: str
    edge_description: str
    edge_embedding: List[float]
    subject_node_id: str
    object_node_id: str


GraphItem = Union[GraphEdge, GraphNode]


_text_embeddings_cache: Dict[str, Tuple[UUID, List[float]]] = {}


def get_existing_text_embedding(
    cursor: Cursor,
    text: str,
    openai_embedding_model: str,
) -> Optional[Tuple[UUID, List[float]]]:
    return_val = _text_embeddings_cache.get(text)
    if return_val is not None:
        return return_val

    cursor.execute(
        "SELECT uuid, vector FROM embedding WHERE text = %s AND model = %s",
        (text, openai_embedding_model),
    )
    result = cursor.fetchone()
    if result is not None:
        return_val = _text_embeddings_cache[text] = (result["uuid"], result["vector"])
        return return_val
    return None


def get_text_embedding(
    cursor: Cursor,
    text: str,
    openai_embedding_model: str,
) -> Tuple[UUID, Vector]:
    if result := get_existing_text_embedding(cursor, text, openai_embedding_model):
        metrics_count("embedding.cache.hit")
        return result
    else:
        metrics_count("embedding.cache.miss")
        metrics_count(
            "embeddings.create",
            value=1,
            tags={"provider": "openai", "model": openai_embedding_model},
        )
        vector = (
            openai_client.embeddings.create(
                input=text,
                model=openai_embedding_model,
            )
            .data[0]
            .embedding
        )
        uuid = str_to_uuid(text)

        # Install this embedding into the cache.
        _text_embeddings_cache[text] = (uuid, vector)

        # Also emplace the embedding into the db.
        upsert_embedding(cursor, uuid, text, openai_embedding_model, vector)

        return uuid, vector


def upsert_embedding(cursor, uuid: UUID, text: str, model: str, vector: Vector) -> None:
    cursor.execute(
        """
            INSERT INTO embedding (uuid, text, model, vector)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (text, model) DO NOTHING
        """,
        (uuid, text, model, Json(vector)),
    )


def process_document_sentences(
    nlp: spacy.language.Language,
    file_path: str,
    openai_embedding_model: str,
) -> None:
    with open(file_path, "r") as file:
        text = file.read()

    with Timer("run nlp on doc"):
        nlp.max_length = 4000000
        doc: spacy.tokens.doc.Doc = nlp(text)
        sentences = list(doc.sents)

    with db_cursor() as cursor:
        ensure_sentence_embeddings(cursor, sentences, openai_embedding_model)


def process_document_paragraphs(
    nlp: spacy.language.Language,
    file_path: str,
    openai_embedding_model: str,
) -> None:
    with open(file_path, "r") as file:
        text = file.read()
    document = Document(filename=file_path, sha256=str_sha256(text))
    paragraphs = get_paragraphs(document, text)
    with Timer("run nlp on each paragraph"):
        with db_cursor() as cursor:
            for paragraph in tqdm(paragraphs):
                ensure_semantic_triples_for_paragraph(
                    cursor, paragraph, openai_embedding_model
                )
                metrics_count("db.commit")
                cursor.connection.commit()


def paragraph_exists_in_db(cursor: Cursor, paragraph_uuid: UUID) -> bool:
    # Paragraphs use content-addressable sentinels in the db to indicate whether they've been processed.
    result = cursor.execute(
        "SELECT COUNT(*) FROM paragraph WHERE uuid=%s", (paragraph_uuid,)
    ).fetchone()
    assert result

    matching_paragraph_count: int = result["count"]
    return matching_paragraph_count != 0


def ensure_semantic_triples_for_paragraph(
    cursor: Cursor,
    paragraph: Paragraph,
    openai_embedding_model: str,
) -> None:
    if paragraph_exists_in_db(cursor, paragraph.uuid):
        # Paragraph has already been processed, skip.
        return

    def _get_uuid(x: str) -> UUID:
        return get_text_embedding(cursor, x, openai_embedding_model)[0]

    # TODO: multi-pass/transact this to avoid dupes in the event of failure midway.
    triples: List[SemanticTriple] = get_triples(paragraph.text)
    sql_triples: List[Tuple[UUID, UUID, UUID, UUID, UUID]] = []
    for triple in triples:
        logging.info(triple.summary)
        sql_triples.append(
            (
                paragraph.uuid,
                _get_uuid(triple.subject),
                _get_uuid(triple.predicate),
                _get_uuid(triple.object),
                _get_uuid(triple.summary),
            )
        )
    cursor.executemany(
        """
        INSERT INTO triple
            (paragraph_uuid, subject_uuid, predicate_uuid, object_uuid, summary_uuid)
        VALUES
            (%s, %s, %s, %s, %s)
        """,
        sql_triples,
    )

    # Mark this paragraph as done.
    cursor.execute(
        "INSERT INTO paragraph (uuid, text) VALUES (%s, %s)",
        (
            paragraph.uuid,
            paragraph.text,
        ),
    )


@main.command()
@click.argument("filename")
def show_paragraphs(filename: str) -> None:
    with open(filename, "r") as file:
        text = file.read()
    document = Document(filename=filename, sha256=str_sha256(text))
    paragraphs = get_paragraphs(document, text)
    for paragraph in paragraphs:
        print(paragraph.text)
        input("Press Enter.")


@main.command()
def migrate() -> None:
    path = Path(__file__).resolve().parent.parent / "schema.sql"

    logging.info(f"[migrate] opening {path}...")
    with open(path, "rt") as f:
        logging.info(f"[migrate] read {path}...")
        content = f.read()
    with db_cursor() as cursor:
        logging.info(f"[migrate] executing {path}...")
        cursor.execute(content)
        cursor.connection.commit()


def ensure_sentence_embeddings(
    cursor: Cursor,
    sentences: List[spacy.tokens.span.Span],
    model: str,
):
    chunk_size = 200
    chunk = []
    for sentence in tqdm(sentences):
        sentence_text = re.sub(r"\s+", " ", sentence.text).strip()
        if num_tokens_from_string(sentence_text, "cl100k_base") > 8000:
            # Skip super-long sentences.
            # logging.info(f"skipping sentence {sentence_text} because it has too many tokens")
            continue
        if get_existing_text_embedding(cursor, sentence_text, model):
            # logging.info(f"skipping sentence {sentence_text} because it already exists in db")
            continue
        chunk.append(sentence_text)
        if len(chunk) >= chunk_size:
            bulk_upsert_embeddings(cursor, chunk, model)
            chunk = []
    bulk_upsert_embeddings(cursor, chunk, model)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def bulk_upsert_embeddings(
    cursor: Cursor,
    sentences: List[str],
    model: str,
):
    if len(sentences) == 0:
        return
    metrics_count(
        "embeddings.create",
        len(sentences),
        tags={"provider": "openai", "model": model},
    )
    vectors = [
        x.embedding
        for x in openai_client.embeddings.create(
            input=sentences,
            model=model,
        ).data
    ]
    for sentence, vector in zip(sentences, vectors):
        uuid = str_to_uuid(sentence)
        upsert_embedding(cursor, uuid, sentence, model, vector)


@main.command()
@click.argument("model", default=DEFAULT_SPACY_MODEL)
def download_spacy_model(model: str) -> None:
    os.system(f".venv/bin/python -m spacy download {model}")


@main.command(name="ingest")
@click.argument("filename")
@click.option("--openai-embedding-model", default=DEFAULT_OPENAI_EMBEDDING_MODEL)
@click.option("--spacy-nlp-model", default=DEFAULT_SPACY_MODEL)
def ingest(filename: str, openai_embedding_model: str, spacy_nlp_model: str) -> None:
    """Read a book and extract subject-predicate-object triples."""
    nlp = spacy.load(spacy_nlp_model)
    with Timer(f"process document paragraphs [filename={filename}]"):
        process_document_paragraphs(nlp, filename, "text-embedding-3-large")


@main.command()
@click.option("--openai-embedding-model", default=DEFAULT_OPENAI_EMBEDDING_MODEL)
def query(openai_embedding_model: str) -> None:
    # filename -> source -> doc -> chunks -> triplets with embeddings and summaries -> storage
    # query -> embedding -> vector query -> gather related edges and nodes -> RAG prompt with query

    # _driver = GraphDatabase.driver("bolt://localhost:7687")

    while True:
        query_str = input("Enter your query: ")
        with db_cursor() as cursor:
            query_embedding: Vector = get_text_embedding(
                cursor,
                query_str,
                openai_embedding_model,
            )[1]
            embeddings_with_distance = get_k_nearest_embeddings(
                cursor,
                query_embedding,
                top_n=10,
            )

        for ewd in embeddings_with_distance:
            logging.info(f"{ewd.distance}: {ewd.embedding.text}")


class Embedding(BaseModel):
    id: int
    text: str
    model: str
    vector: List[float]


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
                SELECT id, sentence_text, model, vector
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


@contextmanager
def db_cursor(row_factory: Any = dict_row) -> Iterator[Cursor]:
    with psycopg.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
    ) as conn:
        with conn.cursor(row_factory=row_factory) as cursor:
            yield cursor


@main.command()
def testdb() -> None:
    with db_cursor() as cursor:
        cursor.execute("SELECT 1, 2 UNION SELECT 3, 4")
        result = cursor.fetchone()
        print(result)


def fetch_embedding(text: str) -> Vector:
    raise NotImplementedError()


def fetch_nearest_graph_items(edge_embedding: Vector, top_n: int) -> List[GraphItem]:
    raise NotImplementedError()


if __name__ == "__main__":
    main()
