import os
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, NamedTuple, Optional, Tuple, Union
from uuid import UUID

import click
import numpy as np
import psycopg
import spacy
import tiktoken
from openai import OpenAI
from psycopg.rows import dict_row
from psycopg.types.json import Json
from pydantic import BaseModel
from scipy.spatial.distance import cosine  # type: ignore
from tqdm import tqdm

from grapple.document import Document
from grapple.paragraph import Paragraph, get_paragraphs
from grapple.timer import Timer
from grapple.utils import str_sha256

# GRAPPLE_OPENAI_KEY_CMD: must be a command that prints your OpenAI API key to stdout.
openai_client = OpenAI(
    api_key=subprocess.check_output(
        os.environ.get("GRAPPLE_OPENAI_KEY_CMD", "pass openai-api-key"), shell=True
    )
    .decode("utf-8")
    .strip()
)
DEFAULT_SPACY_MODEL = "en_core_web_lg"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

# TODO(wbbradley): We are using dictionaries for all cursors at the moment. Clean this up later.
Cursor = psycopg.Cursor[dict[str, Any]]
# TODO(wbbradley): Probably better to use Floats1d or np.array or some-such here.
Vector = List[float]


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


def get_existing_sentence_embedding(
    cursor: Cursor,
    sentence: str,
    openai_embedding_model: str,
) -> Optional[Tuple[int, List[float]]]:
    cursor.execute(
        "SELECT id, vector FROM embedding WHERE sentence_text = %s AND model = %s",
        (sentence, openai_embedding_model),
    )
    result = cursor.fetchone()
    if result is not None:
        return result["id"], result["vector"]
    return None


def get_sentence_embedding(
    cursor: Cursor,
    sentence: str,
    openai_embedding_model: str,
) -> Vector:
    if result := get_existing_sentence_embedding(
        cursor, sentence, openai_embedding_model
    ):
        print(f"[get_sentence_embedding] found result for ({sentence}) in db")
        return result[1]
    else:
        vector = (
            openai_client.embeddings.create(
                input=sentence,
                model=openai_embedding_model,
            )
            .data[0]
            .embedding
        )
        upsert_embedding(cursor, sentence, openai_embedding_model, vector)
        return vector


def upsert_embedding(cursor, sentence: str, model: str, vector: Vector) -> None:
    cursor.execute(
        """
            INSERT INTO embedding (sentence_text, model, vector, created_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (sentence_text, model) DO NOTHING
            RETURNING id
        """,
        (sentence, model, Json(vector)),
    )
    cursor.connection.commit()


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
            for paragraph in paragraphs:
                ensure_semantic_triples_for_paragraph(cursor, paragraph)


def paragraph_exists_in_db(cursor: Cursor, paragraph_uuid: UUID) -> bool:
    # Paragraphs use content-addressable sigils in the db to indicate whether they've been processed.
    result = cursor.execute(
        "SELECT COUNT(*) FROM paragraph WHERE uuid=%s", (paragraph_uuid,)
    ).fetchone()
    assert result

    matching_paragraph_count: int = result["count"]
    return matching_paragraph_count != 0


def ensure_semantic_triples_for_paragraph(cursor: Cursor, paragraph: Paragraph) -> None:
    if paragraph_exists_in_db(cursor, paragraph.uuid):
        return

    # Paragraph has not yet been processed, proceed.
    inner_tx: psycopg.Transaction
    with cursor.connection.transaction() as inner_tx:
        triples: List[SemanticTriple] = get_triples(paragraph.text)
        for triple in triples:
            cursor.execute("INSERT INTO triple")
            assert False


@main.command()
def migrate() -> None:
    path = Path(__file__).resolve().parent.parent / "schema.sql"

    print(f"[migrate] opening {path}...")
    with open(path, "rt") as f:
        print(f"[migrate] read {path}...")
        content = f.read()
    with db_cursor() as cursor:
        print(f"[migrate] executing {path}...")
        cursor.execute(content)
        cursor.connection.commit()


class SemanticTriple(BaseModel):
    subject: str
    predicate: str
    object: str
    summary: str


class SemanticTriples(BaseModel):
    triples: List[SemanticTriple]


def get_triples(paragraph: str) -> List[SemanticTriple]:
    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Context:\n\n--- BEGIN ---\n{paragraph}\n--- END ---\n\nInstruction: "
                    f"Please examine the text within the BEGIN and END blocks above and "
                    f"extract semantic triples for all information contained therein, also include a summary "
                    f"describing the found fact."
                ),
            }
        ],
        response_format=SemanticTriples,
    )
    if parsed := completion.choices[0].message.parsed:
        return parsed.triples
    return []


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
            # print(f"skipping sentence {sentence_text} because it has too many tokens")
            continue
        if get_existing_sentence_embedding(cursor, sentence_text, model):
            # print(f"skipping sentence {sentence_text} because it already exists in db")
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
    vectors = [
        x.embedding
        for x in openai_client.embeddings.create(
            input=sentences,
            model=model,
        ).data
    ]
    for sentence, vector in zip(sentences, vectors):
        upsert_embedding(cursor, sentence, model, vector)


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

        # triples = extract_triples(doc)
    # store_triples(triples, driver)


def normalize(v: np.array) -> np.array:
    norm = np.linalg.norm(v)
    if norm != 0:
        return v / norm
    return v


@main.command()
@click.option("--openai-embedding-model", default=DEFAULT_OPENAI_EMBEDDING_MODEL)
def query(openai_embedding_model: str) -> None:
    # filename -> source -> doc -> chunks -> triplets with embeddings and summaries -> storage
    # query -> embedding -> vector query -> gather related edges and nodes -> RAG prompt with query

    # _driver = GraphDatabase.driver("bolt://localhost:7687")

    while True:
        query_str = input("Enter your query: ")
        with db_cursor() as cursor:
            query_embedding: Vector = get_sentence_embedding(
                cursor,
                query_str,
                openai_embedding_model,
            )
            embeddings_with_distance = get_k_nearest_embeddings(
                cursor,
                query_embedding,
                top_n=10,
            )

        for ewd in embeddings_with_distance:
            print(f"{ewd.distance}: {ewd.embedding.sentence_text}")


class Embedding(BaseModel):
    id: int
    sentence_text: str
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
    # Pipelines:
    # filename -> source -> doc -> chunks -> triplets with embeddings and summaries -> storage
    # query -> embedding -> vector query -> gather related edges and nodes -> RAG prompt with query
    main()
