import os
import re
import subprocess
from typing import List, NamedTuple, Optional, Tuple, Union

import click
import numpy as np
import psycopg
import spacy
import tiktoken
from neo4j import GraphDatabase
from openai import OpenAI
from psycopg.types.json import Json
from tqdm import tqdm

from grapple.timer import Timer

# GRAPPLE_OPENAI_KEY_CMD: must be a command that can be run which returns your OpenAI API key.
openai_client = OpenAI(
    api_key=subprocess.check_output(
        os.environ.get("GRAPPLE_OPENAI_KEY_CMD", "pass openai-api-key"), shell=True
    )
    .strip()
    .decode("utf-8")
)
DEFAULT_SPACY_MODEL = "en_core_web_lg"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

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
    cursor: psycopg.Cursor,
    sentence: str,
    model: str,
) -> Optional[Tuple[int, List[float]]]:
    cursor.execute(
        "SELECT id, vector FROM embedding WHERE sentence_text = %s AND model = %s",
        (sentence, model),
    )
    return cursor.fetchone()


def get_sentence_embedding(
    cursor: psycopg.Cursor,
    sentence: str,
    model: str,
) -> Tuple[int, Vector]:
    if result := get_existing_sentence_embedding(cursor, sentence, model):
        vector = result[1]
    else:
        vector = (
            openai_client.embeddings.create(
                input=sentence,
                model=model,
            )
            .data[0]
            .embedding
        )
        upsert_embedding(cursor, sentence, model, vector)
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


def process_document(
    nlp: spacy.language.Language, file_path: str, openai_embedding_model: str
) -> None:
    with open(file_path, "r") as file:
        text = file.read()

    with Timer("run nlp on doc"):
        doc: spacy.tokens.doc.Doc = nlp(text)
        sentences = list(doc.sents)

    with psycopg.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
    ) as conn:
        with conn.cursor() as cursor:
            ensure_sentence_embeddings(cursor, sentences, openai_embedding_model)


def ensure_sentence_embeddings_core(
    cursor: psycopg.Cursor,
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


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def ensure_sentence_embeddings(
    cursor: psycopg.Cursor,
    sentences: List[spacy.tokens.span.Span],
    model: str,
):
    chunk_size = 100
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
            ensure_sentence_embeddings_core(cursor, chunk, model)
            chunk = []
    ensure_sentence_embeddings_core(cursor, chunk, model)


@main.command()
@click.argument("model", default=DEFAULT_SPACY_MODEL)
def download_spacy_model(model: str) -> None:
    os.system(f".venv/bin/python -m spacy download {model}")


@main.command(name="ingest")
@click.argument("filename")
@click.option("--openai-model", default=DEFAULT_OPENAI_EMBEDDING_MODEL)
@click.option("--spacy-model", default=DEFAULT_SPACY_MODEL)
def ingest(filename: str, openai_model: str, spacy_model: str) -> None:
    """Read a book and extract subject-predicate-object triples."""
    nlp = spacy.load(spacy_model)
    with Timer(f"process document [filename={filename}]"):
        process_document(nlp, filename, "text-embedding-3-large")

        # triples = extract_triples(doc)
    # store_triples(triples, driver)


def normalize(v: np.array) -> np.array:
    norm = np.linalg.norm(v)
    if norm != 0:
        return v / norm
    return v


@main.command()
def query(filename: str, model: str) -> None:
    # filename -> source -> doc -> chunks -> triplets with embeddings and summaries -> storage
    # query -> embedding -> vector query -> gather related edges and nodes -> RAG prompt with query

    _driver = GraphDatabase.driver("bolt://localhost:7687")

    while True:
        query_str = input("Enter your query: ")
        query_embedding = fetch_embedding(query_str)
        _graph_items = fetch_nearest_graph_items(query_embedding, top_n=25)
        # similarities = process_query(embeddings, query_str, driver, 10)
        # pprint(similarities)


@main.command()
def testdb() -> None:
    with psycopg.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1, 2 UNION SELECT 3, 4")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
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
