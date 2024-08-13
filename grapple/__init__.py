import os
import subprocess
from typing import List, NamedTuple, Tuple, Union, reveal_type

import click
import numpy as np  # type: ignore
import psycopg
import spacy  # type: ignore
from neo4j import GraphDatabase  # type: ignore
from openai import OpenAI
from psycopg.types.json import Json
from tqdm import tqdm  # type: ignore

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


def get_sentence_embedding(
    cursor: psycopg.Cursor,
    sentence: str,
    model: str,
) -> Tuple[int, Vector]:
    sentence = sentence.text.strip()
    cursor.execute(
        "SELECT id, vector FROM embedding WHERE sentence_text = %s AND model = %s",
        (sentence, model),
    )
    result = cursor.fetchone()

    if result is None:
        vector = (
            openai_client.embeddings.create(
                input=sentence,
                model=model,
            )
            .data[0]
            .embedding
        )
        embedding_id = upsert_embedding(cursor, sentence, model, vector)
    else:
        embedding_id = result[0]
        vector = result[1]
    return embedding_id, vector


def upsert_embedding(cursor, sentence: str, model: str, vector: Vector) -> int:
    cursor.execute(
        """
            INSERT INTO embedding (sentence_text, model, vector, created_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (sentence_text, model) DO NOTHING
            RETURNING id
        """,
        (sentence, model, Json(vector)),
    )
    result = cursor.fetchone()
    cursor.connection.commit()
    return result[0]


def process_document(
    nlp: spacy.language.Language, file_path: str, openai_embedding_model: str
) -> None:
    with psycopg.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
    ) as conn:
        with conn.cursor() as cursor:
            with open(file_path, "r") as file:
                text = file.read()

            with Timer("run nlp on doc"):
                doc: spacy.tokens.doc.Doc = nlp(text)

            for i, sent in enumerate(tqdm(doc.sents)):
                embedding_id, vector = get_sentence_embedding(
                    cursor,
                    sent,
                    openai_embedding_model,
                )


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
    reveal_type(nlp)
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


"""
def process_query(
    doc: spacy.tokens.doc.Doc,
    embeddings: List,
    query_str: str,
    driver: Any,
    top_n: int,
) -> List[Tuple[float, str]]:
    # query_vec = normalize(get_sentence_embedding(nlp(query_str).vector))
    # similarities = [ ((1 - cosine(query_vec, vec)), sent.text) for sent, vec in embeddings ]
    # similarities.sort()
    # return similarities[:top_n]
    pass

"""


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
