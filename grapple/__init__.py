import os
import subprocess
import sys
import time
from pprint import pprint
from typing import Any, Generator, List, Optional, Tuple

import numpy as np  # type: ignore
import spacy  # type: ignore
from neo4j import GraphDatabase  # type: ignore
from openai import OpenAI
from psycopg2.extras import Json
from scipy.spatial.distance import cosine  # type: ignore

openai_key = (
    subprocess.check_output("pass openai-api-key", shell=True).strip().decode("utf-8")
)
client = OpenAI(api_key="your-api-key")
DEFAULT_SPACY_MODEL = "en_core_web_lg"


def get_sentence_embeddings(sentence: str, model: str) -> list:
    return client.embeddings.create(
        input=sentence,
        model=model,
    )["data"][0]["embedding"]


def upsert_embedding(cursor, sentence: str, model: str, vector: list) -> None:
    cursor.execute(
        """
            INSERT INTO embedding (sentence_text, model, vector, created_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (sentence_text, model) DO NOTHING
        """,
        (sentence, model, Json(vector)),
    )


def insert_provenance(
    cursor, sentence: str, model: str, document: str, position: int
) -> None:
    cursor.execute(
        """
        INSERT INTO provenance (embedding_sentence_text, embedding_model, document_name, sentence_position)
        VALUES (%s, %s, %s, %s)
    """,
        (sentence, model, document, position),
    )


def process_document(nlp: Any, file_path: str, model: str) -> None:
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="localhost",
    )
    cursor = conn.cursor()

    with open(file_path, "r") as file:
        text = file.read()

    doc = nlp(text)

    for i, sent in enumerate(doc.sents):
        sentence = sent.text.strip()
        cursor.execute(
            "SELECT vector FROM embedding WHERE sentence_text = %s AND model = %s",
            (sentence, model),
        )
        result = cursor.fetchone()

        if result is None:
            vector = get_sentence_embeddings(sentence, model)
            upsert_embedding(cursor, sentence, model, vector)
        insert_provenance(cursor, sentence, model, file_path, i)

    conn.commit()
    cursor.close()
    conn.close()


def init(model: str) -> None:
    os.system(".venv/bin/python -m spacy download {model}")


class Timer:
    def __init__(self, name: Optional[str] = "block") -> None:
        self.name = name

    def __enter__(self) -> None:
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"{self.name} took {elapsed_time:.2f} seconds.")


def read_txt_file(filename: str, model: str) -> None:
    """Read a book and extract subject-predicate-object triples."""
    nlp = spacy.load(model)

    with Timer("run nlp on doc"):
        process_document(nlp, filename, "text-embedding-3-large")

        # triples = extract_triples(doc)
    # store_triples(triples, driver)


def normalize(v: np.array) -> np.array:
    norm = np.linalg.norm(v)
    if norm != 0:
        return v / norm
    return v


def sentence_embeddings(doc: Any) -> Generator[Tuple[Any, np.ndarray], None, None]:
    for sent in doc.sents:
        yield sent, sent.vector


# Function for querying the graph database
def query(filename: str, model: str) -> None:
    driver = GraphDatabase.driver("bolt://localhost:7687")
    nlp = spacy.load(model)

    with Timer(f"read file text ({filename})"):
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()

    with Timer(f"run nlp ({model}) on {filename}"):
        doc = nlp(text)

    with Timer("generate sentence embeddings"):
        embeddings = list(sentence_embeddings(doc))

    while True:
        query_str = input("Enter your query: ")
        similarities = process_query(nlp, doc, embeddings, query_str, driver, 10)
        pprint(similarities)


def process_query(
    nlp: Any, doc: Any, embeddings: List, query_str: str, driver: Any, top_n: int
) -> List[Tuple[float, str]]:
    query_vec = normalize(nlp(query_str).vector)
    similarities = [
        ((1 - cosine(query_vec, vec)), sent.text) for sent, vec in embeddings
    ]
    similarities.sort()
    return similarities[:top_n]


# Main function to handle subcommands
def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: grapple <init|read|query>")

    model = DEFAULT_SPACY_MODEL
    subcommand = sys.argv[1]

    if subcommand == "init":
        init(model)
    elif subcommand == "embed":
        if len(sys.argv) < 3:
            sys.exit("Usage: grapple embed <text>")
        nlp = spacy.load(model)
        doc = nlp(sys.argv[2])
        for sent in doc.sents:
            print(f"Sentence: {sent.text}\nVector: {sent.vector}")
    elif subcommand == "read":
        if len(sys.argv) < 3:
            sys.exit("Usage: grapple read <filename>")
        read_txt_file(sys.argv[2], model=model)
    elif subcommand == "query":
        if len(sys.argv) < 3:
            sys.exit("Usage: grapple query <filename>")
        query(sys.argv[2], model=model)
    else:
        print(f"Unknown subcommand: {subcommand}")


if __name__ == "__main__":
    main()
