import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from uuid import UUID

import click
import numpy as np
import psycopg
import spacy
from pgvector.psycopg import register_vector  # type: ignore
from psycopg.rows import dict_row
from tqdm import tqdm

from grapple.colors import colorize
from grapple.document import Document, upsert_document
from grapple.embedding import Embedding
from grapple.metrics import metrics_count, metrics_timer
from grapple.openai import get_completion, openai_client
from grapple.paragraph import Paragraph, get_paragraph, get_paragraphs
from grapple.timer import Timer
from grapple.triple import GatheredTriple, Triple, gather_related_triples, get_triples
from grapple.types import Cursor, Vector
from grapple.utils import num_tokens_from_string, str_to_uuid

DEFAULT_SPACY_MODEL = "en_core_web_lg"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

logging.basicConfig(filename=os.path.expanduser("~/grapple.log"), level=logging.INFO, filemode="a")
logging.getLogger("httpx").setLevel(logging.WARN)


@click.group()
def main() -> None:
    pass


_text_embeddings_cache: Dict[Tuple[str, str], Embedding] = {}


def get_existing_text_embedding(
    cursor: Cursor,
    text: str,
    openai_embedding_model: str,
) -> Optional[Embedding]:
    return_val = _text_embeddings_cache.get((text, openai_embedding_model))
    if return_val is not None:
        return return_val

    cursor.execute(
        """
            SELECT
                  uuid
                , text
                , model
                , vector
                -- , COALESCE((SELECT json_agg(json_build_object('id', t.id, 'text', t.text)) FROM tag t, embedding_tag et WHERE t.id=et.tag_id AND et.embedding_uuid=embedding.uuid), '[]'::json) tags
            FROM embedding WHERE text = %s AND model = %s
        """,
        (text, openai_embedding_model),
    )
    result = cursor.fetchone()
    if result is not None:
        return_val = _text_embeddings_cache[(text, openai_embedding_model)] = Embedding.parse_obj(
            result
        )
        return return_val
    return None


def get_text_embedding(
    cursor: Cursor,
    text: str,
    openai_embedding_model: str,
    tags: Optional[List[str]] = None,
) -> Embedding:
    assert not tags, "tags not yet impl"
    if result := get_existing_text_embedding(cursor, text, openai_embedding_model):
        metrics_count("embedding.cache.hit")
        return result
    else:
        metrics_count("embedding.cache.miss")
        metrics_count(f"embedding.create.{openai_embedding_model}", value=1)
        with metrics_timer("openai.request.embeddings"):
            vector = np.array(
                (
                    openai_client.embeddings.create(input=text, model=openai_embedding_model)
                    .data[0]
                    .embedding
                )
            )
        uuid = str_to_uuid(text)

        # Install this embedding into the cache.
        embedding = Embedding(
            uuid=uuid,
            text=text,
            model=openai_embedding_model,
            vector=vector,
        )
        _text_embeddings_cache[(text, openai_embedding_model)] = embedding
        # Also emplace the embedding into the db.
        upsert_embedding(cursor, embedding)

        return embedding


def upsert_embedding(cursor: Cursor, embedding: Embedding) -> None:
    cursor.execute(
        """
            INSERT INTO embedding (uuid, text, model, vector)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (text, model) DO NOTHING
        """,
        (embedding.uuid, embedding.text, embedding.model, embedding.vector),
    )


def process_document_sentences(
    nlp: spacy.language.Language, file_path: str, openai_embedding_model: str
) -> None:
    with open(file_path, "r") as file:
        text = file.read()

    with metrics_timer("compute.run-nlp-on-doc"):
        nlp.max_length = 4000000
        doc: spacy.tokens.doc.Doc = nlp(text)
        sentences = list(doc.sents)

    with db_cursor() as cursor:
        ensure_sentence_embeddings(cursor, sentences, openai_embedding_model)


def process_document_paragraphs(file_path: str, openai_embedding_model: str) -> None:
    with open(file_path, "r") as file:
        text = file.read()
    document = Document(filename=file_path, uuid=str_to_uuid(text))

    with db_cursor() as cursor:
        upsert_document(cursor, document)
    paragraphs = get_paragraphs(document, text)
    with db_cursor() as cursor:
        for paragraph in tqdm(paragraphs):
            ensure_semantic_triples_for_paragraph(
                cursor,
                document.uuid,
                paragraph,
                openai_embedding_model,
            )
            metrics_count("db.commit")
            cursor.connection.commit()


def paragraph_exists_in_db(cursor: Cursor, uuid: UUID) -> bool:
    result = cursor.execute(
        """
        SELECT 1
        FROM paragraph
        WHERE uuid = %s
        """,
        (uuid,),
    ).fetchone()
    return bool(result)


def ensure_semantic_triples_for_paragraph(
    cursor: Cursor,
    document_uuid: UUID,
    paragraph: Paragraph,
    openai_embedding_model: str,
) -> None:
    if paragraph_exists_in_db(cursor, paragraph.uuid):
        # Paragraph has already been processed, skip.
        return

    def _get_uuid(x: str) -> UUID:
        return get_text_embedding(cursor, x, openai_embedding_model).uuid

    # TODO: batch get_text_embedding calls
    triples: List[Triple] = get_triples(paragraph)
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
    with metrics_timer("db.insert-triples"):
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
            """
                INSERT INTO paragraph
                    (uuid, text, document_uuid, span_index_start, span_index_lim)
                VALUES (%s, %s, %s, %s, %s)
            """,
            (
                paragraph.uuid,
                paragraph.text,
                document_uuid,
                paragraph.span_index_start,
                paragraph.span_index_lim,
            ),
        )


@main.command()
@click.argument("filename")
def show_paragraphs(filename: str) -> None:
    with open(filename, "r") as file:
        text = file.read()
    text = text.strip()
    document = Document(filename=filename, uuid=str_to_uuid(text))
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
    with db_cursor(call_register_vector=False) as cursor:
        logging.info(f"[migrate] executing {path}...")
        cursor.execute(content)
        cursor.connection.commit()


def ensure_sentence_embeddings(cursor: Cursor, sentences: List[spacy.tokens.span.Span], model: str):
    chunk_size = 200
    chunk = []
    for sentence in tqdm(sentences):
        sentence_text = re.sub(r"\s+", " ", sentence.text).strip()
        if num_tokens_from_string(sentence_text) > 8000:
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


def bulk_upsert_embeddings(cursor: Cursor, sentences: List[str], model: str) -> None:
    sentences = [
        sentence
        for sentence in sentences
        if not get_existing_text_embedding(cursor, sentence, model)
    ]
    if len(sentences) == 0:
        # Save an iota of time.
        return

    metrics_count("embeddings.create.", len(sentences))
    metrics_count(f"embedding.create.{model}", value=len(sentences))

    with metrics_timer("openai.request.embeddings"):
        vectors = [
            np.array(x.embedding)
            for x in openai_client.embeddings.create(input=sentences, model=model).data
        ]
    for sentence, vector in zip(sentences, vectors):
        uuid = str_to_uuid(sentence)
        embedding = Embedding(
            uuid=uuid,
            text=sentence,
            model=model,
            vector=vector,
        )
        upsert_embedding(cursor, embedding)


@main.command()
@click.argument("model", default=DEFAULT_SPACY_MODEL)
def download_spacy_model(model: str) -> None:
    os.system(f".venv/bin/python -m spacy download {model}")


@main.command(name="ingest")
@click.argument("filename")
@click.option("--openai-embedding-model", default=DEFAULT_OPENAI_EMBEDDING_MODEL)
def ingest(filename: str, openai_embedding_model: str) -> None:
    """Read a book and extract subject-predicate-object triples."""
    with Timer(f"process document paragraphs [filename={filename}]"):
        process_document_paragraphs(filename, "text-embedding-3-large")


@main.command()
@click.option("--openai-embedding-model", default=DEFAULT_OPENAI_EMBEDDING_MODEL)
def query(openai_embedding_model: str) -> None:
    while True:
        query_str = input(colorize("--------------------\nEnter your query: "))
        with db_cursor() as cursor:
            query_embedding: Vector = get_text_embedding(
                cursor, query_str, openai_embedding_model
            ).vector
            gathered_triples = gather_related_triples(cursor, query_embedding)
            components = get_rag_prompt_components(cursor, gathered_triples, max_tokens=10_000)
            answer = get_completion(
                f"""
Context:

{'\n'.join(components)}

Instruction: Given the above text fragments, please answer the following query:

{query_str}""".strip()
            )
            print(answer)


def get_rag_prompt_components(
    cursor: Cursor,
    gathered_triples: List[GatheredTriple],
    max_tokens: int,
) -> List[str]:
    chunks: List[Tuple[UUID, int, str]] = []
    scored_paragraphs: List[Tuple[float, Paragraph]] = []
    for gathered_triple in gathered_triples:
        logging.info(
            " ".join(
                [
                    f"{gathered_triple.id}: ",
                    gathered_triple.subject,
                    gathered_triple.predicate,
                    gathered_triple.object,
                    gathered_triple.summary,
                ]
            )
        )
        if _paragraph := get_paragraph(cursor, gathered_triple.paragraph_uuid):
            scored_paragraphs.append((gathered_triple.distance, _paragraph))
        else:
            logging.error(
                f"[get_rag_prompt_components] could not find paragraph {gathered_triple.paragraph_uuid} in triple {gathered_triple.id}"
            )

    scored_paragraphs.sort(key=lambda p: (p[0], p[1].uuid))
    tokens = 0
    for _, paragraph in scored_paragraphs:
        chunks.append(
            (
                paragraph.document_uuid,
                paragraph.span_index_start,
                paragraph.text,
            )
        )
        tokens += num_tokens_from_string(paragraph.text)
        if tokens > max_tokens:
            # We're at the limit.
            break

    return [chunk[2] for chunk in sorted(chunks)]


@contextmanager
def db_cursor(row_factory: Any = dict_row, call_register_vector: bool = True) -> Iterator[Cursor]:
    with psycopg.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
    ) as conn:
        if call_register_vector:
            register_vector(conn)
        with conn.cursor(row_factory=row_factory) as cursor:
            yield cursor


@main.command()
def testdb() -> None:
    with db_cursor() as cursor:
        cursor.execute("SELECT 1, 2 UNION SELECT 3, 4")
        result = cursor.fetchone()
        print(result)


if __name__ == "__main__":
    main()
