import os
import sys
import time
from typing import Any, List, Optional, Tuple

import spacy  # type: ignore
from neo4j import GraphDatabase  # type: ignore


# Function to pre-cache the Spacy model
def init() -> None:
    os.system(".venv/bin/python -m spacy download en_core_web_sm")


class Timer:
    def __init__(self, name: Optional[str] = "block") -> None:
        self.name = name

    def __enter__(self) -> None:
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"{self.name} took {elapsed_time:.2f} seconds.")


def read_txt_file(filename: str) -> None:
    """Read a book and extract subject-predicate-object triples."""
    nlp = spacy.load("en_core_web_sm")
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()

    with Timer("run nlp on doc"):
        doc = nlp(text)
    with Timer("extract triples"):
        triples = extract_triples(doc)
    # store_triples(triples, driver)


def extract_triples(doc: Any) -> List[Tuple[str, str, str, str]]:
    triples = []
    for sentence in doc.sents:
        # print(f"Analyzing sentence ({sentence.text})")
        for token in sentence:
            assert token is not None
            if (
                token.dep_ in ("nsubj", "nsubjpass")
                and token.head
                and token.head.dep_
                in (
                    "ROOT",
                    "relcl",
                )
            ):
                subject = get_noun_phrase(token)
                predicate = get_enhanced_verb_phrase(token.head)
                obj = get_noun_phrase(find_obj(token.head))
                if subject and predicate and obj:
                    triples.append((subject, predicate, obj, sentence.text))
    return triples


def get_noun_phrase(token: Any) -> Optional[str]:
    if token is not None and token.dep_ in ("nsubj", "nsubjpass"):
        return " ".join([child.text for child in token.subtree])
    return None


def get_enhanced_verb_phrase(token: Any) -> str:
    if token.lemma_ == "boil":
        return "is angry at"
    # Add more custom rules here as needed
    return token.lemma_


def find_obj(head: Any) -> Any:
    for child in head.children:
        if child.dep_ in ("dobj", "attr", "prep"):
            return child
    return None


def store_triples(triples: List[Tuple[str, str, str, str]], driver: Any) -> None:
    with driver.session() as session:
        for subj, pred, obj, sent_text in triples:
            session.run(
                "MERGE (a:Entity {name: $subj}) "
                "MERGE (b:Entity {name: $obj}) "
                "MERGE (a)-[:RELATION {type: $pred, sentence: $sent_text}]->(b)",
                subj=subj,
                pred=pred,
                obj=obj,
                sent_text=sent_text,
            )


# Function for querying the graph database
def query() -> None:
    driver = GraphDatabase.driver("bolt://localhost:7687")

    while True:
        query_str = input("Enter your query: ")
        process_query(query_str, driver)


def process_query(query_str: str, driver: Any) -> None:
    # Placeholder for normalizing and transforming to Cypher query
    cypher_query = (
        f"MATCH (a)-[r]->(b) WHERE r.sentence CONTAINS '{query_str}' RETURN a, r, b"
    )

    with driver.session() as session:
        result = session.run(cypher_query)
        for record in result:
            print(record)


# Main function to handle subcommands
def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: grapple <init|read|query>")

    subcommand = sys.argv[1]

    if subcommand == "init":
        init()
    elif subcommand == "read":
        if len(sys.argv) < 3:
            sys.exit("Usage: grapple read <filename>")
        read_txt_file(sys.argv[2])
    elif subcommand == "query":
        query()
    else:
        print(f"Unknown subcommand: {subcommand}")


if __name__ == "__main__":
    main()
