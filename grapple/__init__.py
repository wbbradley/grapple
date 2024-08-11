import os
import sys
from typing import Any, List, Tuple

import spacy  # type: ignore
from neo4j import GraphDatabase  # type: ignore


# Function to pre-cache the Spacy model
def init() -> None:
    os.system(".venv/bin/pip install spacy")
    os.system(".venv/bin/python -m spacy download en_core_web_sm")


# Function to read a book and extract subject-predicate-object triples
def read(filename: str) -> None:
    nlp = spacy.load("en_core_web_sm")
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()

    doc = nlp(text)
    triples = extract_triples(doc)
    store_triples(triples, driver)


def extract_triples(doc: Any) -> List[Tuple[str, str, str, str]]:
    triples = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ("nsubj", "nsubjpass") and token.head.dep_ in (
                "ROOT",
                "relcl",
            ):
                subject = token
                predicate = token.head
                for child in token.head.children:
                    if child.dep_ in ("dobj", "attr", "prep"):
                        obj = child
                        triples.append(
                            (subject.text, predicate.text, obj.text, sent.text)
                        )
    return triples


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
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

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
        sys.exit("Usage: grapple <subcommand>")

    subcommand = sys.argv[1]

    if subcommand == "init":
        init()
    elif subcommand == "read":
        if len(sys.argv) < 3:
            print("Usage: grapple read <filename>")
            sys.exit(1)
        read(sys.argv[2])
    elif subcommand == "query":
        query()
    else:
        print(f"Unknown subcommand: {subcommand}")


if __name__ == "__main__":
    main()
