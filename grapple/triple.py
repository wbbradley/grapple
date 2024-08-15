from typing import List

from pydantic import BaseModel

from grapple.metrics import metrics_count
from grapple.openai import openai_client


class SemanticTriple(BaseModel):
    subject: str
    predicate: str
    object: str
    summary: str


class SemanticTriples(BaseModel):
    triples: List[SemanticTriple]


def get_triples(paragraph: str) -> List[SemanticTriple]:
    model = "gpt-4o-2024-08-06"
    metrics_count(
        "beta.chat.completions.parse",
        tags={"provider": "openai", "model": model},
    )
    completion = openai_client.beta.chat.completions.parse(
        model=model,
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
