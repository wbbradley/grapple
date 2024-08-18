import logging
import os
import subprocess
from typing import Optional

from openai import OpenAI

from grapple.metrics import metrics_count, metrics_timer

# GRAPPLE_OPENAI_KEY_CMD: must be a command that prints your OpenAI API key to stdout.
openai_client = OpenAI(
    api_key=subprocess.check_output(
        os.environ.get("GRAPPLE_OPENAI_KEY_CMD", "pass openai-api-key"), shell=True
    )
    .decode("utf-8")
    .strip()
)


def get_completion(prompt: str, model: str = "gpt-4o-2024-08-06") -> Optional[str]:
    metrics_count(f"openai.completions.{model}")
    with metrics_timer("openai.request.completions"):
        completion = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    completion_text = completion.choices[0].message.content
    logging.info(f"[get_completion] log\nPrompt:\n\n{prompt}\n\n Completion:\n\n{completion_text}")
    return completion_text
