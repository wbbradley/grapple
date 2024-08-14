import os
import subprocess

from openai import OpenAI

# GRAPPLE_OPENAI_KEY_CMD: must be a command that prints your OpenAI API key to stdout.
openai_client = OpenAI(
    api_key=subprocess.check_output(
        os.environ.get("GRAPPLE_OPENAI_KEY_CMD", "pass openai-api-key"), shell=True
    )
    .decode("utf-8")
    .strip()
)
