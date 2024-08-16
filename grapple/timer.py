import logging
import time
from typing import Optional


class Timer:
    def __init__(self, name: Optional[str] = "block") -> None:
        self.name = name

    def __enter__(self) -> None:
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        logging.info(f"{self.name} took {elapsed_time:.2f} seconds.")
