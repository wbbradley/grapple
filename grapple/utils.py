import hashlib
import uuid

import tiktoken


def str_to_uuid(content: str) -> uuid.UUID:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content.encode("utf-8"))
    hash_bytearray = bytearray(sha256_hash.digest()[:16])
    hash_bytearray[6] = (hash_bytearray[6] & 0x0F) | 0x40  # Set version to 4
    hash_bytearray[8] = (hash_bytearray[8] & 0x3F) | 0x80  # Set variant to 10
    return uuid.UUID(bytes=bytes(hash_bytearray))


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
