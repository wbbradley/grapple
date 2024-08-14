import hashlib
import uuid


def sha256_to_uuid_v4(sha256_value: str) -> uuid.UUID:
    """Contort a SHA256 into a UUIDv4 for storage purposes."""
    hex_value = sha256_value[:32]
    uuid_bytes = bytearray.fromhex(hex_value)
    uuid_bytes[6] = (uuid_bytes[6] & 0x0F) | 0x40  # Set version to 4
    uuid_bytes[8] = (uuid_bytes[8] & 0x3F) | 0x80  # Set variant to 10
    return uuid.UUID(bytes=bytes(uuid_bytes))


def file_sha256(file_path: str) -> str:
    """Hash the contents of the given file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def str_sha256(content: str) -> str:
    """Hash the contents of the given string."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content.encode("utf-8"))
    return sha256_hash.hexdigest()


def str_to_uuid(content: str) -> uuid.UUID:
    return sha256_to_uuid_v4(str_sha256(content))
