import hashlib
from typing import Generator


def file_sha256_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def string_to_sha256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def split_on_pause(text: str) -> Generator[str, None, None]:
    text = text.replace("...", ", ")

    for char in [".", "!", "?"]:
        text = text.replace(char, char + "\n")

    chunks = text.split("\n")
    for chunk in map(lambda x: x.strip(), chunks):
        if chunk == "":
            continue

        if len(chunk) < 3:
            continue

        yield chunk[0].upper() + chunk[1:]


def join_if_short(iterator: Generator[str, None, None], cut_length: int, cut_flex: float) -> Generator[str, None, None]:
    buffer = ""

    for text in iterator:
        if (len(text) > cut_length * cut_flex) and buffer:
            yield buffer
            buffer = ""

        buffer += text + " "

        if len(buffer) >= cut_length:
            yield buffer.strip()
            buffer = ""

    if buffer:
        yield buffer.strip()


def spacer(iterator, space):
    return [e for i in iterator for e in (i, space)][:-1]


def split_on_nothing(text: str) -> Generator[str, None, None]:
    yield text
