import hashlib
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector import vector_store

ASSETS_DIR = Path("assets")
MANIFEST_PATH = ASSETS_DIR / ".index_manifest.json"
BATCH_SIZE = 20

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def batch(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def chunk_ids(filename: str, num_chunks: int) -> list[str]:
    return [f"{filename}::{i}" for i in range(num_chunks)]


def delete_chunks(filename: str, num_chunks: int) -> None:
    ids = chunk_ids(filename, num_chunks)
    if ids:
        vector_store.delete(ids=ids)


def index_file(path: Path) -> int:
    raw_documents = PyPDFLoader(str(path)).load()
    documents = text_splitter.split_documents(raw_documents)
    ids = chunk_ids(path.name, len(documents))

    for doc_batch, id_batch in zip(batch(documents, BATCH_SIZE), batch(ids, BATCH_SIZE)):
        vector_store.add_documents(documents=doc_batch, ids=id_batch)

    return len(documents)


def main() -> None:
    manifest = load_manifest()
    current_files = {p.name: p for p in ASSETS_DIR.glob("*.pdf")}

    added = updated = removed = unchanged = 0

    for filename, path in current_files.items():
        new_hash = file_hash(path)
        entry = manifest.get(filename)

        if entry is None:
            print(f"Adding {filename}...")
            num_chunks = index_file(path)
            manifest[filename] = {"hash": new_hash, "num_chunks": num_chunks}
            added += 1
        elif entry["hash"] != new_hash:
            print(f"Updating {filename}...")
            delete_chunks(filename, entry["num_chunks"])
            num_chunks = index_file(path)
            manifest[filename] = {"hash": new_hash, "num_chunks": num_chunks}
            updated += 1
        else:
            unchanged += 1

    for filename in list(manifest.keys()):
        if filename not in current_files:
            print(f"Deleting {filename}...")
            delete_chunks(filename, manifest[filename]["num_chunks"])
            del manifest[filename]
            removed += 1

    save_manifest(manifest)

    print(f"{added} added, {updated} updated, {removed} deleted, {unchanged} unchanged")


if __name__ == "__main__":
    main()
