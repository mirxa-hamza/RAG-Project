"""
Build the index from the command line, without starting the API.

    python scripts/ingest.py            # ingest new / changed PDFs
    python scripts/ingest.py --force    # re-ingest everything from scratch
    python scripts/ingest.py --status   # show what's currently in the store

Doing the heavy work here rather than at server startup is the point: the API then boots
in seconds and simply reads an index that already exists.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.services import manifest, vectorstore  # noqa: E402
from src.core.config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL  # noqa: E402
from src.services.ingestion import ingest_data_folder  # noqa: E402
from src.core.logging import get_logger  # noqa: E402

log = get_logger("ingest-cli")


def show_status() -> None:
    summary = manifest.summary()
    print(f"Data folder : {DATA_DIR}")
    print(f"Vector store: {CHROMA_DIR}")
    print(f"Model       : {EMBEDDING_MODEL}")
    print(f"Chunks      : {vectorstore.count()}")
    if not summary["documents"]:
        print("Documents   : (none ingested yet)")
        return
    print("Documents   :")
    for doc in summary["documents"]:
        print(
            f"  - {doc['filename']}  "
            f"({doc.get('pages', '?')} pages, {doc.get('chunks', '?')} chunks, "
            f"ingested {doc.get('ingested_at', '?')})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs from the data folder.")
    parser.add_argument("--force", action="store_true", help="re-ingest every PDF, even if unchanged")
    parser.add_argument("--status", action="store_true", help="show current store contents and exit")
    args = parser.parse_args()

    if args.status:
        show_status()
        return 0

    if args.force:
        log.info("--force: clearing the vector store before re-ingesting.")
        vectorstore.reset_collection()
        manifest.clear()

    results = ingest_data_folder(force=args.force)
    if not results:
        log.warning("No PDFs found in %s", DATA_DIR)
        return 0

    for result in results:
        print(result)
    print(f"\nDone. {vectorstore.count()} chunks in the store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
