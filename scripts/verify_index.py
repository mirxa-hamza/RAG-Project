"""
Check the index against the manifest and the filesystem, and report what disagrees.

    python scripts/verify_index.py            # report only
    python scripts/verify_index.py --fix      # repair what can be repaired safely

`prune_deleted()` already reconciles the manifest against disk on every ingest. Nothing
reconciles ChromaDB against the manifest, so three kinds of drift can survive indefinitely:

* **Orphan chunks** - a document's chunks are in Chroma with no manifest entry. Left by a
  run killed between storing chunks and writing the entry. They are still searchable, so an
  answer can quote a document that /stats says does not exist.
* **Ownerless chunks** - stored before accounts existed and never adopted. Invisible to
  every filter, so they occupy space and can never be retrieved or deleted through the app.
* **Missing chunks** - a manifest entry whose document has no chunks at all, usually a
  half-finished ingest. The document is listed in the UI and answers nothing.

`--fix` deletes orphans and re-ingests documents with missing chunks. It never deletes a
manifest entry whose file still exists, and it never touches ownerless chunks: adopting them
is a decision about who owns what, which belongs to a person, not a script.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import DATA_DIR  # noqa: E402
from src.services import ingestion, manifest, vectorstore  # noqa: E402


def collect():
    """Everything each of the three sources knows about, keyed by document name."""
    chunks_by_source = defaultdict(int)
    owners = {}
    ownerless = Counter()

    for chunk in vectorstore.all_chunks():
        source = chunk.get("source")
        chunks_by_source[source] += 1
        if chunk.get("user_id"):
            owners[source] = chunk["user_id"]
        else:
            ownerless[source] += 1

    manifest_entries = {name: manifest.get(name) for name in manifest.sources()}
    on_disk = set(ingestion._pdf_filenames())
    return chunks_by_source, owners, ownerless, manifest_entries, on_disk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true",
                        help="delete orphan chunks and re-ingest documents with none")
    args = parser.parse_args()

    chunks, owners, ownerless, entries, on_disk = collect()

    orphans = sorted(set(chunks) - set(entries))
    missing_chunks = sorted(name for name in entries if chunks.get(name, 0) == 0)
    missing_files = sorted(name for name in entries if name not in on_disk)
    unowned = sorted(ownerless)

    print(f"Chroma:    {sum(chunks.values()):,} chunks across {len(chunks)} document(s)")
    print(f"Manifest:  {len(entries)} document(s)")
    print(f"Data dir:  {len(on_disk)} PDF(s) under {DATA_DIR}")
    print()

    def report(title, names, note):
        if not names:
            print(f"OK   {title}: none")
            return
        print(f"WARN {title}: {len(names)}")
        for name in names[:10]:
            print(f"       {name}")
        if len(names) > 10:
            print(f"       ... and {len(names) - 10} more")
        print(f"       {note}")

    report("orphan chunks (in Chroma, not in the manifest)", orphans,
           "searchable but invisible in /stats - --fix deletes them")
    report("documents with no chunks", missing_chunks,
           "listed in the UI, answers nothing - --fix re-ingests them")
    report("manifest entries whose file is gone", missing_files,
           "the next ingest prunes these automatically")
    report("ownerless chunks", unowned,
           "invisible to every user; sign up (first account adopts them) or delete by hand")

    if not args.fix:
        problems = bool(orphans or missing_chunks)
        print("\nRun with --fix to repair." if problems else "\nIndex is consistent.")
        return 1 if problems else 0

    print()
    for name in orphans:
        vectorstore.delete_source(name)
        print(f"deleted orphan chunks for {name}")

    for name in missing_chunks:
        if name not in on_disk:
            print(f"skipped {name}: its file is gone, the next ingest will prune it")
            continue
        # Drop the stale entry so the ingester treats it as new rather than "unchanged".
        manifest.remove(name)
        result = ingestion.ingest_one(name, reason="new")
        print(f"re-ingested {name}: {result['status']}")

    print("\nDone. Re-run without --fix to confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
