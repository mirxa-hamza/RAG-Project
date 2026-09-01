"""
Back up the two things that cannot be rebuilt: uploaded PDFs and user accounts.

    python scripts/backup.py                     # -> backups/2026-09-01T1200/
    python scripts/backup.py --out D:/backups    # somewhere off this disk
    python scripts/backup.py --verify <folder>   # check an existing backup is readable

What is backed up, and why only this:

* `data/`      the uploaded PDFs. The ONLY copy anywhere. Losing it loses user documents
               permanently.
* MongoDB      the accounts. Small, but losing it means everyone signs up again - and the
               first new account then adopts every ownerless document.

What is NOT backed up, on purpose:

* `storage/`   the Chroma index and manifest. Derived data: `python scripts/ingest.py`
               rebuilds it from `data/` in minutes. Backing it up would multiply the size
               of every backup to protect something reproducible.

`mongodump` is used when it is on PATH, because it produces a consistent snapshot. When it
is not, the accounts are exported as JSON through pymongo instead - less efficient, same
information, and it means this script still works on a machine with only the Python client.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import DATA_DIR, MONGO_DB, MONGO_URI  # noqa: E402


def _human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def backup_documents(destination: Path) -> Path:
    """Tars data/ - one file is easier to copy off the machine than thousands."""
    archive = destination / "documents.tar.gz"
    if not DATA_DIR.is_dir():
        print(f"! {DATA_DIR} does not exist; nothing to archive")
        return archive

    with tarfile.open(archive, "w:gz") as tar:
        tar.add(DATA_DIR, arcname="data")
    print(f"  documents -> {archive.name} ({_human(archive.stat().st_size)})")
    return archive


def backup_accounts(destination: Path) -> Path:
    """mongodump if available, else a JSON export of the collections we own."""
    target = destination / "mongo"

    if shutil.which("mongodump"):
        subprocess.run(
            ["mongodump", f"--uri={MONGO_URI}", f"--db={MONGO_DB}", f"--out={target}"],
            check=True, capture_output=True,
        )
        print(f"  accounts  -> {target.name}/ (mongodump)")
        return target

    print("  mongodump not found; exporting accounts as JSON instead")
    from pymongo import MongoClient

    target.mkdir(parents=True, exist_ok=True)
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        database = client[MONGO_DB]
        for name in database.list_collection_names():
            rows = list(database[name].find({}))
            for row in rows:
                row["_id"] = str(row["_id"])       # ObjectId is not JSON
            (target / f"{name}.json").write_text(
                json.dumps(rows, indent=2, default=str), encoding="utf-8")
            print(f"  accounts  -> {name}.json ({len(rows)} document(s))")
    finally:
        client.close()
    return target


def verify(folder: Path) -> int:
    """
    Reads the backup back. A backup nobody has restored is a hope, not a backup.

    This does not restore anything - it checks the archive is readable, lists what is in it,
    and confirms the account export parses.
    """
    ok = True
    archive = folder / "documents.tar.gz"
    if archive.exists():
        with tarfile.open(archive) as tar:
            pdfs = [m.name for m in tar.getmembers() if m.name.lower().endswith(".pdf")]
        print(f"documents.tar.gz: readable, {len(pdfs)} PDF(s)")
    else:
        print("! documents.tar.gz missing")
        ok = False

    mongo = folder / "mongo"
    if mongo.is_dir():
        exports = list(mongo.rglob("*.json")) + list(mongo.rglob("*.bson"))
        print(f"mongo/: {len(exports)} export file(s)")
        if not exports:
            # An empty mongo/ folder means the account dump failed and nobody noticed -
            # exactly the failure a verify step exists to catch.
            print("! mongo/ is empty: the account backup did not run")
            ok = False
        for path in mongo.rglob("*.json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except ValueError as exc:
                print(f"! {path.name} is not valid JSON: {exc}")
                ok = False
    else:
        print("! mongo/ missing")
        ok = False

    print("\nBackup looks usable." if ok else "\nBackup is INCOMPLETE.")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "backups"),
                        help="where to write the backup (default: ./backups)")
    parser.add_argument("--verify", metavar="FOLDER",
                        help="check an existing backup instead of making one")
    parser.add_argument("--keep", type=int, default=7,
                        help="how many previous backups to keep (default: 7)")
    args = parser.parse_args()

    if args.verify:
        return verify(Path(args.verify))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
    destination = Path(args.out) / stamp
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Backing up to {destination}")

    backup_documents(destination)
    try:
        backup_accounts(destination)
    except Exception as exc:
        # A failed account dump must not throw away a good document archive.
        print(f"! accounts backup failed: {exc}")
        print("  the document archive above is still valid")
        return 1

    # Retention, so this can be run from a scheduled task without filling the disk.
    root = Path(args.out)
    existing = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    for old in existing[args.keep:]:
        shutil.rmtree(old, ignore_errors=True)
        print(f"  removed old backup {old.name}")

    print(f"\nDone. Verify it with:\n  python scripts/backup.py --verify {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
