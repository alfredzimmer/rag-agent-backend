from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from pymilvus import DataType, MilvusClient, MilvusException, connections, db

logger = logging.getLogger("legacy-milvus-migration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate the legacy C++ ingestion staging collection.",
    )
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--source-db", default="default")
    parser.add_argument("--source-collection", default="ingestion_staging")
    parser.add_argument("--destination-uri", required=True)
    parser.add_argument("--destination-db", default="rag1")
    parser.add_argument("--destination-collection", default="HeaderInContentTrial")
    parser.add_argument("--global-scope", default="global")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/tmp/rag-agent-milvus-migration.json"),
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--expected-rows", type=int)
    return parser.parse_args()


def ensure_database(uri: str, database: str) -> None:
    alias = "legacy-migration-admin"
    connections.connect(alias=alias, uri=uri)
    try:
        if database not in db.list_database(using=alias):
            db.create_database(database, using=alias)
    finally:
        connections.disconnect(alias)


def create_destination_collection(
    client: MilvusClient,
    collection_name: str,
) -> None:
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(
        field_name="pk",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65_535,
    )
    schema.add_field(
        field_name="dense",
        datatype=DataType.FLOAT_VECTOR,
        dim=768,
    )

    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name="dense",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=indexes,
        consistency_level="Bounded",
    )


def load_checkpoint(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"batches": 0, "rows": 0}
    return json.loads(path.read_text())


def save_checkpoint(path: Path, batches: int, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps({"batches": batches, "rows": rows}))
    temporary.replace(path)


def wait_for_source(
    client: MilvusClient,
    collection_name: str,
    timeout_seconds: int = 300,
) -> None:
    client.load_collection(collection_name)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            client.query(
                collection_name=collection_name,
                filter="id >= 0",
                output_fields=["id"],
                limit=1,
            )
            return
        except MilvusException:
            if time.monotonic() >= deadline:
                raise
            time.sleep(5)


def document_id(job_id: str, filename: str) -> str:
    identity = f"legacy-cpp-ingestor:{job_id}:{filename}"
    return hashlib.sha256(identity.encode()).hexdigest()


def deduplication_key(row: dict[str, Any]) -> bytes:
    raw_metadata = row.get("metadata_json") or "{}"
    try:
        metadata = json.loads(raw_metadata)
        filename = str(
            metadata.get("name") or metadata.get("filename") or "unknown"
        )
        metadata_key = json.dumps(
            metadata,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except json.JSONDecodeError:
        filename = "unknown"
        metadata_key = raw_metadata
    identity = f"{filename}\0{metadata_key}\0{row.get('content') or ''}"
    return hashlib.sha256(identity.encode()).digest()


def transform(row: dict[str, Any], global_scope: str) -> dict[str, Any]:
    raw_metadata = row.get("metadata_json") or "{}"
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError:
        metadata = {"legacy_metadata_json": raw_metadata}

    filename = str(metadata.get("name") or metadata.get("filename") or "unknown")
    job_id = str(row.get("job_id") or "unknown")
    metadata.update(
        {
            "text": row["content"],
            "dense": row["vector"],
            "scope_id": global_scope,
            "document_id": document_id(job_id, filename),
            "job_id": job_id,
            "original_filename": filename,
            "legacy_id": str(row["id"]),
            "legacy_target_collection": row.get("target_collection", ""),
            "migration_source": "cpp-ingestor",
        }
    )
    return metadata


def migrate(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    source = MilvusClient(uri=args.source_uri, db_name=args.source_db)
    ensure_database(args.destination_uri, args.destination_db)
    destination = MilvusClient(
        uri=args.destination_uri,
        db_name=args.destination_db,
    )

    if args.reset:
        if destination.has_collection(args.destination_collection):
            destination.drop_collection(args.destination_collection)
        args.checkpoint.unlink(missing_ok=True)

    if not destination.has_collection(args.destination_collection):
        create_destination_collection(destination, args.destination_collection)

    state = load_checkpoint(args.checkpoint)
    skip_batches = state["batches"]
    migrated_rows = state["rows"]
    destination_rows = int(
        destination.get_collection_stats(args.destination_collection)["row_count"]
    )
    if destination_rows != migrated_rows:
        raise RuntimeError(
            "Destination row count does not match the checkpoint: "
            f"destination={destination_rows}, checkpoint={migrated_rows}. "
            "Use --reset to restart from an empty destination."
        )

    wait_for_source(source, args.source_collection)
    iterator = source.query_iterator(
        collection_name=args.source_collection,
        batch_size=args.batch_size,
        filter="status == 1",
        output_fields=[
            "id",
            "job_id",
            "content",
            "metadata_json",
            "target_collection",
            "vector",
        ],
    )

    batch_number = 0
    seen: set[bytes] = set()
    exhausted = False
    try:
        while True:
            batch = iterator.next()
            if not batch:
                exhausted = True
                break
            batch_number += 1

            unique_rows = []
            for row in batch:
                key = deduplication_key(row)
                if key in seen:
                    continue
                seen.add(key)
                unique_rows.append(row)

            if batch_number <= skip_batches:
                continue

            entities = [transform(row, args.global_scope) for row in unique_rows]
            if entities:
                destination.insert(
                    collection_name=args.destination_collection,
                    data=entities,
                )
            migrated_rows += len(entities)
            save_checkpoint(args.checkpoint, batch_number, migrated_rows)

            if batch_number % 20 == 0:
                logger.info(
                    "Migrated %s rows across %s batches",
                    f"{migrated_rows:,}",
                    batch_number,
                )
            if args.max_batches and batch_number >= args.max_batches:
                break
    finally:
        iterator.close()

    destination.flush(args.destination_collection)
    final_rows = int(
        destination.get_collection_stats(args.destination_collection)["row_count"]
    )
    if exhausted and args.expected_rows is not None and final_rows != args.expected_rows:
        raise RuntimeError(
            f"Expected {args.expected_rows} migrated rows, found {final_rows}"
        )
    logger.info(
        "Migration %s with %s destination rows",
        "completed" if exhausted else "paused",
        f"{final_rows:,}",
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    migrate(parse_args())


if __name__ == "__main__":
    main()
