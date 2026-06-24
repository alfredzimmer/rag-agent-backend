from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.migrate_legacy_milvus import (
    deduplication_key,
    load_checkpoint,
    save_checkpoint,
    transform,
)


class LegacyMilvusMigrationTests(unittest.TestCase):
    def test_deduplication_ignores_staging_identity_fields(self) -> None:
        first = {
            "id": 1,
            "job_id": "job-one",
            "content": "same chunk",
            "metadata_json": '{"name":"source.pdf","page":2}',
            "target_collection": "arbitrary_collection",
        }
        duplicate = {
            **first,
            "id": 2,
            "job_id": "job-two",
            "target_collection": "HeaderInContentTrial",
        }

        self.assertEqual(
            deduplication_key(first),
            deduplication_key(duplicate),
        )

    def test_transform_maps_legacy_row_to_global_scope(self) -> None:
        row = {
            "id": 42,
            "job_id": "job-source",
            "content": "chunk text",
            "metadata_json": '{"name":"source.pdf","page":3}',
            "target_collection": "arbitrary_collection",
            "vector": [0.1, 0.2],
        }

        entity = transform(row, "global")

        self.assertEqual(entity["text"], "chunk text")
        self.assertEqual(entity["dense"], [0.1, 0.2])
        self.assertEqual(entity["scope_id"], "global")
        self.assertEqual(entity["original_filename"], "source.pdf")
        self.assertEqual(entity["legacy_id"], "42")
        self.assertEqual(entity["legacy_target_collection"], "arbitrary_collection")
        self.assertEqual(len(entity["document_id"]), 64)

    def test_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"

            save_checkpoint(checkpoint, batches=12, rows=345)

            self.assertEqual(
                load_checkpoint(checkpoint),
                {"batches": 12, "rows": 345},
            )


if __name__ == "__main__":
    unittest.main()
