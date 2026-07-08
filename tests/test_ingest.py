from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag.ingest import category_for, clean_markdown, discover, looks_unparseable


class IngestHelperTests(unittest.TestCase):
    def test_discover_skips_generated_dirs_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep = root / "guides" / "policy.md"
            skip = root / "parsed" / "policy.md"
            keep.parent.mkdir()
            skip.parent.mkdir()
            keep.write_text("keep", encoding="utf-8")
            skip.write_text("skip", encoding="utf-8")

            self.assertEqual(discover(root, include_parsed=False), [keep])
            self.assertEqual(discover(root, include_parsed=True), [keep, skip])

    def test_category_uses_first_relative_folder(self) -> None:
        root = Path("/knowledge")
        path = root / "benefits" / "plan.md"

        self.assertEqual(category_for(path, root=root, override=None), "benefits")
        self.assertEqual(category_for(path, root=root, override="manual"), "manual")

    def test_clean_markdown_removes_common_pdf_noise(self) -> None:
        text = clean_markdown(
            "Title\n"
            "Chapter 1 ................................ page 2\n"
            "iii\n"
            "Useful text abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz end\n"
        )

        self.assertEqual(text, "Title\nUseful text   end")

    def test_short_text_notes_are_allowed_but_short_pdfs_are_not(self) -> None:
        text = "This is a concise but useful markdown note with enough ordinary words."

        self.assertFalse(looks_unparseable(text, ext=".md"))
        self.assertTrue(looks_unparseable(text, ext=".pdf"))


if __name__ == "__main__":
    unittest.main()
