"""The seeded launcher copy: LAB_RECIPE_SEED_LEAF picks the seed leaf, refuses a bad one, and unset keeps the stock origin."""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("lab_serve_seeded", sys.argv.pop())
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)
SOURCE = {"Config": {"Env": ["PT_HPU_RECIPE_CACHE_CONFIG=/recipe_cache/unit-cache,false,8192"]}}


class SeedOrigin(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        lab.RECIPE_ROOT = self.root
        os.environ.pop(lab.SEED_LEAF_ENV, None)

    def test_unset_keeps_the_source_unit_cache(self) -> None:
        self.assertEqual(lab.recipe_seed_origin(SOURCE), self.root / "unit-cache")

    def test_a_served_leaf_is_used(self) -> None:
        leaf = self.root / "claude-flash-good"
        leaf.mkdir()
        (leaf / "1.recipe").write_bytes(b"x")
        os.environ[lab.SEED_LEAF_ENV] = str(leaf)
        self.assertEqual(lab.recipe_seed_origin(SOURCE), leaf)

    def test_a_leaf_without_recipes_or_outside_the_root_is_refused(self) -> None:
        empty = self.root / "claude-flash-empty"
        empty.mkdir()
        for bad in (str(empty), "/tmp", str(self.root / "missing")):
            os.environ[lab.SEED_LEAF_ENV] = bad
            with self.assertRaises(SystemExit):
                lab.recipe_seed_origin(SOURCE)

    def test_seeding_copies_the_named_leaf(self) -> None:
        leaf = self.root / "claude-flash-good"
        leaf.mkdir()
        (leaf / "1.recipe").write_bytes(b"x")
        os.environ[lab.SEED_LEAF_ENV] = str(leaf)
        new = self.root / "claude-flash-new"
        lab.seed_recipe_leaf(SOURCE, new, fresh_cache=False)
        self.assertEqual((new / "1.recipe").read_bytes(), b"x")


class FreshCacheVersusSeed(unittest.TestCase):
    """LAB_RECIPE_SEED_LEAF wins over --fresh-cache; unset, --fresh-cache still means an empty leaf."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        lab.RECIPE_ROOT = self.root
        os.environ.pop(lab.SEED_LEAF_ENV, None)
        self.seed = self.root / "claude-flash-good"
        self.seed.mkdir()
        (self.seed / "1.recipe").write_bytes(b"x")
        (self.seed / "notes.txt").write_bytes(b"not a recipe")

    def seed_quietly(self, new: Path, fresh_cache: bool) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lab.seed_recipe_leaf(SOURCE, new, fresh_cache=fresh_cache)
        return out.getvalue()

    def test_seed_env_wins_over_fresh_cache(self) -> None:
        os.environ[lab.SEED_LEAF_ENV] = str(self.seed)
        new = self.root / "claude-flash-new"
        self.seed_quietly(new, fresh_cache=True)
        self.assertEqual((new / "1.recipe").read_bytes(), b"x")

    def test_fresh_cache_without_seed_env_is_an_empty_leaf(self) -> None:
        new = self.root / "claude-flash-new"
        self.assertEqual(self.seed_quietly(new, fresh_cache=True), "")
        self.assertTrue(new.is_dir())
        self.assertEqual(list(new.iterdir()), [])

    def test_a_seed_that_is_the_new_leaf_is_refused_and_left_intact(self) -> None:
        os.environ[lab.SEED_LEAF_ENV] = str(self.seed)
        with self.assertRaises(SystemExit):
            self.seed_quietly(self.seed, fresh_cache=True)
        self.assertEqual((self.seed / "1.recipe").read_bytes(), b"x")

    def test_a_bad_seed_is_refused_with_fresh_cache_and_leaves_no_leaf(self) -> None:
        os.environ[lab.SEED_LEAF_ENV] = str(self.root / "missing")
        new = self.root / "claude-flash-new"
        with self.assertRaises(SystemExit):
            self.seed_quietly(new, fresh_cache=True)
        self.assertFalse(new.exists())

    def test_seeding_prints_one_leaf_seeded_line_counting_recipes(self) -> None:
        os.environ[lab.SEED_LEAF_ENV] = str(self.seed)
        new = self.root / "claude-flash-new"
        out = self.seed_quietly(new, fresh_cache=True)
        self.assertEqual(out, f"LEAF_SEEDED origin={self.seed} recipes=1\n")


if __name__ == "__main__":
    unittest.main()
