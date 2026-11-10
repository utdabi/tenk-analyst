"""Shared test setup. pytest runs this file before the tests."""
import json
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so that "import backend" works in the tests
load_dotenv(ROOT / ".env")


@pytest.fixture(scope="session")
def test_data() -> dict:
    """The contents of test_data/questions.json."""
    return json.loads((ROOT / "test_data" / "questions.json").read_text(encoding="utf-8"))
