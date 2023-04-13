import pathlib

import pytest


@pytest.fixture
def test_image() -> bytes:
    return (pathlib.Path(__file__).parent / "fixtures" / "test_image.png").read_bytes()
