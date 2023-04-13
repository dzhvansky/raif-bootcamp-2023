import pathlib

import pytest
from fastapi.testclient import TestClient

from painting_estimation.api.main import APP, INSTRUMENTATOR


@pytest.fixture
def test_image() -> bytes:
    return (pathlib.Path(__file__).parent / "fixtures" / "test_image.png").read_bytes()


@pytest.fixture
def test_client() -> TestClient:
    INSTRUMENTATOR.expose(APP)
    return TestClient(APP)
