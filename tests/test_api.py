import pytest
from fastapi.testclient import TestClient

from src.predict import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")


@pytest.fixture(scope="module")
def client():
    from service.app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_angry_review_routes_to_support(client):
    response = client.post(
        "/review",
        json={"text": "Terrible, awful experience — the worst. Charged twice for a broken stream. I want a REFUND, cancelling today!!!"},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["p_negative"] > 0.5
    assert body["route"] == "support_urgent"


def test_positive_review_routes_to_analytics(client):
    response = client.post("/review", json={"text": "Fantastic movie, great acting, wonderful soundtrack. Loved it."})
    body = response.json()
    assert body["p_negative"] < 0.5
    assert body["route"] == "analytics"
