import pytest
from steamship.data import TagValue

from cohere.api_spec import MODELS

__copyright__ = "Steamship"
__license__ = "MIT"

from util import cohere

from cohere.client import CohereEmbeddingClient

TEST_DATA = []
for m in MODELS:
    d = MODELS[m]
    TEST_DATA.append((m, d))


@pytest.mark.usefixtures("cohere")
@pytest.mark.parametrize("model,dimensions", TEST_DATA)
def test_embed(cohere: CohereEmbeddingClient, model: str, dimensions: int):
    texts = ["apple", "orange", "banana", "kiwi", "blueberry", "car"]
    res = cohere.request(model, texts)
    assert len(res) == len(texts)
    for tags in res:
        assert len(tags) == 1
        tag = tags[0]
        assert tag.value is not None
        assert tag.value[TagValue.VECTOR_VALUE] is not None
        assert len(tag.value[TagValue.VECTOR_VALUE]) == dimensions
