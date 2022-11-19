from enum import Enum
from typing import List, Optional

from pydantic import BaseModel
from steamship.data import TagKind, TagValue
from steamship.data.tags import Tag

from cohere.api_spec import validate_model
from cohere.request_utils import concurrent_json_posts


class OpenAIObject(str, Enum):
    LIST = 'list'
    EMBEDDING = 'embedding'

class CohereEmbedding(BaseModel):
    object: OpenAIObject # 'embedding'
    index: int
    embedding: List[float]


class CohereEmbeddingList(BaseModel):
    id: str
    texts: Optional[List[str]] = None
    embeddings: List[List[float]]

    def to_tag(self, embedding: List[float], model: str) -> Tag.CreateRequest:
        return Tag.CreateRequest(
            kind=TagKind.EMBEDDING,
            name=model,
            value={
                "service": "cohere",
                TagValue.VECTOR_VALUE: embedding
            },
        )

    def to_tags(self, model: str) -> List[Tag.CreateRequest]:
        return [self.to_tag(embedding, model) for embedding in self.embeddings]


class CohereEmbeddingClient:
    URL = "https://api.cohere.ai/embed"

    def __init__(self, key: str):
        self.key = key

    def request(
        self, model: str, inputs: List[str], truncate: str = "NONE", **kwargs
    ) -> List[List[Tag.CreateRequest]]:
        """Performs an OpenAI request. Throw a SteamshipError in the event of error or empty response."""

        validate_model(model, truncate)

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

        def items_to_body(items: List[str]):
            return {
                "model": model,
                "texts": items,
                "truncate": truncate
            }

        responses = concurrent_json_posts(self.URL, headers, inputs, 6, items_to_body, "cohere")

        ret: List[List[Tag.CreateRequest]] = []
        for response in responses:
            obj = CohereEmbeddingList.parse_obj(response)
            for embedding_tag in obj.to_tags(model):
                ret.append([embedding_tag])

        return ret
