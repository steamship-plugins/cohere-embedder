"""Collection of object specifications used to communicate with the NLPCloud API."""

from steamship import SteamshipError

MODELS = {
    "small": 1024,
    "medium": 2048,
    "large": 4096
}

TRUNCATE_TYPES = ["NONE", "LEFT", "RIGHT"]



def validate_model(model: str, truncate: str):
    # We know from docs.nlpcloud.com that only certain task<>model pairings are valid.
    if model not in MODELS:
        raise SteamshipError(
            message=f"Model {model} is not supported by this plugin.. " +
            f"Valid models for this task are: {[m.value for m in MODELS]}."
        )
    if truncate not in TRUNCATE_TYPES:
        raise SteamshipError(
            message=f"Truncate type {truncate} is not supported by this plugin.. " +
            f"Valid truncate types for this task are: {TRUNCATE_TYPES}."
        )
