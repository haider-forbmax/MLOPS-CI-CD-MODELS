import tiktoken
from config import Config
from fastapi import Header, HTTPException

INPUT_TOKENS_LIMIT = 4000  

config = Config()
def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Counts the number of tokens in the given text for the specified model.
    """
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # fallback

    return len(enc.encode(text))


def truncate_to_token_limit(text: str, INPUT_TOKENS_LIMIT: int, model_name: str = "gpt-3.5-turbo") -> str:
    """
    Truncates the text to fit within the max token limit.
    """
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)
    if len(tokens) > INPUT_TOKENS_LIMIT:
        truncated_text = enc.decode(tokens[:INPUT_TOKENS_LIMIT])
        # logger.warning(f"Input truncated from {len(tokens)} → {INPUT_TOKENS_LIMIT} tokens.")
        return truncated_text
    return text


def verify_api_key(x_api_key: str = Header(...)):
    """
    Simple header-based API key verification.
    """
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True