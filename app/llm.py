import json
import time
from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, LLM_MODEL, EMBED_MODEL, EMBED_DIM

client = genai.Client(api_key=GEMINI_API_KEY)


def _retry(fn, attempts=3):
    """Retry on 429 rate limits with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if "429" not in str(exc) or "PerDay" in str(exc) or i == attempts - 1:
                raise  # daily quota: waiting won't help
            wait = 10 * 2 ** i
            print(f"Rate limited, retrying in {wait}s...")
            time.sleep(wait)


def _embed_batch(chunk, task):
    res = _retry(lambda: client.models.embed_content(
        model=EMBED_MODEL,
        contents=chunk,
        config=types.EmbedContentConfig(task_type=task,
                                        output_dimensionality=EMBED_DIM),
    ))
    return [e.values for e in res.embeddings]


def embed(texts, task="RETRIEVAL_DOCUMENT", batch=50):
    """Embed a list of strings. Use task='RETRIEVAL_QUERY' for questions."""
    vectors = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        vecs = _embed_batch(chunk, task)
        if len(vecs) != len(chunk):
            # some models return one aggregated vector per request:
            # fall back to one text per request
            vecs = [v for t in chunk for v in _embed_batch([t], task)]
        if len(vecs) != len(chunk):
            raise RuntimeError(
                f"Embedding model returned {len(vecs)} vectors for {len(chunk)} texts")
        vectors.extend(vecs)
        time.sleep(1)
    return vectors


def generate(prompt, json_mode=False):
    cfg = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    resp = _retry(lambda: client.models.generate_content(
        model=LLM_MODEL, contents=prompt, config=cfg))
    if not json_mode:
        return resp.text
    out = json.loads(resp.text)
    # some models wrap the JSON object in a single-item list
    while isinstance(out, list) and len(out) == 1:
        out = out[0]
    return out
