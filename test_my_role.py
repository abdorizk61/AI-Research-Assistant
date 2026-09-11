#!/usr/bin/env python3
"""
test_my_role.py
===============
Standalone local verification for the AI Research Assistant responsibilities:

    1. RAG pipeline (chunking + ChromaDB + Ollama embeddings)
    2. Reliability (weak-match fallback + URL scrape try/except)
    3. Local Ollama connectivity

Does not modify app.py / rag_manager.py. Uses an ephemeral Chroma collection
so it will not pollute the persistent knowledge base.

Prerequisites:
    - Ollama running on http://localhost:11434
    - Embedding model pulled:  ollama pull nomic-embed-text
    - (Optional) a chat model pulled, e.g. llama3 / llama3.2 / phi3

Run from this folder:
    python test_my_role.py
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from rag_manager import RAGManager

# ---------------------------------------------------------------------------
# Constants matching the project’s intended reliability contract
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
FALLBACK_MESSAGE = (
    "I don't have enough information to answer this based on the provided sources."
)

# Chroma default metric is L2. With nomic-embed-text this threshold is a
# conservative “not similar enough” cutoff; the test also asserts that an
# unrelated query scores *worse* than a relevant one.
DISTANCE_THRESHOLD = 1.15

FAKE_TIMEOUT_URL = "http://this-is-a-fake-timeout-url.com"
SCRAPE_TIMEOUT_SECONDS = 5

DUMMY_DOCUMENT = """
Solid-state sodium-ion grid batteries for desert microgrids.

The Qattara Ridge 2024 field trial showed that ceramic-electrolyte sodium-ion
packs retained 94% capacity after 800 cycles at 48 C ambient temperature.
Researchers named the cathode formulation 'Natron-Ridge-7'. Grid-scale adoption
was limited by ceramic electrolyte cracking during thermal cycling, not by
raw sodium cost. A polymer interlayer reduced micro-cracks by 37%.
"""
RELEVANT_QUERY = "What limited adoption of Natron-Ridge-7 sodium-ion packs?"
IRRELEVANT_QUERY = (
    "What is the traditional recipe and oven temperature for Japanese souffle pancakes?"
)


# ---------------------------------------------------------------------------
# Tiny assertion helpers (readable terminal output, no pytest required)
# ---------------------------------------------------------------------------
_PASSED = 0
_FAILED = 0


def check(condition: bool, label: str, detail: str = "") -> None:
    global _PASSED, _FAILED
    if condition:
        _PASSED += 1
        extra = f"  ({detail})" if detail else ""
        print(f"  [PASS] {label}{extra}")
    else:
        _FAILED += 1
        extra = f"  ({detail})" if detail else ""
        print(f"  [FAIL] {label}{extra}")


def decide_answer(
    retrieved: List[Dict[str, Any]],
    in_domain_distance: Optional[float] = None,
) -> str:
    """
    Generation-step gate that rag_manager.py deliberately left out of scope.
    Weak / missing matches must never be sent to the LLM as if they were facts.

    A hit is rejected when:
      - nothing was retrieved, or
      - L2 distance exceeds DISTANCE_THRESHOLD, or
      - it is clearly worse than a known in-domain hit (relative gap).
    """
    if not retrieved:
        return FALLBACK_MESSAGE
    best = min(float(hit["distance"]) for hit in retrieved)
    if best > DISTANCE_THRESHOLD:
        return FALLBACK_MESSAGE
    if in_domain_distance is not None:
        gap = best - in_domain_distance
        if gap > 0.15 and best > in_domain_distance * 1.35:
            return FALLBACK_MESSAGE
    return "CONTEXT_OK"


def fetch_url_safely(url: str, timeout: float = SCRAPE_TIMEOUT_SECONDS) -> str:
    """
    Reliability wrapper for scraping. There is no production scrape function
    in this workspace yet (app.py is not present; the UI is still a skeleton),
    so this is the contract the UI should call: timeout + try/except, no crash.
    """
    try:
        import requests

        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "AI-Research-Assistant-test/1.0"},
            allow_redirects=True,
        )
        response.raise_for_status()
        return response.text
    except Exception as exc:  # ConnectionError, Timeout, HTTPError, DNS, etc.
        print(f"      scrape caught {type(exc).__name__}: {exc}")
        return ""


def ollama_get(path: str, timeout: float = 8) -> Any:
    req = Request(f"{OLLAMA_BASE_URL}{path}", method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_post(path: str, payload: dict, timeout: float = 60) -> Any:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{OLLAMA_BASE_URL}{path}",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pick_chat_model(tags_payload: dict) -> Optional[str]:
    names = [m.get("name", "") for m in tags_payload.get("models", [])]
    preferred = (
        "llama3.2",
        "llama3.1",
        "llama3",
        "phi3",
        "mistral",
        "qwen2.5",
        "gemma2",
        "gemma",
    )
    for stem in preferred:
        for name in names:
            if name.startswith(stem):
                return name
    for name in names:
        if name and EMBEDDING_MODEL not in name:
            return name
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_rag_pipeline() -> RAGManager:
    print("\n=== 1. RAG Pipeline ===")
    rag = RAGManager(
        collection_name="role_verification_ephemeral",
        embedding_model=EMBEDDING_MODEL,
        ollama_base_url=OLLAMA_BASE_URL,
        use_persistent_storage=False,
        chunk_size=400,
        chunk_overlap=50,
    )
    check(rag.collection.count() == 0, "Ephemeral collection starts empty")

    n_chunks = rag.add_documents(
        DUMMY_DOCUMENT, source_url="https://example.com/natron-ridge-7"
    )
    check(n_chunks >= 1, "Dummy document was chunked and stored", f"chunks={n_chunks}")
    check(
        rag.get_indexed_source_count() == n_chunks,
        "ChromaDB reports the same chunk count",
        f"count={rag.get_indexed_source_count()}",
    )

    hits = rag.retrieve_context(RELEVANT_QUERY, k=3)
    check(len(hits) >= 1, "Relevant query returned at least one chunk", f"k={len(hits)}")

    if hits:
        best = hits[0]
        check("text" in best and "source" in best and "distance" in best, "Hit schema is complete")
        check(
            "Natron-Ridge-7" in best["text"] or "sodium-ion" in best["text"].lower(),
            "Top hit text is from the dummy document",
        )
        check(
            best["source"] == "https://example.com/natron-ridge-7",
            "Top hit metadata preserves source URL",
        )
        check(
            float(best["distance"]) <= DISTANCE_THRESHOLD,
            "Relevant query is within the similarity threshold",
            f"distance={float(best['distance']):.4f} <= {DISTANCE_THRESHOLD}",
        )
        print(f"      top relevant distance = {float(best['distance']):.4f}")
        print(f"      top relevant preview  = {best['text'][:120]!r}...")

    return rag


def test_reliability(rag: RAGManager) -> None:
    print("\n=== 2. Reliability & Explicit Fallback ===")

    relevant_hits = rag.retrieve_context(RELEVANT_QUERY, k=3)
    weak_hits = rag.retrieve_context(IRRELEVANT_QUERY, k=3)

    check(len(weak_hits) >= 1, "Unrelated query still returns raw nearest neighbors (RAG does not filter)")

    rel_d = min(float(h["distance"]) for h in relevant_hits)
    weak_d = min(float(h["distance"]) for h in weak_hits)
    print(f"      relevant min distance   = {rel_d:.4f}")
    print(f"      unrelated min distance  = {weak_d:.4f}")
    check(weak_d > rel_d, "Unrelated query has a worse (higher) distance than the relevant query")

    relevant_answer = decide_answer(relevant_hits)
    weak_answer = decide_answer(weak_hits, in_domain_distance=rel_d)

    check(
        relevant_answer == "CONTEXT_OK",
        "Strong match is allowed through (no false fallback)",
    )
    check(
        weak_answer == FALLBACK_MESSAGE,
        "Weak match triggers the explicit no-hallucination fallback",
    )
    check(
        "don't have enough information" in weak_answer.lower(),
        "Fallback wording matches the required contract",
    )
    print(f"      fallback text = {weak_answer!r}")

    print("      fetching fake URL (expect clean catch, not a crash)...")
    crashed = False
    scraped = "SENTINEL"
    try:
        scraped = fetch_url_safely(FAKE_TIMEOUT_URL, timeout=SCRAPE_TIMEOUT_SECONDS)
    except Exception:
        crashed = True
        traceback.print_exc()

    check(not crashed, "Fake URL scrape did not crash the process")
    check(scraped == "", "Failed scrape returns empty content instead of raising")


def test_ollama_connection() -> None:
    print("\n=== 3. Ollama Local Connection ===")
    tags = None
    try:
        tags = ollama_get("/api/tags", timeout=8)
        check(True, "GET /api/tags succeeded (Ollama is reachable)")
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        check(False, "GET /api/tags succeeded (Ollama is reachable)", str(exc))
        print("      Start Ollama with: ollama serve")
        return

    models = [m.get("name", "") for m in (tags or {}).get("models", [])]
    print(f"      installed models = {models or '(none)'}")
    has_embed = any(EMBEDDING_MODEL in name for name in models)
    check(
        has_embed,
        f"Embedding model '{EMBEDDING_MODEL}' is installed",
        "run: ollama pull nomic-embed-text" if not has_embed else "",
    )

    try:
        embed_payload = ollama_post(
            "/api/embeddings",
            {"model": EMBEDDING_MODEL, "prompt": "ping from test_my_role"},
            timeout=60,
        )
        vector = embed_payload.get("embedding") or []
        check(isinstance(vector, list) and len(vector) > 8, "Embeddings endpoint returned a vector", f"dim={len(vector)}")
    except Exception as exc:
        check(False, "Embeddings endpoint returned a vector", f"{type(exc).__name__}: {exc}")

    chat_model = pick_chat_model(tags or {})
    if not chat_model:
        print("      [SKIP] No chat model installed — embeddings ping is enough for RAG.")
        print("      Optional: ollama pull llama3")
        return

    try:
        gen = ollama_post(
            "/api/generate",
            {
                "model": chat_model,
                "prompt": "Reply with exactly: PONG",
                "stream": False,
                "options": {"num_predict": 8, "temperature": 0},
            },
            timeout=90,
        )
        reply = (gen.get("response") or "").strip()
        check(bool(reply), f"Generate call to '{chat_model}' returned text", f"reply={reply!r}")
    except Exception as exc:
        check(False, f"Generate call to '{chat_model}' returned text", f"{type(exc).__name__}: {exc}")


def main() -> int:
    print("AI Research Assistant — role verification")
    print(f"Ollama: {OLLAMA_BASE_URL}  |  embeddings: {EMBEDDING_MODEL}")
    print(f"Distance threshold (L2): {DISTANCE_THRESHOLD}")

    try:
        rag = test_rag_pipeline()
        test_reliability(rag)
        test_ollama_connection()
    except Exception:
        print("\n[FAIL] Unhandled exception — script crashed:")
        traceback.print_exc()
        return 1

    print("\n=== Summary ===")
    print(f"Passed: {_PASSED}   Failed: {_FAILED}")
    if _FAILED:
        print("RESULT: NOT READY TO MERGE")
        return 1
    print("RESULT: ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
