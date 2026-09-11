"""
integration_notes.py
======================
NOT a standalone app — this is a reference snippet showing the specific
edits to make in `app_ui_skeleton.py` to wire in `RAGManager`.

Only 3 things change in the main app file:
    1. Import + instantiate RAGManager once, near the top.
    2. `add_url_placeholder` calls `rag.add_documents(...)` after scraping.
    3. `run_chat_turn` / `generate_report_placeholder` call
       `rag.retrieve_context(...)` before the (still-placeholder) LLM step.

LLM prompt construction and generation are still NOT included here —
that's Step 3. For now we just print/inspect the retrieved chunks so we
can visually confirm retrieval is working.
"""

# ============================================================
# 1. Near the top of app_ui_skeleton.py, add:
# ============================================================
from rag_manager import RAGManager

# Instantiate ONCE at module load time (not inside a function!) so the
# ChromaDB connection and Ollama embedding function are reused across
# every request instead of being recreated on every click.
rag = RAGManager(
    collection_name="research_assistant_kb",
    persist_directory="./chroma_db",
    embedding_model="nomic-embed-text",   # must be pulled via `ollama pull nomic-embed-text`
    use_persistent_storage=True,
)


# ============================================================
# 2. Update `add_url_placeholder` to actually index scraped text.
#    (Scraping itself is still a placeholder — that's part of your
#    Reliability/Error-Handling work, e.g. timeouts + fetch logic.
#    For now we simulate "scraped_text" until that piece exists.)
# ============================================================
def add_url_placeholder(url_input, url_list_state, progress=None):
    import gradio as gr

    if not url_input or not url_input.strip():
        gr.Warning("Please enter a URL first.")
        return url_list_state, gr.update(value="")

    if not validate_url_placeholder(url_input.strip()):
        gr.Warning(f"'{url_input}' does not look like a valid URL.")
        return url_list_state, gr.update(value=url_input)

    progress(0, desc="Validating URL...")
    gr.Info(f"Scraping content from: {url_input}")
    progress(0.5, desc="Scraping...")

    # --- Placeholder scraped content (replace with real scraper later) ---
    scraped_text = (
        f"[Placeholder scraped content for {url_input}]. "
        f"Real scraping + timeout handling will replace this string."
    )

    gr.Info("Chunking and indexing content...")
    try:
        num_chunks = rag.add_documents(scraped_text, source_url=url_input.strip())
        progress(1.0, desc="Indexed")
        gr.Info(f"Indexed {num_chunks} chunk(s) from this source.")
    except ValueError as e:
        gr.Warning(str(e))
        return url_list_state, gr.update(value=url_input)

    url_list_state = url_list_state + [url_input.strip()]
    return url_list_state, gr.update(value="")


# ============================================================
# 3. Update `run_chat_turn` to retrieve context before "generating".
#    The actual LLM call is still a dummy string — Step 3 replaces
#    the `dummy_answer` line with a real Ollama prompt + response,
#    and adds the "I don't have enough information" fallback based
#    on how far/close the retrieved distances are.
# ============================================================
def run_chat_turn(user_message, chat_history, progress=None):
    import gradio as gr

    if not user_message or not user_message.strip():
        gr.Warning("Please enter a research question before sending.")
        return chat_history, ""

    progress(0, desc="Starting...")
    gr.Info("Searching knowledge base...")

    # --- NEW: real similarity retrieval from ChromaDB ---
    retrieved_chunks = rag.retrieve_context(user_message, k=3)
    progress(0.5, desc="Analyzing retrieved context...")

    if not retrieved_chunks:
        gr.Info("No relevant context found in the knowledge base yet.")
        context_preview = "(no context retrieved)"
    else:
        gr.Info(f"Retrieved {len(retrieved_chunks)} relevant chunk(s).")
        context_preview = "\n---\n".join(
            f"[{c['source']}] {c['text'][:150]}..." for c in retrieved_chunks
        )

    progress(1.0, desc="Done")

    # Still a dummy answer — Step 3 will feed `retrieved_chunks` into an
    # actual Ollama prompt instead of just previewing them here.
    dummy_answer = (
        f"[DUMMY RESPONSE] Retrieved context for '{user_message}':\n\n"
        f"{context_preview}\n\n"
        f"(Real Ollama generation comes in Step 3.)"
    )

    chat_history = chat_history + [(user_message, dummy_answer)]
    return chat_history, ""
