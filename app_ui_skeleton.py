"""
AI Research Assistant — Gradio UI (Integrated: RAG + Agent)
=============================================================
- Modern Gradio 6.x messages format for Chatbot
- Robust UI-level Try/Except error handling
- ChromaDB RAG integration with distance threshold fallback
- W3Schools Green Theme styling
"""

import traceback
import gradio as gr

from rag_manager import RAGManager
import main as agent_backend


# ---------------------------------------------------------------------------
# 0. CONFIG & THEME (W3Schools Style)
# ---------------------------------------------------------------------------

DISTANCE_THRESHOLD = 0.8

NOT_ENOUGH_INFO_MESSAGE = (
    "I don't have enough information to answer this based on the provided sources."
)

INSUFFICIENT_CONTEXT_NOTICE = (
    "### ⚠️ Insufficient Context\n\n"
    "No relevant content found in the knowledge base for this topic or source. "
    "Please ensure the target URL was successfully indexed and contains extractable text."
)

YOUTUBE_NOT_INDEXED_WARNING = (
    "YouTube transcript unavailable or disabled for this video. "
    "The URL was NOT indexed into the knowledge base."
)

# Custom W3Schools Theme & CSS
w3_theme = gr.themes.Default(
    primary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "sans-serif"],
).set(
    button_primary_background_fill="#04AA6D",
    button_primary_background_fill_hover="#059862",
    button_primary_text_color="#ffffff",
    border_color_primary="#04AA6D",
)

custom_css = """
.primary-btn {
    background-color: #04AA6D !important;
    color: white !important;
    font-weight: 600;
}
.primary-btn:hover {
    background-color: #059862 !important;
}
"""


# ---------------------------------------------------------------------------
# 1. RAG MANAGER INITIALIZATION
# ---------------------------------------------------------------------------
try:
    rag = RAGManager(
        collection_name="research_assistant_kb",
        persist_directory="./chroma_db",
        embedding_model="nomic-embed-text",
        use_persistent_storage=True,
    )
    RAG_AVAILABLE = True
except Exception as startup_error:
    print(f"[STARTUP WARNING] Could not initialize RAGManager: {startup_error}")
    rag = None
    RAG_AVAILABLE = False


def retrieve_context_with_fallback(query: str, k: int = 3, allowed_sources=None):
    if not RAG_AVAILABLE:
        return [], False

    try:
        if rag.is_empty():
            return [], False
        good_chunks = rag.retrieve_relevant_context(
            query,
            k=k,
            distance_threshold=DISTANCE_THRESHOLD,
            allowed_sources=allowed_sources,
        )
    except Exception as e:
        print(f"[RAG ERROR] retrieve_context failed: {e}")
        return [], False

    return good_chunks, bool(good_chunks)


# ---------------------------------------------------------------------------
# 2. VALIDATION HELPERS
# ---------------------------------------------------------------------------

def validate_url_placeholder(url: str) -> bool:
    if not url or not url.strip():
        return False
    clean = url.strip()
    return clean.startswith("http://") or clean.startswith("https://")


# ---------------------------------------------------------------------------
# 3. HANDLER FUNCTIONS
# ---------------------------------------------------------------------------

def run_chat_turn(user_message, chat_history, progress=gr.Progress()):
    if not user_message or not user_message.strip():
        gr.Warning("Please enter a research question before sending.")
        return chat_history, ""

    user_message = user_message.strip()
    answer = NOT_ENOUGH_INFO_MESSAGE

    try:
        progress(0, desc="Checking local knowledge base...")
        good_chunks, has_context = retrieve_context_with_fallback(user_message, k=3)

        if has_context:
            gr.Info(f"Found {len(good_chunks)} relevant chunk(s) in your indexed sources.")
            context_block = "\n\n".join(
                f"[Source: {c['source']}]\n{c['text']}" for c in good_chunks
            )
            agent_prompt = (
                "Relevant context from the local knowledge base:\n\n"
                f"{context_block}\n\n"
                f"User question: {user_message}\n\n"
                "Use the context above if it answers the question. If it doesn't "
                "fully answer it, use your tools (web search, scraping) to find "
                "more current or complete information. Never invent facts that "
                "aren't in the context or in your tool results."
            )
        else:
            gr.Info("Nothing relevant found locally — researching live sources...")
            agent_prompt = user_message

        progress(0.4, desc="Running research agent...")
        result = agent_backend.run_agent(agent_prompt)
        progress(1.0, desc="Done")

        if isinstance(result, dict) and "error" in result:
            answer = f"Sorry, the research agent hit an error: {result['error']}"
        elif result and str(result).strip():
            answer = str(result)

    except Exception as e:
        traceback.print_exc()
        gr.Warning(f"Something went wrong while answering: {e}")
        answer = "Sorry, an internal error occurred while processing your question."

    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer},
    ]
    return chat_history, ""


def add_url_placeholder(url_input, url_list_state, progress=gr.Progress()):
    if not url_input or not url_input.strip():
        gr.Warning("Please enter a URL first.")
        return url_list_state, gr.update(value="")

    url_input = url_input.strip()

    if not validate_url_placeholder(url_input):
        gr.Warning(f"'{url_input}' does not look like a valid URL (must start with http:// or https://).")
        return url_list_state, gr.update(value=url_input)

    if not RAG_AVAILABLE:
        gr.Warning("Knowledge base is offline (Ollama/embeddings unreachable). Check setup.")
        return url_list_state, gr.update(value=url_input)

    try:
        progress(0, desc="Fetching page...")
        scrape_result = agent_backend.scrape_page(url_input)

        is_youtube = agent_backend.is_youtube_url(url_input)
        scraped_text = (scrape_result.get("content") or "").strip()
        scrape_failed = (
            "error" in scrape_result
            or scrape_result.get("skipped_ingestion")
            or not scraped_text
            or scraped_text.startswith("[Warning:")
        )

        if scrape_failed:
            if is_youtube:
                gr.Warning(YOUTUBE_NOT_INDEXED_WARNING)
            else:
                gr.Warning(
                    f"Failed to fetch content from {url_input}. "
                    "Nothing was added to the knowledge base."
                )
            return url_list_state, gr.update(value=url_input)

        progress(0.5, desc="Chunking and indexing content...")
        num_chunks = rag.add_documents(scraped_text, source_url=url_input)
        progress(1.0, desc="Indexed")

        if not num_chunks:
            gr.Warning(
                f"Failed to fetch content from {url_input}. "
                "Nothing was added to the knowledge base."
            )
            return url_list_state, gr.update(value=url_input)

        url_list_state = url_list_state + [url_input]
        gr.Info(f"Successfully indexed: {url_input} ({num_chunks} chunks added)")

    except ValueError as e:
        if agent_backend.is_youtube_url(url_input):
            gr.Warning(YOUTUBE_NOT_INDEXED_WARNING)
        else:
            gr.Warning(
                f"Failed to fetch content from {url_input}. "
                "Nothing was added to the knowledge base."
            )
        print(e)
        return url_list_state, gr.update(value=url_input)
    except Exception as e:
        traceback.print_exc()
        gr.Warning(f"Unexpected error while adding source: {e}")
        return url_list_state, gr.update(value=url_input)

    return url_list_state, gr.update(value="")


def generate_report_placeholder(topic, url_list_state, progress=gr.Progress()):
    if not topic or not topic.strip():
        gr.Warning("Please enter a research topic before generating a report.")
        return "*No report generated yet.*"

    topic = topic.strip()

    if not RAG_AVAILABLE:
        gr.Warning("Knowledge base is offline (Ollama/embeddings unreachable).")
        return "*Report unavailable — knowledge base is offline.*"

    try:
        progress(0, desc="Searching indexed sources...")
        kb_empty = rag.is_empty()
        good_chunks, has_context = retrieve_context_with_fallback(
            topic,
            k=6,
            allowed_sources=url_list_state or [],
        )

        if kb_empty or not has_context or not url_list_state:
            gr.Warning(
                "No relevant indexed content for this topic. "
                "The report was not generated from leftover knowledge-base data."
            )
            return INSUFFICIENT_CONTEXT_NOTICE

        progress(0.4, desc="Compiling sources for report...")
        sources_text = "\n\n".join(
            f"SOURCE URL:\n{c['source']}\n\nSOURCE CONTENT:\n{c['text']}"
            for c in good_chunks
        )

        gr.Info("Generating structured report...")
        progress(0.8, desc="Generating structured report...")
        result = agent_backend.generate_report(sources_text)
        progress(1.0, desc="Done")

        if "error" in result:
            gr.Warning(f"Report generation failed: {result['error']}")
            return f"*Report generation failed: {result['error']}*"

        sources_note = (
            f"\n\n**Sources used:** {len(url_list_state)} manual URL(s) added, "
            f"{len(good_chunks)} indexed chunk(s) matched this topic "
            f"(distance threshold: {DISTANCE_THRESHOLD})."
        )
        return result["report"] + sources_note

    except Exception as e:
        traceback.print_exc()
        gr.Warning(f"Unexpected error while generating report: {e}")
        return "*An internal error occurred while generating the report.*"


def clear_chat():
    return [], ""


def clear_knowledge_base(url_list_state):
    if not RAG_AVAILABLE:
        gr.Warning("Knowledge base is offline (Ollama/embeddings unreachable).")
        return url_list_state or []

    try:
        rag.reset_knowledge_base()
        gr.Info("Knowledge base has been cleared.")
        return []
    except Exception as e:
        traceback.print_exc()
        gr.Warning(f"Failed to clear knowledge base: {e}")
        return url_list_state or []


# ---------------------------------------------------------------------------
# 4. GRADIO UI LAYOUT
# ---------------------------------------------------------------------------

with gr.Blocks(title="AI Research Assistant", css=custom_css) as demo:
    gr.Markdown(
        """
        # 🔎 AI Research Assistant
        Ask research questions, add reference URLs, and generate a structured report.
        Backed by local Ollama models, ChromaDB retrieval, and web research tools.
        """
    )

    chat_history_state = gr.State([])
    url_list_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            chatbot = gr.Chatbot(
                label="Research Assistant Chat",
                height=420,
                type="messages",
            )
            with gr.Row():
                query_input = gr.Textbox(
                    label="Research Question",
                    placeholder="e.g., What are the latest trends in renewable energy?",
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Clear Chat", size="sm")

        with gr.Column(scale=1):
            gr.Markdown("### 🔗 Manual Source URLs")
            url_input = gr.Textbox(
                label="Add a URL",
                placeholder="https://example.com/article",
            )
            add_url_btn = gr.Button("➕ Add URL")
            clear_kb_btn = gr.Button("🗑️ Clear Knowledge Base", size="sm")

            url_display = gr.Markdown(
                value="_No URLs added yet._",
                label="Added Sources",
            )

            gr.Markdown("### 📄 Generate Structured Report")
            topic_input = gr.Textbox(
                label="Report Topic",
                placeholder="e.g., Impact of AI on renewable energy research",
            )
            generate_report_btn = gr.Button("📝 Generate Report", variant="primary")

            report_output = gr.Markdown(
                value="*Report will appear here once generated.*",
                label="Final Report",
            )

    # -----------------------------------------------------------------
    # 5. EVENT WIRING
    # -----------------------------------------------------------------

    send_btn.click(
        fn=run_chat_turn,
        inputs=[query_input, chat_history_state],
        outputs=[chat_history_state, query_input],
    ).then(
        fn=lambda h: h,
        inputs=chat_history_state,
        outputs=chatbot,
    )

    query_input.submit(
        fn=run_chat_turn,
        inputs=[query_input, chat_history_state],
        outputs=[chat_history_state, query_input],
    ).then(
        fn=lambda h: h,
        inputs=chat_history_state,
        outputs=chatbot,
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chat_history_state, query_input],
    ).then(
        fn=lambda h: h,
        inputs=chat_history_state,
        outputs=chatbot,
    )

    def refresh_url_display(url_list):
        if not url_list:
            return "_No URLs added yet._"
        return "\n".join(f"- {u}" for u in url_list)

    add_url_btn.click(
        fn=add_url_placeholder,
        inputs=[url_input, url_list_state],
        outputs=[url_list_state, url_input],
    ).then(
        fn=refresh_url_display,
        inputs=url_list_state,
        outputs=url_display,
    )

    clear_kb_btn.click(
        fn=clear_knowledge_base,
        inputs=[url_list_state],
        outputs=[url_list_state],
    ).then(
        fn=refresh_url_display,
        inputs=url_list_state,
        outputs=url_display,
    )

    generate_report_btn.click(
        fn=generate_report_placeholder,
        inputs=[topic_input, url_list_state],
        outputs=report_output,
    )


# ---------------------------------------------------------------------------
# 6. LAUNCH
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=w3_theme)