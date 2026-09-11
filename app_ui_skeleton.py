"""
AI Research Assistant — Gradio UI Skeleton
============================================

Scope of this file (Step 1 of the project):
    - Gradio Blocks layout only.
    - Dummy/placeholder handler functions that demonstrate status
      indicators (gr.Info / gr.Progress) and the expected function
      signatures.
    - NO RAG pipeline, NO LangChain, NO ChromaDB, NO Ollama calls yet.

Why it's structured this way:
    - Each "real" piece of logic (search, scrape, RAG retrieval, LLM
      call) is isolated into its own placeholder function with a
      clear docstring describing what it will do later. In the next
      steps we simply replace the body of these functions — the UI
      wiring (inputs/outputs/events) does not need to change.
    - `gr.State` holds the chat history so it survives across turns
      without relying on global variables (important once we deploy
      or run multiple concurrent users).
"""

import time
import gradio as gr


# ---------------------------------------------------------------------------
# 1. PLACEHOLDER / DUMMY BACKEND FUNCTIONS
#    These will be replaced with real RAG + Ollama logic in later steps.
#    Keeping them separate keeps the UI code decoupled from the logic.
# ---------------------------------------------------------------------------

def validate_url_placeholder(url: str) -> bool:
    """
    PLACEHOLDER: Will later perform real URL validation
    (scheme check, reachability check, timeout handling, etc.).
    For now, just a naive check so the UI has something to call.
    """
    if not url:
        return True  # empty is allowed (URL field is optional)
    return url.startswith("http://") or url.startswith("https://")


def run_chat_turn(user_message, chat_history, progress=gr.Progress()):
    """
    PLACEHOLDER for the chatbot's "Send" action.

    In future steps this function will:
        - Route the query through the RAG pipeline (ChromaDB retrieval).
        - Fall back to "I don't have enough information" when
          similarity scores are too low.
        - Call the local Ollama model for generation.

    For now, it only simulates the pipeline stages using gr.Progress()
    and gr.Info() so the team can see how status indicators work.
    """
    if not user_message or not user_message.strip():
        gr.Warning("Please enter a research question before sending.")
        return chat_history, ""

    # --- Simulated pipeline stages (to be replaced with real logic) ---
    progress(0, desc="Starting...")
    gr.Info("Searching knowledge base...")
    time.sleep(0.5)
    progress(0.33, desc="Searching...")

    gr.Info("Retrieving relevant context...")
    time.sleep(0.5)
    progress(0.66, desc="Analyzing...")

    gr.Info("Generating answer with local model...")
    time.sleep(0.5)
    progress(1.0, desc="Done")

    dummy_answer = (
        f"[DUMMY RESPONSE] This is a placeholder answer to: '{user_message}'. "
        f"Real RAG + Ollama generation will be wired in during the next step."
    )

    # gr.Chatbot expects a list of (user, bot) tuples for its history
    chat_history = chat_history + [(user_message, dummy_answer)]

    return chat_history, ""  # clear the textbox after sending


def add_url_placeholder(url_input, url_list_state, progress=gr.Progress()):
    """
    PLACEHOLDER for adding a manual URL to the knowledge source list.

    In future steps this function will:
        - Validate the URL (format + reachability + timeout).
        - Scrape/fetch content.
        - Chunk it (LangChain text splitters) and embed it into ChromaDB.

    For now, it only validates the format and simulates scraping.
    """
    if not url_input or not url_input.strip():
        gr.Warning("Please enter a URL first.")
        return url_list_state, gr.update(value="")

    if not validate_url_placeholder(url_input.strip()):
        gr.Warning(f"'{url_input}' does not look like a valid URL (must start with http:// or https://).")
        return url_list_state, gr.update(value=url_input)

    progress(0, desc="Validating URL...")
    time.sleep(0.3)
    gr.Info(f"Scraping content from: {url_input}")
    progress(0.5, desc="Scraping...")
    time.sleep(0.5)

    gr.Info("Chunking and indexing content...")
    progress(1.0, desc="Indexed")
    time.sleep(0.3)

    url_list_state = url_list_state + [url_input.strip()]
    gr.Info(f"Added source ({len(url_list_state)} total). ")

    # Clear the textbox, and return updated state for display
    return url_list_state, gr.update(value="")


def generate_report_placeholder(topic, url_list_state, progress=gr.Progress()):
    """
    PLACEHOLDER for the "Generate Report" action.

    In future steps this function will:
        - Run similarity retrieval over ChromaDB for the given topic.
        - Aggregate retrieved chunks + any manually added URLs.
        - Prompt the local Ollama model to produce a structured report.
        - Return "I don't have enough information about this topic."
          when retrieval confidence is too low.

    For now, it just simulates the stages and returns dummy Markdown.
    """
    if not topic or not topic.strip():
        gr.Warning("Please enter a research topic before generating a report.")
        return "*No report generated yet.*"

    progress(0, desc="Starting report generation...")
    gr.Info("Searching indexed sources...")
    time.sleep(0.5)
    progress(0.4, desc="Searching...")

    gr.Info("Analyzing retrieved content...")
    time.sleep(0.5)
    progress(0.8, desc="Analyzing...")

    gr.Info("Compiling structured report...")
    time.sleep(0.4)
    progress(1.0, desc="Done")

    sources_note = (
        f"\n\n**Sources used:** {len(url_list_state)} manual URL(s) + indexed documents."
        if url_list_state else "\n\n**Sources used:** No manual URLs added; using indexed documents only."
    )

    dummy_report = (
        f"## Research Report: {topic}\n\n"
        f"*(This is placeholder content. Real content will come from the "
        f"RAG pipeline + Ollama in a later step.)*\n\n"
        f"### Summary\n"
        f"- Point 1 about {topic}\n"
        f"- Point 2 about {topic}\n"
        f"- Point 3 about {topic}\n"
        f"{sources_note}"
    )
    return dummy_report


def clear_chat():
    """Resets the chatbot history and input box."""
    return [], ""


# ---------------------------------------------------------------------------
# 2. GRADIO UI LAYOUT
# ---------------------------------------------------------------------------

with gr.Blocks(title="AI Research Assistant", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🔎 AI Research Assistant
        Ask research questions, add reference URLs, and generate a structured report.
        *(Backend logic is a placeholder for now — RAG + Ollama integration comes next.)*
        """
    )

    # --- STATE ---
    # chat_history_state: list of (user, bot) tuples, used by gr.Chatbot
    # url_list_state: list of manually added URLs (will feed into RAG indexing later)
    chat_history_state = gr.State([])
    url_list_state = gr.State([])

    with gr.Row():

        # ------------------ LEFT COLUMN: Chat Interface ------------------
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")

            chatbot = gr.Chatbot(
                label="Research Assistant Chat",
                height=420,
                show_copy_button=True,
            )

            with gr.Row():
                query_input = gr.Textbox(
                    label="Research Question",
                    placeholder="e.g., What are the latest trends in renewable energy?",
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Clear Chat", size="sm")

        # ------------------ RIGHT COLUMN: Sources + Report ------------------
        with gr.Column(scale=1):
            gr.Markdown("### 🔗 Manual Source URLs")

            url_input = gr.Textbox(
                label="Add a URL",
                placeholder="https://example.com/article",
            )
            add_url_btn = gr.Button("➕ Add URL")

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
    # 3. EVENT WIRING
    #    Each control is connected to its corresponding placeholder
    #    function. Only these function bodies need to change later.
    # -----------------------------------------------------------------

    # --- Chat send (button click + Enter key) ---
    send_btn.click(
        fn=run_chat_turn,
        inputs=[query_input, chat_history_state],
        outputs=[chat_history_state, query_input],
    ).then(
        fn=lambda h: h,  # sync state -> visible chatbot component
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

    # --- Clear chat ---
    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chat_history_state, query_input],
    ).then(
        fn=lambda h: h,
        inputs=chat_history_state,
        outputs=chatbot,
    )

    # --- Add URL ---
    def refresh_url_display(url_list):
        """Small helper to render the current URL list as Markdown bullets."""
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

    # --- Generate report ---
    generate_report_btn.click(
        fn=generate_report_placeholder,
        inputs=[topic_input, url_list_state],
        outputs=report_output,
    )


# ---------------------------------------------------------------------------
# 4. LAUNCH
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.queue()  # required for gr.Progress() / streaming-style updates
    demo.launch()
