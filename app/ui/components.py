"""
Gradio UI Components

Builds the full Gradio Blocks interface with:
- Left panel:  Ingestion controls (topic input, max papers slider, ingest button, status log)
- Right panel: Chat interface with streaming agent responses
- Header:      Collection selector dropdown
"""

import asyncio
import logging
import uuid

import gradio as gr

from app.config import settings
from app.ingestion.pipeline import ingest_papers
from app.ingestion.indexer import list_collections, delete_collection, topic_to_collection_name
from app.agent.graph import get_graph, stream_agent

logger = logging.getLogger(__name__)


# ── Color Palette & Theme ─────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%)",
    body_background_fill_dark="linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%)",
    block_background_fill="#1a1a2e",
    block_background_fill_dark="#1a1a2e",
    block_border_color="#2d2b55",
    block_border_color_dark="#2d2b55",
    block_label_text_color="#c4b5fd",
    block_label_text_color_dark="#c4b5fd",
    block_title_text_color="#e0e0ff",
    block_title_text_color_dark="#e0e0ff",
    input_background_fill="#16213e",
    input_background_fill_dark="#16213e",
    input_border_color="#2d2b55",
    input_border_color_dark="#2d2b55",
    button_primary_background_fill="linear-gradient(135deg, #7c3aed 0%, #a855f7 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #7c3aed 0%, #a855f7 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #6d28d9 0%, #9333ea 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #6d28d9 0%, #9333ea 100%)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#2d2b55",
    button_secondary_background_fill_dark="#2d2b55",
    button_secondary_text_color="#c4b5fd",
    button_secondary_text_color_dark="#c4b5fd",
    shadow_drop="0 4px 14px 0 rgba(124, 58, 237, 0.15)",
    shadow_drop_lg="0 8px 24px 0 rgba(124, 58, 237, 0.2)",
    checkbox_background_color="#16213e",
    checkbox_background_color_dark="#16213e",
    slider_color="#7c3aed",
    slider_color_dark="#7c3aed",
)


# ── Custom CSS ─────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────── */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* ── Header ─────────────────────────────────────── */
#header-title {
    text-align: center;
    background: linear-gradient(135deg, #c4b5fd 0%, #a855f7 50%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0;
    padding-bottom: 0;
}

#header-subtitle {
    text-align: center;
    color: #8b8b9e !important;
    font-size: 0.95rem;
    margin-top: 0;
    padding-top: 0;
}

/* ── Panels ─────────────────────────────────────── */
.panel-container {
    border: 1px solid #2d2b55;
    border-radius: 16px;
    padding: 20px;
    background: rgba(26, 26, 46, 0.8);
    backdrop-filter: blur(10px);
}

/* ── Chatbot ────────────────────────────────────── */
#chatbot {
    height: 520px !important;
    border-radius: 12px;
    border: 1px solid #2d2b55;
}

#chatbot .message {
    border-radius: 12px !important;
}

/* ── Ingest Log ─────────────────────────────────── */
#ingest-log {
    height: 220px !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    border-radius: 12px;
    background: #0f0c29 !important;
    color: #a5b4fc !important;
    border: 1px solid #2d2b55;
}

/* ── Status Badge ───────────────────────────────── */
.status-ready { color: #4ade80; }
.status-busy  { color: #facc15; }
.status-error { color: #f87171; }

/* ── Smooth transitions ────────────────────────── */
button, input, textarea, select {
    transition: all 0.2s ease-in-out !important;
}

button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3) !important;
}

/* ── Collection Selector ────────────────────────── */
#collection-selector {
    border-radius: 12px;
}

/* ── Ingestion progress ─────────────────────────── */
.ingestion-stat {
    background: linear-gradient(135deg, #1e1b4b 0%, #2d2b55 100%);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
"""


# ── Callback Handlers ─────────────────────────────────────────────────


def refresh_collections():
    """Refresh the collection dropdown with current ChromaDB collections."""
    try:
        collections = list_collections()
        if not collections:
            return gr.update(choices=[], value=None), "No collections found. Ingest papers to get started."
        choices = [
            (f"{c.display_name} ({c.document_count} docs)", c.name)
            for c in collections
        ]
        return gr.update(choices=choices, value=choices[0][1]), f"Found {len(collections)} collection(s)."
    except Exception as e:
        return gr.update(choices=[], value=None), f"Error: {e}"


def run_ingestion(topic, max_papers, progress=gr.Progress()):
    """Run the ingestion pipeline with progress updates."""
    if not topic or not topic.strip():
        yield "⚠️ Please enter a research topic.", gr.update(), gr.update()
        return

    log_lines = []

    def progress_callback(stage, detail):
        emoji_map = {
            "fetch": "📥",
            "parse": "📄",
            "chunk": "✂️",
            "embed": "🧠",
            "done": "✅",
            "error": "❌",
        }
        emoji = emoji_map.get(stage, "ℹ️")
        line = f"{emoji} [{stage.upper()}] {detail}"
        log_lines.append(line)

    # Yield initial status
    log_lines.append(f"🚀 Starting ingestion for topic: '{topic}' (max {max_papers} papers)")
    yield "\n".join(log_lines), gr.update(), gr.update()

    try:
        result = ingest_papers(
            topic=topic.strip(),
            max_results=int(max_papers),
            progress_callback=progress_callback,
        )

        # Final status
        if result.success:
            log_lines.append(f"\n🎉 SUCCESS! Stored {result.chunks_stored} chunks from {result.papers_parsed} papers.")
        else:
            log_lines.append(f"\n⚠️ Ingestion completed with issues. Check logs above.")

        # Refresh collections
        collections_update, collections_status = refresh_collections()
        log_lines.append(f"📚 {collections_status}")

        yield "\n".join(log_lines), collections_update, gr.update(value=topic_to_collection_name(topic.strip()))

    except Exception as e:
        log_lines.append(f"\n❌ FATAL ERROR: {e}")
        yield "\n".join(log_lines), gr.update(), gr.update()


def delete_collection_handler(collection_name):
    """Delete a collection and refresh the dropdown."""
    if not collection_name:
        return "No collection selected.", gr.update(), gr.update()

    try:
        # We need the topic name — find it from the collections list
        collections = list_collections()
        target = next((c for c in collections if c.name == collection_name), None)
        if target:
            delete_collection(target.display_name)
            collections_update, _ = refresh_collections()
            return f"🗑️ Deleted collection '{target.display_name}'.", collections_update, gr.update(value=None)
        else:
            return f"Collection '{collection_name}' not found.", gr.update(), gr.update()
    except Exception as e:
        return f"Error deleting collection: {e}", gr.update(), gr.update()


async def chat_respond(message, chat_history, collection_name):
    """Stream the agent's response into the chat."""
    if not message or not message.strip():
        yield chat_history, ""
        return

    if not collection_name:
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Please select a collection first. Ingest some papers using the panel on the left."},
        ]
        yield chat_history, ""
        return

    # Add user message
    chat_history = chat_history + [{"role": "user", "content": message}]
    yield chat_history, ""

    # Stream the agent response
    thread_id = f"session_{uuid.uuid4().hex[:8]}"
    full_response = ""
    tool_updates = []

    try:
        async for event in stream_agent(
            query=message.strip(),
            collection_name=collection_name,
            thread_id=thread_id,
        ):
            if event["type"] == "tool_call":
                tool_updates.append(event["content"])
                # Show tool calls as a temporary assistant message
                tool_text = "\n".join(tool_updates)
                chat_history_with_tools = chat_history + [
                    {"role": "assistant", "content": tool_text},
                ]
                yield chat_history_with_tools, ""

            elif event["type"] == "tool_result":
                tool_updates.append(event["content"])
                tool_text = "\n".join(tool_updates)
                chat_history_with_tools = chat_history + [
                    {"role": "assistant", "content": tool_text},
                ]
                yield chat_history_with_tools, ""

            elif event["type"] == "text":
                full_response += event["content"]
                # Show streaming text after tool updates
                if tool_updates:
                    display_text = "\n".join(tool_updates) + "\n\n---\n\n" + full_response
                else:
                    display_text = full_response
                chat_history_stream = chat_history + [
                    {"role": "assistant", "content": display_text},
                ]
                yield chat_history_stream, ""

            elif event["type"] == "end":
                break

        # Final response
        if not full_response and not tool_updates:
            full_response = "I wasn't able to generate a response. Please try rephrasing your question."

        if tool_updates and full_response:
            final_text = "\n".join(tool_updates) + "\n\n---\n\n" + full_response
        elif full_response:
            final_text = full_response
        else:
            final_text = "\n".join(tool_updates)

        chat_history_final = chat_history + [
            {"role": "assistant", "content": final_text},
        ]
        yield chat_history_final, ""

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        chat_history_error = chat_history + [
            {"role": "assistant", "content": error_msg},
        ]
        yield chat_history_error, ""


# ── Build the Gradio App ──────────────────────────────────────────────


def create_app() -> tuple[gr.Blocks, dict]:
    """Build and return the Gradio Blocks application and launch kwargs.

    Returns:
        A tuple of (app, launch_kwargs) where launch_kwargs contains
        theme and css for Gradio 6.x compatibility.
    """
    # In Gradio 6.x, theme/css moved from Blocks() to launch()
    launch_extras = {
        "theme": THEME,
        "css": CUSTOM_CSS,
    }

    with gr.Blocks(
        title="Agentic Research Assistant",
        analytics_enabled=False,
    ) as app:

        # ── Header ────────────────────────────────────────────
        gr.Markdown(
            "# 🔬 Agentic Research Assistant",
            elem_id="header-title",
        )
        gr.Markdown(
            "Ingest arXiv papers • Ask questions • Get cited answers powered by Gemini + RAG",
            elem_id="header-subtitle",
        )

        with gr.Row(equal_height=False):

            # ══════════════════════════════════════════════════
            # LEFT PANEL — Ingestion Controls
            # ══════════════════════════════════════════════════
            with gr.Column(scale=1, min_width=380):

                gr.Markdown("### 📥 Paper Ingestion")

                topic_input = gr.Textbox(
                    label="Research Topic",
                    placeholder="e.g., transformer architectures, diffusion models...",
                    lines=1,
                    max_lines=1,
                    elem_id="topic-input",
                )

                max_papers_slider = gr.Slider(
                    label="Max Papers",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=25,
                    info="Number of papers to fetch from arXiv",
                )

                with gr.Row():
                    ingest_btn = gr.Button(
                        "🚀 Ingest Papers",
                        variant="primary",
                        size="lg",
                        elem_id="ingest-btn",
                    )

                ingest_log = gr.Textbox(
                    label="Ingestion Log",
                    value="Ready. Enter a topic and click 'Ingest Papers' to begin.",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                    elem_id="ingest-log",
                )

                gr.Markdown("---")
                gr.Markdown("### 📚 Collections")

                collection_dropdown = gr.Dropdown(
                    label="Active Collection",
                    choices=[],
                    value=None,
                    interactive=True,
                    elem_id="collection-selector",
                    info="Select a collection to query",
                )

                with gr.Row():
                    refresh_btn = gr.Button(
                        "🔄 Refresh",
                        variant="secondary",
                        size="sm",
                    )
                    delete_btn = gr.Button(
                        "🗑️ Delete",
                        variant="stop",
                        size="sm",
                    )

                collection_status = gr.Markdown(
                    "Click 'Refresh' to load collections.",
                    elem_id="collection-status",
                )

            # ══════════════════════════════════════════════════
            # RIGHT PANEL — Chat Interface
            # ══════════════════════════════════════════════════
            with gr.Column(scale=2, min_width=500):

                gr.Markdown("### 💬 Ask Your Research Questions")

                chatbot = gr.Chatbot(
                    label="Research Chat",
                    elem_id="chatbot",
                    avatar_images=(None, "https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690b6.svg"),
                    placeholder=(
                        "Select a collection and ask a question.\n\n"
                        "**Try asking:**\n"
                        "- What are the key contributions of the attention mechanism?\n"
                        "- What is the latest state-of-the-art in this area?\n"
                        "- Compare the methods described in the papers\n"
                    ),
                )

                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about the ingested papers or recent research...",
                        lines=1,
                        max_lines=3,
                        scale=4,
                        elem_id="chat-input",
                    )
                    send_btn = gr.Button(
                        "Send ➤",
                        variant="primary",
                        scale=1,
                        size="lg",
                        elem_id="send-btn",
                    )

                with gr.Row():
                    clear_btn = gr.Button(
                        "🗑️ Clear Chat",
                        variant="secondary",
                        size="sm",
                    )

        # ── Footer ────────────────────────────────────────────
        gr.Markdown(
            "<center style='color: #6b7280; font-size: 0.8rem; margin-top: 20px;'>"
            "Powered by Google Gemini 1.5 Flash • LangGraph • ChromaDB • arXiv"
            "</center>"
        )

        # ══════════════════════════════════════════════════════
        # Event Handlers
        # ══════════════════════════════════════════════════════

        # Ingestion
        ingest_btn.click(
            fn=run_ingestion,
            inputs=[topic_input, max_papers_slider],
            outputs=[ingest_log, collection_dropdown, collection_dropdown],
        )

        # Collection management
        refresh_btn.click(
            fn=refresh_collections,
            inputs=[],
            outputs=[collection_dropdown, collection_status],
        )

        delete_btn.click(
            fn=delete_collection_handler,
            inputs=[collection_dropdown],
            outputs=[collection_status, collection_dropdown, collection_dropdown],
        )

        # Chat — Send button
        send_btn.click(
            fn=chat_respond,
            inputs=[chat_input, chatbot, collection_dropdown],
            outputs=[chatbot, chat_input],
        )

        # Chat — Enter key
        chat_input.submit(
            fn=chat_respond,
            inputs=[chat_input, chatbot, collection_dropdown],
            outputs=[chatbot, chat_input],
        )

        # Clear chat
        clear_btn.click(
            fn=lambda: ([], ""),
            inputs=[],
            outputs=[chatbot, chat_input],
        )

        # Load collections on app start
        app.load(
            fn=refresh_collections,
            inputs=[],
            outputs=[collection_dropdown, collection_status],
        )

    return app, launch_extras
