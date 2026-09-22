"""
Agentic Research Assistant — Application Entry Point

Launches the Gradio Blocks application with the full UI.
Run with: python -m app.main
"""

import logging
import sys

# ── Logging Setup ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Reduce noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("gradio").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main():
    """Launch the Gradio application."""
    from app.ui.components import create_app

    logger.info("=" * 60)
    logger.info("  🔬 Agentic Research Assistant")
    logger.info("  Powered by Gemini 1.5 Flash • LangGraph • ChromaDB")
    logger.info("=" * 60)

    app, launch_extras = create_app()

    app.queue()  # Enable queuing for streaming
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path=None,
        **launch_extras,
    )


if __name__ == "__main__":
    main()
