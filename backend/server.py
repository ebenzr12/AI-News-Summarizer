"""
Uvicorn server launcher for AI News Summarizer web application.
Usage:
    python -m backend.server
"""

import uvicorn
from config.config import API_HOST, API_PORT


def main():
    uvicorn.run(
        "backend.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
