# 🎥 YouTube Video Summarizer Agent

An asynchronous, production-ready AI agent built with **FastAPI**, **Groq (Whisper Large-v3-Turbo & Llama 3.3 70B)**, and a resilient **Hybrid Audio/Transcript Pipeline**.

The service accepts direct YouTube video URLs or free-text search queries (e.g., *"Current Petrol prices in Pakistan"*) and returns strictly grounded, structured summaries with zero hallucination.

---

## ⚡ Key Features

- **Multi-Format Input Routing:** Accepts standard watch URLs, shortlinks (`youtu.be`), Shorts, and natural language search topics via `yt-dlp`.
- **Resilient Hybrid Transcription:**
  - **Path A (Fast Path):** Direct text caption retrieval via `youtube-transcript-api` (zero compute overhead).
  - **Path B (Fallback Engine):** Audio stream extraction with mobile client impersonation + ultra-fast transcription using Groq's `whisper-large-v3-turbo`.
- **Zero-Hallucination Summarization:** Grounded prompt architecture powered by `llama-3.3-70b-versatile`.
- **Production API:** Fully typed Pydantic v2 schemas and interactive OpenAPI docs (`/docs`).

---

## 🛠️ Tech Stack

- **Framework:** FastAPI, Uvicorn
- **Speech-to-Text:** OpenAI Whisper Large v3 Turbo (via Groq LPUs)
- **LLM Engine:** Meta Llama 3.3 70B Versatile (via Groq API)
- **Scraping & Streaming:** `yt-dlp`, `youtube-transcript-api`
- **Validation:** Pydantic
