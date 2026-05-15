# DocBot 🤖

An AI-powered conversational assistant that lets you find files in Google Drive using natural language.

## What It Does
Type plain English like *"Find the quarterly report PDF"* or *"Show me spreadsheets from last week"* and DocBot searches your Google Drive and returns matching files with direct links.

## Tech Stack
| Layer | Technology |
|-------|------------|
| **LLM** | Google Gemini (Flash) via LangChain |
| **Backend** | FastAPI + LangChain + Google Drive API |
| **Frontend** | Streamlit |
| **Hosting** | Railway |

## Key Features
- 🧠 **Natural language queries** — no need to learn Drive search syntax
- 🔗 **Direct file links** — open results in one click
- 📁 **Scoped search** — optionally restrict to a specific Drive folder
- 💬 **Conversation memory** — maintains context across messages
- 🎨 **Modern dark UI** — clean, responsive Streamlit interface

## Local Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file:
   ```env
   GOOGLE_API_KEY=AI...
   GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
   DRIVE_FOLDER_ID=1qkx58doSeYrcLjHPDysJyVJ36PsSqqlt
   ```

   - `GOOGLE_API_KEY` — from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - `GOOGLE_SERVICE_ACCOUNT_JSON` — minified service account key (double quotes, single line)
   - `DRIVE_FOLDER_ID` — the Drive folder you want to search inside

3. Start the backend:
   ```bash
   uvicorn backend.main:app --reload
   ```

4. In a new terminal, start the frontend:
   ```bash
   streamlit run frontend/app.py
   ```

## Deployment (Railway)

1. Push this repo to GitHub.
2. In Railway, create a project from the repo.
3. Add a **Backend** service (auto-detects `Procfile`):
   - Set `PYTHON_VERSION=3.12.0`
   - Set env vars: `GOOGLE_API_KEY`, `GOOGLE_SERVICE_ACCOUNT_JSON`, `DRIVE_FOLDER_ID`
4. Add a **Frontend** service with start command:
   ```
   streamlit run frontend/app.py --server.port $PORT
   ```
   - Set env var: `API_URL=<your-backend-public-url>/chat`
5. Deploy both services.

## How It Works
1. User sends a message in the Streamlit chat UI.
2. Frontend forwards it to the FastAPI `/chat` endpoint.
3. The LangChain agent powered by Gemini translates natural language into a Google Drive `q` query string.
4. The Drive API returns matching files.
5. The LLM summarizes results into a conversational response with clickable links.

## Author
Made by **Aniruddha Saini**
