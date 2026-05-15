# DocBot

A conversational AI agent for discovering files in Google Drive.

## Stack
- **Backend:** FastAPI + LangChain (Groq LLM) + Google Drive API
- **Frontend:** Streamlit
- **Hosting:** Railway

## Local Setup
1. `pip install -r requirements.txt`
2. Create `.env` with:
   ```env
   GROQ_API_KEY=gsk_...
   GOOGLE_SERVICE_ACCOUNT_JSON={...}
   DRIVE_FOLDER_ID=...
   ```
3. Run backend: `uvicorn backend.main:app --reload`
4. Run frontend: `streamlit run frontend/app.py`

## Deployment (Railway)
1. Push to GitHub
2. Create Railway project from repo
3. Add **Backend** service (uses `Procfile`)
4. Add **Frontend** service with start command:
   ```
   streamlit run frontend/app.py --server.port $PORT
   ```
5. Set env vars on both services:
   - `GROQ_API_KEY`
   - `GOOGLE_SERVICE_ACCOUNT_JSON` (minified to one line)
   - `DRIVE_FOLDER_ID`
   - Frontend only: `API_URL=<backend service public URL>`

## How It Works
The agent translates natural language (e.g., "Find PDFs from last week") into a Google Drive `q` query string and returns file results with links.
