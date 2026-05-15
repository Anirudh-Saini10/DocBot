import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from typing import Optional, List, Dict, Any

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_drive_service():
    """Authenticate and return a Google Drive API service object."""
    creds_info = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not creds_info:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON environment variable is not set.")

    # Accept either raw JSON string or a file path
    if creds_info.strip().startswith("{"):
        try:
            info = json.loads(creds_info)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"GOOGLE_SERVICE_ACCOUNT_JSON is malformed JSON: {e}. "
                "Make sure it uses double quotes (not single quotes) and is minified to one line."
            ) from e
    else:
        with open(creds_info, "r", encoding="utf-8") as f:
            info = json.load(f)

    credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=credentials, static_discovery=False)


def search_drive(
    q: str,
    folder_id: Optional[str] = None,
    page_size: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search Google Drive using the Drive API `q` parameter.
    Optionally restrict to a specific folder.
    """
    service = get_drive_service()

    # If folder_id is provided, scope the search to that folder
    if folder_id:
        # Make sure the query is scoped to the folder
        if q:
            q = f"{q} and '{folder_id}' in parents"
        else:
            q = f"'{folder_id}' in parents"

    results = (
        service.files()
        .list(
            q=q,
            pageSize=page_size,
            fields="files(id, name, mimeType, modifiedTime, webViewLink, size, createdTime)",
        )
        .execute()
    )

    files = results.get("files", [])
    return files
