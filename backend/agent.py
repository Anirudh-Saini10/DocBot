import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from backend.drive_tool import search_drive

logger = logging.getLogger(__name__)

# Load .env from project root regardless of CWD
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "")


@tool
def drive_search_tool(query_string: str) -> str:
    """
    Search Google Drive using a valid Drive API 'q' parameter string.
    The query_string must follow Google Drive query syntax.
    Examples:
        - name contains 'report'
        - mimeType='application/pdf'
        - modifiedTime > '2024-05-01T00:00:00'
        - fullText contains 'budget'
    """
    try:
        folder_id = DRIVE_FOLDER_ID if DRIVE_FOLDER_ID else None
        files = search_drive(q=query_string, folder_id=folder_id, page_size=10)
        if not files:
            return "No files found matching your criteria."

        lines = []
        for f in files:
            name = f.get("name", "Unnamed")
            mime = f.get("mimeType", "unknown")
            modified = f.get("modifiedTime", "N/A")
            link = f.get("webViewLink", "")
            lines.append(f"- **{name}** ({mime}) | Modified: {modified} | [Open]({link})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching Drive: {e}"


SYSTEM_PROMPT = """You are DocBot, a helpful file-discovery assistant integrated with Google Drive.
Your job is to understand the user's request, translate it into a valid Google Drive `q` query string, and call the `drive_search_tool` to find files.

## Building the query_string

The `query_string` parameter must be a single, properly-formatted string that follows Google Drive query syntax exactly. Malformed strings will cause 400 errors from the API.

### Quote rules — critical
- Every string value must be wrapped in SINGLE quotes: `'value'`
- NEVER use double quotes (`"`) anywhere inside the query string
- NEVER mix single and double quotes
- NEVER add unmatched or extra parentheses

### Operators
- Equality:   `mimeType = 'application/pdf'`
- Contains:   `name contains 'report'`
- Comparison: `modifiedTime > '2024-05-01T00:00:00'`
- Logical:    `and`, `or`, `not`

### Combining conditions
Wrap each individual condition in parentheses when joining with `and` / `or`:
  `(name contains 'budget') and (modifiedTime > '2024-05-01T00:00:00')`

### Common field reference
| Goal                        | Query fragment                                          |
|-----------------------------|---------------------------------------------------------|
| File name (partial match)   | `name contains 'keyword'`                               |
| File name (exact match)     | `name = 'filename.pdf'`                                 |
| PDF files                   | `mimeType = 'application/pdf'`                          |
| Google Doc                  | `mimeType = 'application/vnd.google-apps.document'`     |
| Google Sheet                | `mimeType = 'application/vnd.google-apps.spreadsheet'`  |
| Google Slide                | `mimeType = 'application/vnd.google-apps.presentation'` |
| Full-text search            | `fullText contains 'keyword'`                           |
| Modified after a date       | `modifiedTime > '2024-05-01T00:00:00'`                  |
| Modified before a date      | `modifiedTime < '2024-06-01T00:00:00'`                  |
| Not in trash                | `trashed = false`                                       |

### Valid query_string examples
Single condition:
  `name contains 'project'`
  `mimeType = 'application/pdf'`
  `modifiedTime > '2024-05-01T00:00:00'`
  `fullText contains 'quarterly report'`

Combined conditions:
  `(name contains 'project') and (modifiedTime > '2024-05-01T00:00:00')`
  `(mimeType = 'application/pdf') and (name contains 'invoice')`
  `(fullText contains 'budget') and (modifiedTime > '2024-01-01T00:00:00') and (trashed = false)`

### Common mistakes to avoid
- WRONG: `name contains "project"`              — double quotes are not allowed
- WRONG: `name contains 'project')`             — unmatched closing parenthesis
- WRONG: `modifiedTime > "2024-05-01T00:00:00"` — double quotes around date
- WRONG: `(name contains 'project'`             — unmatched opening parenthesis
- RIGHT: `(name contains 'project') and (modifiedTime > '2024-05-01T00:00:00')`

## Behavior guidelines
- If the user mentions a file name (exact or partial), use `name contains 'keyword'`.
- If the user wants a specific file type, map it to the correct mimeType string.
- If the user refers to a date (e.g., "last week", "after May 1"), convert it to ISO 8601 format and use `modifiedTime > '...'`.
- If the user wants text inside documents, use `fullText contains 'keyword'`.
- IMPORTANT: After receiving tool results, you MUST always respond with a plain-text conversational summary. Never call a tool again after receiving tool results — just summarize what was found and include the file links.
- If no files were found, tell the user clearly and suggest they try a different search.
- If the query is ambiguous, ask a clarifying question instead of guessing.
"""


def get_agent():
    """Build and return (llm_with_tools, plain_llm) for the two-step agent loop."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )
    llm_with_tools = llm.bind_tools([drive_search_tool])
    # Plain LLM (no tools bound) is used for the summarization step so the
    # model cannot loop back into another tool call and is forced to produce text.
    plain_llm = llm
    return llm_with_tools, plain_llm


def _build_fallback_summary(tool_results: list) -> str:
    """Construct a plain-text summary directly from raw tool results when the
    LLM fails to produce one."""
    parts = []
    for tr in tool_results:
        result = tr.get("result", "")
        if result and result != "No files found matching your criteria.":
            parts.append(result)
    if parts:
        return "Here are the files I found in your Drive:\n\n" + "\n".join(parts)
    return "I searched your Google Drive but couldn't find any files matching your request. Try rephrasing or using different keywords."


def run_agent(user_message: str, conversation_history: list = None) -> Dict[str, Any]:
    """
    Run the agent on a user message.
    conversation_history is a list of dicts: {"role": "user"|"assistant", "content": str}
    """
    from langchain_core.messages import ToolMessage

    if conversation_history is None:
        conversation_history = []

    llm_with_tools, plain_llm = get_agent()

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in conversation_history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_message))

    logger.debug("Invoking LLM (with tools) for user message: %r", user_message)
    response = llm_with_tools.invoke(messages)
    logger.debug("Initial LLM response — content: %r  tool_calls: %s", response.content, response.tool_calls)

    # If the LLM decided to call a tool
    if response.tool_calls:
        tool_results = []
        for tc in response.tool_calls:
            if tc["name"] == "drive_search_tool":
                logger.debug("Executing drive_search_tool with args: %s", tc["args"])
                result = drive_search_tool.invoke(tc["args"])
                logger.debug("drive_search_tool result: %r", result)
                tool_results.append({"tool_call_id": tc["id"], "result": result})

        # Append the assistant's tool-call turn and each tool result so the
        # plain LLM has full context for its summary.
        messages.append(response)
        for tr in tool_results:
            messages.append(ToolMessage(content=tr["result"], tool_call_id=tr["tool_call_id"]))

        # Use the plain LLM (no tools bound) so the model is forced to return
        # a text summary rather than attempting another tool call.
        logger.debug("Invoking plain LLM for summarization step")
        final_response = plain_llm.invoke(messages)
        logger.debug("Final LLM response — content: %r", final_response.content)

        content = final_response.content

        # Fallback: if the model still returned empty content, build the
        # summary directly from the raw tool results.
        if not content or not content.strip():
            logger.warning("Final LLM response was empty — using fallback summary from tool results")
            content = _build_fallback_summary(tool_results)

        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_results,
        }

    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [],
    }
