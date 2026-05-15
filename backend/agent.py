import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from backend.drive_tool import search_drive

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


SYSTEM_PROMPT = """You are TailorTalk, a helpful file-discovery assistant integrated with Google Drive.
Your job is to understand the user's request, translate it into a valid Google Drive `q` query string, and call the `drive_search_tool` to find files.

Guidelines:
- If the user mentions a file name (exact or partial), use `name contains 'keyword'`.
- If the user wants a specific file type, use `mimeType='...'` (e.g., `application/pdf` for PDFs, `application/vnd.google-apps.document` for Google Docs).
- If the user refers to a date (e.g., "last week", "after May 1"), convert it to ISO 8601 and use `modifiedTime > '...'`.
- If the user wants text inside documents, use `fullText contains 'keyword'`.
- You can combine conditions with `and` / `or`.
- Always be conversational. After receiving tool results, summarize them nicely for the user and provide the file links.
- If the query is ambiguous, ask a clarifying question instead of guessing.
"""


def get_agent():
    """Build and return the LangChain agent with tool calling."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )
    llm_with_tools = llm.bind_tools([drive_search_tool])
    return llm_with_tools


def run_agent(user_message: str, conversation_history: list = None) -> Dict[str, Any]:
    """
    Run the agent on a user message.
    conversation_history is a list of dicts: {"role": "user"|"assistant", "content": str}
    """
    if conversation_history is None:
        conversation_history = []

    agent = get_agent()

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in conversation_history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_message))

    response = agent.invoke(messages)

    # If the LLM decided to call a tool
    if response.tool_calls:
        tool_results = []
        for tc in response.tool_calls:
            if tc["name"] == "drive_search_tool":
                result = drive_search_tool.invoke(tc["args"])
                tool_results.append({"tool_call_id": tc["id"], "result": result})

        # Append tool results back to conversation for the LLM to summarize
        messages.append(response)
        for tr in tool_results:
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(content=tr["result"], tool_call_id=tr["tool_call_id"]))

        final_response = agent.invoke(messages)
        return {
            "role": "assistant",
            "content": final_response.content,
            "tool_calls": tool_results,
        }

    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [],
    }
