import os
import json
import pandas as pd
from typing import TypedDict, List, Tuple, Union, Optional
from urllib.request import urlopen
from urllib.parse import quote
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field as PydanticField

# --- NEW IMPORTS FOR REAL FILE PROCESSING ---
import base64
import tempfile
import io
import traceback
# ---

# *** THIS IS THE CORS FIX ***
from fastapi.middleware.cors import CORSMiddleware
# *** END FIX ***

# Core LangChain and Pydantic
from pydantic import BaseModel as LangChainBaseModel, Field
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain.agents.middleware import wrap_tool_call
from langchain.agents import create_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Partner Package Imports (Google)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 

# --- NEW IMPORT FOR PDFS ---
from langchain_community.document_loaders import PyPDFLoader
# ---

# Community Package Imports
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    class FAISS:
        def __init__(self, *args, **kwargs):
            raise ImportError("FAISS is required for RAG functionality. Install faiss-cpu or faiss-gpu.")

# ==============================================================================
# --- 2. TOOLS AREA ---
# ==============================================================================

# Global API Key setup
API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    raise EnvironmentError("API_KEY (GOOGLE_API_KEY) is not set. Cannot initialize LLM.")

# --- CONTEXT DEFINITION (For Middleware/Agent State) ---
class Context(TypedDict):
    user_role: str
    # This will hold the *content* of the files, not just names
    file_context: str 

# --- STRUCTURED OUTPUT SCHEMA (For new tool) ---
class AuditFinding(LangChainBaseModel):
    finding_title: str = Field(description="A concise title for the audit finding.")
    description: str = Field(description="A detailed description of the finding.")

class AuditReport(LangChainBaseModel):
    report_title: str = Field(description="The main title for this audit summary report.")
    summary: str = Field(description="A brief, one-paragraph summary of the key findings.")
    findings: List[AuditFinding] = Field(description="A list of specific audit findings.")

# --- TOOL 1: Web Search ---
@tool
def web_search(query: str) -> str:
    """Search for live weather information and return the current weather description."""
    try:
        if "weather" in query.lower():
            location = query.lower().replace("weather", "").strip()
            encoded_location = quote(location)
            url = f"https://wttr.in/{encoded_location}?format=j1"
            with urlopen(url) as response:
                data = json.loads(response.read())
            #current = data["current_condition"][0]
            current = data["current_condition"][0]
            desc = current["weatherDesc"][0]["value"]
            temp_c = current["temp_C"]
            feels_like = current["FeelsLikeC"]
            humidity = current["humidity"]
            wind_speed = current["windspeedKmph"]

            return (
                f"🌦️ Current weather in {location.title()}:\n"
                f"Condition: {desc}\n"
                f"Temperature: {temp_c}°C (feels like {feels_like}°C)\n"
                f"Humidity: {humidity}%\n"
                f"Wind: {wind_speed} km/h"
            )
            #return (f"Current weather in {location.title()}: {current['weatherDesc'][0]['value']}.")
        else:
            return f"Results for {query}: Search functionality placeholder."
    except Exception as e:
        return f"Error during web search: {e}"

# --- TOOL 2: Calculation ---
@tool
def calcPowerIfGivenNum(query: int) -> int:
    """returns square of given number."""
    return query**2

# --- TOOL 3: RAG/Audit Search (Tool for LLM decision) ---
class AuditQueryInput(LangChainBaseModel):
    query: str = PydanticField(description="The specific audit question or request to be run against the data.")
    # This field is now just for context, the real data is in the 'file_context'
    file_names: List[str] = PydanticField(description="A list of file names that are currently attached. The agent will use this to perform the RAG query.")

def get_embeddings():
    """Uses the GoogleGenerativeAIEmbeddings model."""
    try:
        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=API_KEY)
    except Exception:
        raise EnvironmentError("GoogleGenerativeAIEmbeddings library must be installed and configured.")

@tool(args_schema=AuditQueryInput)
def search_bank_ledger_and_policies(query: str, file_names: List[str]) -> str:
    """
    Signals the API to perform the RAG execution. 
    The file data is already passed in the main context, so this tool just confirms the RAG query.
    Returns the query string prepended with 'RAG_QUERY:'.
    """
    return f"RAG_QUERY:{query}"


# --- TOOL 4: Structured Report Generator ---
@tool
def generate_summary_report(conversation_context: str) -> AuditReport:
    """
    Generates a structured audit report based on the conversation history.
    The input is the full conversation text.
    The output must be a JSON object matching the AuditReport schema.
    """
    llm_with_tools = LLM_MODEL.with_structured_output(AuditReport)
    
    prompt = (
        "You are an expert auditor. Based on the following conversation, generate a "
        "final audit report with a title, summary, and a list of specific findings.\n\n"
        f"CONVERSATION CONTEXT:\n{conversation_context}"
    )
    try:
        report = llm_with_tools.invoke(prompt)
        # Convert the Pydantic model to a JSON string for the agent
        return report.json()
    except Exception as e:
        return json.dumps({"error": f"Failed to generate structured report: {e}"})


# --- ERROR WRAPPER (Middleware) ---
@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool execution failed. Error: ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

# --- Consolidated Tool List ---
tools_list = [
    web_search, 
    calcPowerIfGivenNum, 
    search_bank_ledger_and_policies,
    generate_summary_report 
]

# --- NEW: Define the structure for an uploaded file ---
class UploadedFile(BaseModel):
    name: str
    content: str # This will be a base64 data URI

# --- RAG EXECUTION FUNCTION (DECOUPLED) ---
# *** THIS IS THE REAL IMPLEMENTATION ***
def execute_rag_and_respond(query_to_run: str, uploaded_files: List[UploadedFile]) -> str:
    """
    Loads documents from base64 strings, creates vectorstore (non-persistent), 
    performs RAG query, and returns final prompt context.
    """
    if not uploaded_files:
        return "I cannot perform the audit query because no documents were provided."

    all_docs_list = [Document(page_content="## Standard Banking Audit and Internal Control Principles\n\n**Segregation of Duties (SOD):** Different individuals must be assigned responsibility for the authorization, custody, and recordkeeping of transactions.", metadata={"source": "Internal_Audit_Policy_V2.1"})]
    
    temp_files_to_clean = []

    try:
        # 1. Decode base64 and save to temp files
        for file in uploaded_files:
            try:
                # Split the data URI (e.g., "data:application/pdf;base64,JVBERi0x...")
                header, base64_data = file.content.split(",", 1)
                file_bytes = base64.b64decode(base64_data)
                
                # Create a named temporary file to hold the bytes
                # Use a specific suffix based on the file name to help loaders
                file_suffix = f"_{file.name}"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix)
                temp_file.write(file_bytes)
                temp_file.close() # Close the file so it can be read by loaders
                temp_files_to_clean.append(temp_file.name)

                # 2. Load the data from the temporary file
                if file.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(temp_file.name, engine='openpyxl')
                    doc_content = df.to_csv(index=False)
                    all_docs_list.append(Document(page_content=doc_content, metadata={"source": file.name}))
                
                elif file.name.endswith(".pdf"):
                    loader = PyPDFLoader(temp_file.name)
                    all_docs_list.extend(loader.load())

            except Exception as e:
                # Provide a more detailed error message
                print(f"!!! Error processing file {file.name}: {e}")
                traceback.print_exc()
                return f"Error processing file {file.name}. Make sure it is a valid Excel or PDF file."
        
        if len(all_docs_list) == 1: # Only the default policy doc is present
             return "No processable (Excel/PDF) files were provided for the audit."

        # 3. Create vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(all_docs_list)
        if not splits: return "Error: Documents loaded but no processable text found."

        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # 4. Perform search
        retrieved_docs = vectorstore.similarity_search(query_to_run, k=5)
            
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 5. Generate final response
        final_prompt = (
            f"You are a professional auditor. Use the following CONTEXT to answer the AUDIT QUESTION fully, citing data points from the CONTEXT. If the context is insufficient, state that.\n\n"
            f"AUDIT QUESTION: {query_to_run}\n\n"
            f"CONTEXT (Transaction/Policy Data):\n\n{context}"
        )
        
        final_llm_response = LLM_MODEL.invoke([HumanMessage(content=final_prompt)])
        return final_llm_response.content

    finally:
        # 6. Clean up all temporary files
        for path in temp_files_to_clean:
            try:
                os.remove(path)
            except Exception as e:
                print(f"Warning: could not clean up temp file {path}: {e}")


# ==============================================================================
# --- 3. AGENT CREATION AREA ---
# ==============================================================================
LLM_MODEL = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)

AGENT_EXECUTOR = create_agent(
    model=LLM_MODEL,
    tools=tools_list, 
    context_schema=Context,
    middleware=[handle_tool_errors], 
    system_prompt=(
        "You are a helpful assistant and an expert auditor. "
        "You should be able to use the provided tools to assist with audit-related queries. "
        "Reply diplomatically and professionally."
        "Include structured data when replying tabular"
        "When the user asks about documents, you MUST call the "
        "`search_bank_ledger_and_policies` tool with the user's question as the **query**."
        "You have access to the file content via the 'file_context' variable."
        "If the user asks to generate a report, call "
        "web_search for live weather information"
        "the `generate_summary_report` tool, providing the conversation context."
    )
)
# ==============================================================================
# --- 4. FASTAPI IMPLEMENTATION ---
# ==============================================================================

app = FastAPI(title="LangChain Gemini Audit Agent API")

# --- *** THIS IS THE CORS FIX *** ---
origins = [
    "http://localhost:5173",  # The default Vite dev server port
    "http://localhost:3000",  # Common React dev port
    "http://localhost",
    "*"                       # Allow all for maximum flexibility
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,    # Allows specified origins
    allow_credentials=True, # Allows cookies
    allow_methods=["*"],    # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allows all headers
)
# --- *** END OF CORS FIX *** ---


# --- UPDATED API input model ---
class ChatRequest(BaseModel):
    chat_history: List[Tuple[str, str]] = PydanticField(default_factory=list, description="The previous turns of the conversation.")
    message: str = PydanticField(description="The current message from the user.")
    files: List[UploadedFile] = PydanticField(default_factory=list, description="List of files uploaded by the user, including name and base64 content.")

# --- *** NEW: SUMMARIZE ENDPOINT MODELS *** ---
class SummarizeRequest(BaseModel):
    chat_history: List[Tuple[str, str]] = PydanticField(default_factory=list)

class SummarizeResponse(BaseModel):
    summary: str
    success: bool
# --- *** END NEW MODELS --- ---


@app.get("/health")
def read_health():
    """Simple health check endpoint."""
    return {"status": "Agent API is running with FastAPI."}

# --- *** NEW: SUMMARIZE ENDPOINT *** ---
@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_endpoint(request: SummarizeRequest):
    """
    Uses the Gemini API to generate a summary of the conversation.
    """
    full_conversation_text = ""
    for human_text, ai_text in request.chat_history:
        if human_text:
            full_conversation_text += f"User: {human_text}\n"
        if ai_text:
            full_conversation_text += f"Auditor: {ai_text}\n"
    
    if not full_conversation_text.strip():
        return SummarizeResponse(summary="There is no conversation to summarize.", success=True)

    prompt = (
        "You are an expert auditor. Please provide a concise, professional summary "
        "of the following conversation between a user and an auditor. Focus on key "
        "questions, findings, and action items.\n\n"
        f"CONVERSATION:\n{full_conversation_text}"
    )
    
    try:
        response = LLM_MODEL.invoke([HumanMessage(content=prompt)])
        summary = response.content
        return SummarizeResponse(summary=summary, success=True)
    except Exception as e:
        print(f"!!! Summarize error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Summarization error: {e}"
        )
# --- *** END NEW ENDPOINT *** ---


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main endpoint to process chat requests using the LangChain Agent.
    """
    langchain_history = []
    full_context_for_report = "" # For the new summarize tool
    for human_text, ai_text in request.chat_history:
        if human_text:
            langchain_history.append(HumanMessage(content=human_text))
            full_context_for_report += f"User: {human_text}\n"
        if ai_text:
            langchain_history.append(AIMessage(content=ai_text))
            full_context_for_report += f"Auditor: {ai_text}\n"

    # Get just the names for the context message
    file_names = [file.name for file in request.files]
    file_context_for_agent = "\n".join([f"File: {f.name} (Content omitted, will be handled by RAG)" for f in request.files])

    if file_names:
        context_message = f"(Attached files: {', '.join(file_names)})"
    else:
        context_message = "(No files are currently attached.)"
        
    message_for_agent = f"{context_message} {request.message}"
    full_context_for_report += f"User: {message_for_agent}\n"

    agent_input = {
        "messages": [HumanMessage(content=message_for_agent)],
        "chat_history": langchain_history,
        # *** THIS IS THE FIX for the file context ***
        "file_context": file_context_for_agent,
        "user_role": "auditor" 
    }
    
    final_agent_output = ""
    total_tokens_used = 0

    try:
        # 4. Invoke the Agent Executor
        result = AGENT_EXECUTOR.invoke(agent_input)
        
        # 5. Token and Output Extraction
        for msg in result.get('messages', []):
            if isinstance(msg, AIMessage) and msg.response_metadata:
                usage = msg.response_metadata.get('usage_metadata')
                if usage:
                    total_tokens_used += usage.get('total_tokens', 0)
        
        # Check if the RAG Tool was called (intent detection)
        tool_call_message = next((msg for msg in result.get("messages", []) if msg.type == "tool"), None)
        
        if tool_call_message and tool_call_message.content and tool_call_message.content.startswith("RAG_QUERY:"):
            
            query_to_run = tool_call_message.content.replace("RAG_QUERY:", "").strip()
            
            # Execute the actual RAG pipeline using the decoupled function
            final_agent_output = execute_rag_and_respond(query_to_run, request.files)
        
        elif tool_call_message and tool_call_message.tool_call_id and "generate_summary_report" in tool_call_message.name:
            # The tool was called, and its *output* (the JSON string) is the final_agent_output
            final_agent_output = tool_call_message.content

        else:
            # Non-RAG/Summary path
            final_agent_output = next((msg.content for msg in reversed(result.get("messages", [])) if msg.type == "ai"), "No direct response generated.")

        return {
            "response": final_agent_output,
            "tokens_used_this_turn": total_tokens_used,
            "success": True
        }

    except Exception as e:
        # Print the full traceback to the server console for debugging
        print("!!! Agent runtime error:", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Agent runtime error: {e}"
        )

# --- 5. SAFE STARTUP BLOCK ---
if __name__ == "__main__":
    # Use PORT 8000 for local testing to avoid permission issues
    port_to_listen = int(os.environ.get("PORT", 8000)) 
    print(f"Starting server on port {port_to_listen}...")
    # Enable reload for local development
    import uvicorn
    # *** THIS IS THE FIX ***
    # The app must be specified as "agent_api:app" (module_name:app_variable)
    uvicorn.run("agent_api:app", host="0.0.0.0", port=port_to_listen, reload=True)

