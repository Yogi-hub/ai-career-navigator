import os
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.tools.tavily_search import TavilySearchResults

SUPPORTED_FORMATS = ['.pdf', '.docx'] # Critical parameter
SEARCH_RESULTS_K = 5 # Critical parameter

@tool
def extract_resume_tool(file_path: str) -> str:
    """Parses a resume (PDF or DOCX) and returns the text content. 
    Use this to ingest the user's background information."""
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.docx':
        loader = Docx2txtLoader(file_path)
    else:
        return f"Error: Unsupported file format. Please upload {SUPPORTED_FORMATS}"

    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])

@tool
def career_market_search(query: str) -> str:
    """Searches for real-time 2026 career trends, salaries, and visa info. 
    Use this to perform semantic mapping for countries."""
    search = TavilySearchResults(max_results=SEARCH_RESULTS_K)
    results = search.invoke({"query": query})
    return str(results)