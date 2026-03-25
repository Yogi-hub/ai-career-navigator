import os
from typing import Annotated, TypedDict, List, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from utils import extract_resume_tool, career_market_search

load_dotenv(override=True)

# Critical parameters
MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0.0
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model=MODEL_NAME, temperature=TEMPERATURE, api_key=GROQ_API_KEY)
tools = [extract_resume_tool, career_market_search]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a Global Career Mapping Engine.
Your objective is to analyze a user's resume and personal goals to determine their optimal global destination.
You utilize semantic search to map specific technical skills and life expectations against a global database to find the best fit.

STRICT OPERATIONAL STEPS:
1. RESUME INGESTION: If resume data is missing, call 'extract_resume_tool'.
2. INTENT GATE (MANDATORY): Check if the user has specified their primary track: 'Study' (Higher Education) or 'Work'. If unknown, you MUST stop and ask. Do not proceed without this.
3. EXPECTATIONS GATE (OPTIONAL): Ask if they have preferences for climate, target salary, visa flexibility, or work-life balance. Explicitly tell the user these are optional. If the user skips them or says they don't have a preference, proceed to Step 4.
4. SEMANTIC MAPPING & RESEARCH: Once Intent is confirmed (and Expectations gathered or skipped), use 'career_market_search' to cross-reference their profile with real-time 2026 data.
5. FINAL OUTPUT: Provide the ranked Top 5 list.

FORMATTING CONSTRAINTS:
- DO NOT use tables for narrative text or explanations. 
- Use standard Markdown headings (###) and bullet points for the detailed reasoning of each country.
- If the user provided expectations, explicitly state how the country matches them. If not, highlight the general economic/educational pros of the country.
- Do not include redundant summaries or TL;DRs at the end of the response."""

REVIEWER_PROMPT = """You are a Career Quality Auditor. 
Your job is to critique the Career Advisor's draft.
Check for:
1. Did it include all 5 countries?
2. Did it explicitly address the user's weather, visa, and salary preferences (IF the user provided any)?
3. Is the formatting clean (no big tables for text)?
4. Ensure the recommendations do not include the user's current country of residence unless there is a specific 'stay-back' strategy provided.

If the draft is perfect, reply with ONLY the word: "PASSED".
If it needs improvement, provide a bulleted list of specific instructions for the advisor."""

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    resume_path: Optional[str]
    critique: Optional[str]
    revision_count: int

def call_model(state: AgentState):
    messages = state['messages']
    resume_path = state.get('resume_path')
    critique = state.get('critique')
    
    prompt = SYSTEM_PROMPT
    
    # IMPROVED: Forceful command and forward slashes to prevent 400 errors
    if resume_path:
        normalized_path = resume_path.replace("\\", "/")
        prompt += f"\n\nCOMMAND: You must now call 'extract_resume_tool' with file_path='{normalized_path}' to parse the user's resume."
    
    if critique and critique != "PASSED":
        prompt += f"\n\n### CRITICAL FEEDBACK FROM AUDITOR ###\n{critique}\n"
        prompt += "Please revise your previous draft to address the feedback above."
        
    filtered_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    messages = [SystemMessage(content=prompt)] + filtered_messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def reviewer_node(state: AgentState):
    last_msg = state['messages'][-1]
    review_input = [SystemMessage(content=REVIEWER_PROMPT), HumanMessage(content=last_msg.content)]
    critique = llm.invoke(review_input)
    return {"critique": critique.content, "revision_count": state.get("revision_count", 0) + 1}

def should_revise(state: AgentState):
    if state.get("revision_count", 0) >= 2 or "PASSED" in state.get("critique", ""):
        return END
    return "agent"

def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    
    # NEW: Only check for tool_calls if the message is from the AI
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    # 2. Check if the message is a question for the user
    content = str(last_message.content).lower()
    if "top 5" not in content and any(x in content for x in ["work", "study", "expectations"]):
        return END
        
    return "reviewer"

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("reviewer", reviewer_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "reviewer": "reviewer", END: END})
workflow.add_conditional_edges("reviewer", should_revise, {"agent": "agent", END: END})
workflow.add_edge("tools", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)