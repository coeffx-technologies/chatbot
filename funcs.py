import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import Annotated
from typing_extensions import TypedDict

load_dotenv()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_model() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key,
    )


# ---------------------------------------------------------------------------
# RAG — load PDFs, embed, build retriever
# ---------------------------------------------------------------------------

def build_retriever(knowledge_dir: str = "/media/prince/5A4E832F4E83034D/Rocketsteer_chatbot/knowledge"):
    """Load all PDFs from knowledge/, embed them, return a retriever."""

    # Load every PDF in the folder
    loader = PyPDFDirectoryLoader(knowledge_dir)
    docs = loader.load()

    if not docs:
        raise ValueError(f"No PDFs found in '{knowledge_dir}/' folder.")

    # Split into smaller chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    version="v1"  # Force v1 to avoid the v1beta 'Not Found' error
    )

    # Build in-memory Chroma vector store (fresh every run)
    vectorstore = Chroma.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 5})


def retrieve_context(retriever, query: str) -> str:
    """Fetch relevant chunks for a query and return as plain text."""
    docs = retriever.invoke(query)
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Intent Assessment Tool
# ---------------------------------------------------------------------------

# Rules used to judge a lead's intent level
INTENT_RULES = [
    ("replied_to_email",   "Replied to outreach email",          3),
    ("opened_email",       "Opened email (tracked)",              1),
    ("call_connected",     "Connected on a call",                 3),
    ("call_attempted",     "Call attempted (no connect)",         1),
    ("meeting_booked",     "Meeting / demo booked",               4),
    ("positive_response",  "Expressed interest or asked Qs",      3),
    ("no_response",        "No response at all",                 -2),
    ("rejected",           "Explicitly said not interested",     -4),
    ("follow_up_pending",  "Follow-up is pending / in progress",  1),
    ("accepted_not_replied","Accepted request but no reply",       1),
]

def assess_lead_intent(lead_text: str) -> str:
    """
    Score a lead based on keywords found in their info text.
    Returns a formatted intent report.
    """
    text_lower = lead_text.lower()
    score = 0
    matched_reasons = []

    for key, label, points in INTENT_RULES:
        # Check for keyword signals in the lead notes
        keyword_map = {
            "replied_to_email":    ["replied", "replied to email", "email reply"],
            "opened_email":        ["opened email", "email opened"],
            "call_connected":      ["call connected", "spoke", "talked"],
            "call_attempted":      ["called", "call attempt", "no answer", "voicemail"],
            "meeting_booked":      ["meeting booked", "demo booked", "scheduled"],
            "positive_response":   ["interested", "asked", "wants to know", "tell me more"],
            "no_response":         ["no response", "no reply", "ghosted", "not responded"],
            "rejected":            ["not interested", "rejected", "unsubscribed", "do not contact"],
            "follow_up_pending":   ["follow-up", "in progress", "following up", "pending"],
            "accepted_not_replied":["accepted", "connected but", "accepted but not replied"],
        }
        if any(kw in text_lower for kw in keyword_map[key]):
            score += points
            matched_reasons.append(f"  {'✅' if points > 0 else '⚠️' if points > -3 else '❌'} {label} ({'+' if points > 0 else ''}{points} pts)")

    # Determine intent level
    if score >= 6:
        level = "🔥 HIGH Intent"
        summary = "Strong signals — prioritize this lead immediately."
    elif score >= 2:
        level = "🟡 MID Intent"
        summary = "Some engagement — nurture and follow up consistently."
    elif score >= 0:
        level = "🔵 LOW Intent"
        summary = "Minimal signals — light touch, monitor for change."
    else:
        level = "❌ Very Low / Cold"
        summary = "Negative signals — consider pausing outreach for now."

    reasons_text = "\n".join(matched_reasons) if matched_reasons else "  No strong signals detected."

    return (
        f"**Lead Intent Assessment**\n"
        f"Score: {score} pts → {level}\n\n"
        f"Signals found:\n{reasons_text}\n\n"
        f"Recommendation: {summary}"
    )


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Alex, a Business Development Representative (BDR) assistant at Rocketsteer.

Your job is to help the Rocketsteer sales team manage and understand their leads. You have access to a leads database (retrieved for you automatically) and you use it to answer questions.

**What you can help with:**
- Looking up lead details (name, email, company, notes, status)
- Summarizing what's happened with a lead (calls, emails, follow-ups)
- Telling the team what the next best action might be for a lead
- Assessing a lead's intent to become a customer (when asked)

**What you cannot do:**
- Answer anything outside of the leads data you have been given
- Make up information about a lead that is not in the data
- Help with topics unrelated to the leads and GTM work

**If something is not in your data, say:**
"I only have access to the leads info I've been given — I don't see that detail. You may want to check your CRM or reach out to the team."

Keep your answers short, clear, and actionable. Talk like a helpful teammate, not a robot."""


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------

def make_chat_node(model, retriever):
    def chat_node(state: ChatState):
        print("Thinking...")

        # Get the latest user message
        last_message = state["messages"][-1].content

        # Retrieve relevant lead context from the vector DB
        context = retrieve_context(retriever, last_message)

        # Check if the user is asking about intent — if so, run the tool
        intent_keywords = ["intent", "likely to buy", "convert", "customer potential", "hot lead", "qualify"]
        intent_report = ""
        if any(kw in last_message.lower() for kw in intent_keywords) and context:
            print("Using tool...")
            intent_report = f"\n\n---\n{assess_lead_intent(context)}"

        # Build the full prompt with context injected
        context_block = f"\n\n**Relevant Lead Data:**\n{context}" if context else "\n\n**Relevant Lead Data:** No matching lead data found."

        system_with_context = SYSTEM_PROMPT + context_block

        messages = [SystemMessage(content=system_with_context)] + state["messages"]
        response = model.invoke(messages)

        # Append intent report to the response if it was triggered
        if intent_report:
            from langchain_core.messages import AIMessage
            full_content = response.content + intent_report
            response = AIMessage(content=full_content)

        return {"messages": [response]}

    return chat_node


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(model, retriever) -> StateGraph:
    graph = StateGraph(ChatState)
    graph.add_node("chat", make_chat_node(model, retriever))
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Chat runner
# ---------------------------------------------------------------------------

def run_chat(graph):
    print("Rocketsteer BDR Assistant (Alex) — type 'quit' to exit.\n")
    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            print("Bye!")
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))
        result = graph.invoke({"messages": history})

        # Keep full conversation history for multi-turn memory
        history = result["messages"]

        assistant_reply = history[-1].content
        print(f"\nAlex: {assistant_reply}\n")