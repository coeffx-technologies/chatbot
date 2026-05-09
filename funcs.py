import os
import ast
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scrapling.fetchers import StealthyFetcher
import json
import re
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
# Extracting companies names 
# ---------------------------------------------------------------------------
def extract_companies(user_message, model):

    prompt = f"""
You are a company name extractor.

Your job: extract ONLY companies the user explicitly wants researched or wants information about.

Rules:
- INCLUDE a company if the user says: "research", "look up", "get info on", "tell me about", "find out about", or similar intent
- EXCLUDE a company if the user says: "don't research", "skip", "ignore", "I don't like", "not interested in", or expresses negative intent
- EXCLUDE companies mentioned only as context or background ("we used to work with X", "I heard about X")
- If no companies need researching, return an empty list

Examples:
- "research Google and Microsoft but skip Apple" → ["Google", "Microsoft"]
- "I don't like Tesla but look into Rivian" → ["Rivian"]
- "we lost a deal with Salesforce, research HubSpot instead" → ["HubSpot"]
- "tell me about our leads" → []
- "research Google" → ["Google"]

Return ONLY a valid Python list, nothing else. No explanation, no markdown.

Text: {user_message}
"""

    try:
        return ast.literal_eval(
            model.invoke(prompt).content.strip().lower()
        )
    except:
        return []

def linkedin_fetch(company_name):

    StealthyFetcher.adaptive = True

    # --- dismiss login modal if it pops up ---
    def after_load(page):
        page.wait_for_timeout(5000)

        modal = page.locator("div#base-contextual-sign-in-modal div.modal__overlay-visible")
        try:
            modal.first.wait_for(state="visible", timeout=10000)
        except Exception:
            pass

        dismiss = page.locator("button[data-tracking-control-name='organization_guest_contextual-sign-in-modal_modal_dismiss']")
        try:
            dismiss.first.wait_for(state="attached", timeout=5000)
            dismiss.first.click(force=True)
            page.wait_for_timeout(3000)
        except Exception:
            pass

    page = StealthyFetcher.fetch(
        f"https://www.linkedin.com/company/{company_name}/",
        headless=True,
        network_idle=True,
        page_action=after_load,
    )

    # --- helpers ---
    def text(selector):
        el = page.css(selector)
        return el[0].text.strip() if el else None

    def attr(selector, attribute):
        el = page.css(selector)
        return el[0].attrib.get(attribute) if el else None

    # --- dt/dd key-value pairs (industry, hq, founded, type, specialties, company size) ---
    details = {}
    for dt, dd in zip(page.css("dt"), page.css("dd")):
        key = dt.text.strip().lower()
        val = dd.text.strip()
        if key:
            details[key] = val

    # --- followers: parse from og:description meta tag ---
    tagline_raw = attr("meta[name='description']", "content")
    followers_match = re.search(r"([\d,]+)\s+followers on LinkedIn", tagline_raw or "")
    linkedin_followers = followers_match.group(1) if followers_match else None

    # --- website: LinkedIn renders it as <a> inside dd, not plain text ---
    website_el = page.css("dd a[data-test-id='about-us__website']") or page.css("dd a[href*='http']")
    website = website_el[0].attrib.get("href") if website_el else None

    return {
        "company_slug":       company_name,
        "company_name":       (attr("meta[property='og:title']", "content") or "").replace(" | LinkedIn", "").strip(),
        "tagline":            tagline_raw,
        "logo_url":           attr("meta[property='og:image']", "content"),
        "description":        text("p[data-test-id='about-us__description']"),
        "linkedin_followers": linkedin_followers,
        "website":            website,
        "industry":           details.get("industry"),
        "hq":                 details.get("headquarters"),
        "founded":            details.get("founded"),
        "company_type":       details.get("type"),
        "employee_count":     details.get("company size"),
        "specialties":        details.get("specialties"),
    }


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

        companies_list = extract_companies(last_message, model)
        print(companies_list)

        companies_info = ""

        if companies_list:
            print("extracting companies info")

            fetched = []

            for i in companies_list:
                company = linkedin_fetch(i)

                if company:
                    fetched.append(json.dumps(company, indent=2))

            if fetched:
                companies_info = (
                    "\n\n**Companies Info from LinkedIn:**\n"
                    + "\n".join(fetched)
                )

        # Retrieve relevant lead context from the vector DB
        context = retrieve_context(retriever, last_message)

        # Build the full prompt with context injected
        context_block = (
            f"\n\n**Relevant Lead Data:**\n{context}"
            if context
            else "\n\n**Relevant Lead Data:** No matching lead data found."
        )

        system_with_context = (
            SYSTEM_PROMPT
            + context_block
            + companies_info
        )

        messages = [SystemMessage(content=system_with_context)] + state["messages"]

        response = model.invoke(messages)

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
        