import os
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Annotated
from typing_extensions import TypedDict

load_dotenv()


# --- State ---

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# --- Setup ---

def load_knowledge(path: str = "knowledge.json") -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2)


def get_model() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key,
    )


def build_system_prompt(knowledge: str) -> str:
    return f"""You are the Rocketsteer GTM assistant. You help users understand what Rocketsteer does, how whoismyprospect? works, and any other questions about the platform.

**Knowledge Base:**
{knowledge}

**Rules:**
- Only answer questions using the knowledge base above
- If something is not covered in the knowledge base, say: "I don't have that information — you can reach out to the Rocketsteer team for more details."
- Keep answers clear, direct, and concise
- Do not make up features, pricing, or claims not in the knowledge base
- Sound helpful and professional, not robotic"""


# --- Graph node ---

def make_chat_node(model, system_prompt: str):
    def chat_node(state: ChatState):
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}
    return chat_node


# --- Graph builder ---

def build_graph(model, system_prompt: str) -> StateGraph:
    graph = StateGraph(ChatState)
    graph.add_node("chat", make_chat_node(model, system_prompt))
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile()


# --- Chat runner ---

def run_chat(graph):
    print("Rocketsteer GTM Assistant — type 'quit' to exit.\n")
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

        # update history with full state (includes assistant reply)
        history = result["messages"]

        assistant_reply = history[-1].content
        print(f"\nAssistant: {assistant_reply}\n")
