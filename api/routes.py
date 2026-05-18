from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage
from api.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    graph = request.app.state.graph
    sessions = request.app.state.sessions  # in-memory session store

    # Load existing history for this session, or start fresh
    history = sessions.get(req.session_id, [])

    # Add the new user message
    history.append(HumanMessage(content=req.message))

    # Run the graph
    result = graph.invoke({"messages": history})

    # Save updated history back to session store
    sessions[req.session_id] = result["messages"]

    # Last message in history is Alex's reply
    reply = result["messages"][-1].content

    return ChatResponse(session_id=req.session_id, reply=reply)