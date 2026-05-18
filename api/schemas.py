from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str       # unique ID per user/conversation
    message: str          # the user's message


class ChatResponse(BaseModel):
    session_id: str
    reply: str            # Alex's response