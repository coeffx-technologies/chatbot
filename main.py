from fastapi import FastAPI
from api.routes import router
from llm_funcs import get_model, build_retriever, build_graph

app = FastAPI(title="Rocketsteer BDR Assistant")


@app.on_event("startup")
def startup():
    model = get_model()
    retriever = build_retriever()
    app.state.graph = build_graph(model, retriever)
    app.state.sessions = {}   # { session_id: [messages] }
    print("Alex is ready.")

# Register routes under /api
app.include_router(router, prefix="/api")