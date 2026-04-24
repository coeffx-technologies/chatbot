from funcs import get_model, build_retriever, build_graph, run_chat

model = get_model()
retriever = build_retriever()   # loads all PDFs from knowledge/ folder
graph = build_graph(model, retriever)

run_chat(graph)