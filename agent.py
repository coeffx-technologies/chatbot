from funcs import load_knowledge, get_model, build_system_prompt, build_graph, run_chat

knowledge = load_knowledge("knowledge.json")
model = get_model()
system_prompt = build_system_prompt(knowledge)
graph = build_graph(model, system_prompt)

run_chat(graph)
