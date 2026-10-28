
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class RAGState(TypedDict):
    user_id: str
    question: str
    top_k: int
    documents: list[dict]
    results: list[dict]


def retrieve_documents(state: RAGState):
    question_words = set(state["question"].lower().split())
    results = []

    for document in state["documents"]:
        if document["user_id"] != state["user_id"]:
            continue

        text = document["text"]
        document_words = set(text.lower().split())
        matching_words = question_words.intersection(document_words)

        if matching_words:
            results.append({
                "filename": document["filename"],
                "text": text,
                "matched_terms": len(matching_words),
            })

    results.sort(
        key=lambda item: item["matched_terms"],
        reverse=True,
    )

    return {"results": results[:state["top_k"]]}


builder = StateGraph(RAGState)

builder.add_node("retrieve_documents", retrieve_documents)

builder.add_edge(START, "retrieve_documents")
builder.add_edge("retrieve_documents", END)

rag_graph = builder.compile()
