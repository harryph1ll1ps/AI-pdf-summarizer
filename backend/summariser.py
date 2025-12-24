import ollama 
from typing import List

SUMMARY_MODEL = "llama3.2:3b"

def summarise_chunk(text: str) -> str:
    prompt = f"""
    Summarise the following document section.
    Focus on key points (facts, rules, procedures).
    Use concise bullet points.

    Text:
    {text}

    Summary:
    """.strip()

    resp = ollama.chat(
        model=SUMMARY_MODEL,
        messages=[{"role":"user", "content": prompt}],
        options={"temperature": 0.2, "num_predict": 150}
        )

    return resp["message"]["content"].strip()


def summarise_doc(chunks: List[str]) -> str:
    partial_summaries = []
    for chunk in chunks:
        partial = summarise_chunk(chunk)
        partial_summaries.append(partial)

    combined = "\n".join(partial_summaries)

    final_prompt = f"""
    You are producing an executive summary of a document.

    Combine the ofllowing section summaries into a clear, concise, structured overview.

    The executive summary should not exceed 120 words.

    Summaries:
    {combined}

    Final Summary:
    """.strip()

    resp = ollama.chat(
        model=SUMMARY_MODEL,
        messages=[{"role":"user", "content": final_prompt}],
        options={"temperature": 0.2, "num_predict": 250}
        )
    
    return resp["message"]["content"].strip()