def build_grounded_prompt(question: str, context: str) -> str:
    """Build a conservative answer prompt over already-approved evidence."""
    clean_question = question.strip()
    clean_context = context.strip()
    if not clean_question:
        raise ValueError("Question cannot be empty")
    if not clean_context:
        raise ValueError("Evidence context cannot be empty")

    return (
        "Answer the question using only the supplied evidence.\n"
        "Do not use outside knowledge or invent legal conclusions.\n"
        "Cite every substantive factual claim with the matching marker such as [1].\n"
        "The opening answer and any transition or list-introduction claim must also "
        "include citations; do not state an uncited yes/no conclusion.\n"
        "For multiple sources, use separate markers such as [2][3]; never combine "
        "citation numbers inside one marker.\n"
        "If the evidence does not establish a definitive yes or no, explain the "
        "conditions and limitations shown by the evidence.\n\n"
        f"QUESTION:\n{clean_question}\n\n"
        f"EVIDENCE:\n{clean_context}\n\n"
        "ANSWER:"
    )
