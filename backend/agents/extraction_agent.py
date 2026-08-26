def extraction_agent(text: str):
    """
    Extraction agent entry point.

    The actual orchestration is handled
    by LangGraph.
    """

    return {
        "text": text
    }