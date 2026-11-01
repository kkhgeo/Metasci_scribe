"""Reflection tool for strategic decision-making."""

from langchain_core.tools import tool


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for structured reflection and strategic decision-making.

    Use this tool to pause and reason carefully at any decision point — not just
    after searches, but before, during, and after any significant step. This creates
    a deliberate checkpoint for quality thinking.

    When to use:
    - Before starting work: What do I know? What skills and prior knowledge are available?
    - After reading sources: What did I learn? Does this change the draft?
    - When choosing between options: What are the trade-offs? Which reading is strongest?
    - When stuck: What is missing? Is there a skill that handles it?
    - Before concluding: Is the evidence sufficient? What does the author need from me?

    Your reflection should address the relevant dimensions below:

    1. Progress — What has been accomplished? What concrete steps remain?
    2. Evidence quality — Do the project's sources support each claim?
       Would a critical reviewer accept it, or does a gap need `[evidence needed]`?
    3. Skills leverage — Is there an installed skill that provides a structured
       workflow for what I'm doing? Check your available skills listing and read
       the relevant `SKILL.md` for full instructions. Follow a skill's workflow
       rather than improvising when one is available.
    4. Prior knowledge — Is there relevant memory, prior work, or a
       skill-provided workflow that should change the plan? Use the memory
       guidance present in the system prompt when memory is enabled; otherwise
       avoid inventing remembered facts.
    5. Strategy — Should I continue the current approach, adjust it, or try
       something different? What evidence supports this decision?
    6. Handoff — Is this step complete? What text, citations and open markers
       does the author or the caller need? Am I leaving clear, well-organized output?
    7. Resource limits — Shell commands have a 300s timeout and a 100KB output
       limit. If a task would exceed them, do not work around it; tell the author
       what you could not do.

    Not every reflection needs all seven dimensions. Pick the ones relevant to
    the current moment. A focused two or three dimension reflection is better
    than a shallow pass over all seven.

    Args:
        reflection: Your structured reflection addressing the relevant dimensions above

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
