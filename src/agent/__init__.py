"""The answering layer: prompts, a swappable LLM provider, tools, and the graph.

Two entry points, both kept callable so the ablation can compare them:

* `answer_question` — plain RAG. Retrieve once, answer once, no checking.
* `run_agent`       — the LangGraph loop. Retrieve, answer, reflect, and retry
                      retrieval when the answer is not grounded.
"""

from .graph import MAX_ATTEMPTS, build_graph, run_agent, should_retry
from .llm import (
    PROVIDERS,
    AnthropicProvider,
    EchoProvider,
    LLMError,
    LLMResponse,
    LocalProvider,
    Provider,
    get_provider,
    has_credentials,
)
from .prompts import REFLECTION_PROMPT, SYSTEM_PROMPT, build_user_prompt, format_context
from .qa import Answer, answer_question
from .state import AgentResult, AgentState, Critique
from .tools import TOOLS, FileSlice, TestRun, detect_language, read_file, run_tests, search_code

__all__ = [
    "MAX_ATTEMPTS",
    "PROVIDERS",
    "REFLECTION_PROMPT",
    "SYSTEM_PROMPT",
    "TOOLS",
    "AgentResult",
    "AgentState",
    "Answer",
    "AnthropicProvider",
    "Critique",
    "EchoProvider",
    "FileSlice",
    "LLMError",
    "LLMResponse",
    "LocalProvider",
    "Provider",
    "TestRun",
    "answer_question",
    "build_graph",
    "build_user_prompt",
    "detect_language",
    "format_context",
    "get_provider",
    "has_credentials",
    "read_file",
    "run_agent",
    "run_tests",
    "search_code",
    "should_retry",
]
