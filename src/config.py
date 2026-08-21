"""Single source of truth for model names, paths, and tunable defaults.

Nothing here should be duplicated elsewhere in the codebase. If a number needs
tuning for the ablation study, it belongs in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

REPOS_DIR: Path = DATA_DIR / "repos"          # cloned repositories under test
CHROMA_DIR: Path = DATA_DIR / "chroma"        # persistent vector store
BM25_DIR: Path = DATA_DIR / "bm25"            # pickled keyword indexes
EVAL_DIR: Path = DATA_DIR / "eval"            # gold set + run outputs
EMBEDDING_CACHE_DIR: Path = DATA_DIR / "embeddings"  # content-hash -> vector

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

# 768-dim, ~440 MB, 512-token window. bge-base over bge-small on measurement:
# R@8 — the k the agent actually consumes — improved on every gold slice
# (dev 0.767 -> 0.822, held-out test 0.813 -> 0.833, test-semantic, the
# weakest slice, 0.558 -> 0.608) at the cost of a small test MRR dip
# (0.821 -> 0.806) and ~3x slower indexing. The model reads all 8 chunks, so
# more right chunks in the window beats slightly better ordering of fewer.
# (bge-small was itself chosen over all-MiniLM-L6-v2, whose 256-token window
# truncated 12% of chunks.) Changing this constant is safe: the embedding
# cache and the Chroma collection are both keyed on the model name and rebuild
# themselves — but every indexed repo must be re-indexed before it can answer.
EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM: int = 768

# Instruction prefix applied to the *query* side only. bge and e5 are trained
# with one and lose retrieval quality without it; MiniLM was not, and wants "".
EMBEDDING_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

# NOTE: 512 tokens is still under MAX_CHUNK_CHARS, so the longest module/gap
# chunks remain partly invisible to the vector retriever. Capping gap chunks is
# the next fix; BM25 in phase 3 should cover the rest.
EMBEDDING_BATCH_SIZE: int = 64

# The most characters of any one chunk rendered into the model's prompt. A
# chunk can be MAX_CHUNK_CHARS (8000) on disk, but prefill is over half of
# time-to-first-token on a CPU-bound model, and one oversized module chunk can
# double the prompt on its own. 3000 chars ≈ 60-80 lines — the head of a chunk
# carries its signature and docstring, which is what grounding needs.
PROMPT_CHUNK_CHAR_CAP: int = 3_000

# Agent LLM. Answering and reviewing need the strongest model; the reflection
# critic is a cheaper judgement call and can run on a smaller one.
ANSWER_MODEL: str = "claude-opus-5"
REFLECTION_MODEL: str = "claude-sonnet-5"
LLM_MAX_TOKENS: int = 16_000
LLM_EFFORT: str = "high"  # low | medium | high | xhigh | max

# Which llm.py provider to use.
#   "local"     -> an OpenAI-compatible server on this machine (Ollama, LM
#                  Studio, llama.cpp, vLLM). The default: the eval makes
#                  hundreds of calls and a hosted model is not worth that.
#   "anthropic" -> the Claude API, for a final high-quality run.
#   "echo"      -> returns the prompt instead of an answer; no key, no network.
LLM_PROVIDER: str = "local"

# Any OpenAI-compatible /v1 endpoint. Ollama is the default; LM Studio uses
# http://localhost:1234/v1 and llama.cpp's server http://localhost:8080/v1.
# A *hosted* OpenAI-compatible endpoint works here too — set LOCAL_API_KEY and
# point this at it. That is the only practical route to a frontier open model:
# Kimi K3 is 2.8T parameters and 1.56 TB of weights, needing 8x H100 to self-host.
LOCAL_BASE_URL: str = "http://localhost:11434/v1"

# Read from the environment so a key never lands in the repo. Left empty for a
# local server, which needs no auth.
LOCAL_API_KEY_ENV: str = "LOCAL_API_KEY"

# qwen2.5-coder 7B: 88.4% HumanEval and still the strongest model in its class.
# Quantised it is ~4.7 GB, so on a 4 GB GPU some layers spill to system RAM —
# slower than the 3b, but grounded citation is exactly where the larger model
# earns its keep, and that is what the eval measures. Swap to "qwen2.5-coder:3b"
# while iterating; it fits a 4 GB GPU whole and runs several times faster.
LOCAL_MODEL: str = "qwen2.5-coder:7b"

# Local generation is slow and has no per-token billing to protect, so the
# timeout is generous compared with a hosted call.
LOCAL_TIMEOUT_SECONDS: int = 600

# NOTE: Ollama derives a default context length from available VRAM — on a 4 GB
# card that is 4096 tokens. Measured on this repo, a FINAL_TOP_K=8 prompt runs
# 1800-2600 tokens, so it fits but without much headroom; a question that
# retrieves one of the large module/gap chunks could overflow it, and an
# overflow truncates the excerpts silently, which breaks grounding without any
# error. Raise the server's limit (OLLAMA_CONTEXT_LENGTH=8192) or lower
# FINAL_TOP_K if answers start ignoring the code they were given.

# --------------------------------------------------------------------------
# Ingest / chunking
# --------------------------------------------------------------------------

# A chunk is one function or class body. These bound the pathological case:
# generated files with 2000-line functions. There is deliberately no minimum —
# a one-line function is still a real symbol, and dropping small chunks would
# mean losing code.
MAX_CHUNK_LINES: int = 200
MAX_CHUNK_CHARS: int = 8_000

# When a symbol exceeds MAX_CHUNK_LINES it is split into windows with overlap
# so a construct straddling the boundary still appears whole in one chunk.
CHUNK_OVERLAP_LINES: int = 20

MAX_FILE_BYTES: int = 1_000_000  # skip anything larger; almost always generated

# Documentation indexed alongside code. "What is this repo about?" is answered
# by the README, not by any function body — an index without prose cannot
# ground overview questions, and the agent's citation check then rejects every
# attempt at them. Markdown is split on headings so citations point at a
# section; other formats fall back to fixed windows.
DOC_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".rst", ".txt"})

# Which doc files qualify: repo-root docs plus README-style names anywhere.
# NOT whole docs/ trees — measured on the gold set, indexing click's docs/
# added ~950 prose chunks that describe code behaviour in exactly the words a
# semantic query uses, and held-out semantic MRR fell 0.649 -> 0.564 as prose
# outranked the code it describes. READMEs answer the overview questions;
# narrative documentation mostly duplicates the code and fights it.
DOC_STEM_PATTERN: str = r"^(readme|changelog|changes|contributing|install|usage|faq|news|history|license|authors)"

IGNORE_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "vendor", "dist", "build", "target",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env",
    ".tox", ".eggs", "site-packages",
})

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

VECTOR_TOP_K: int = 20  # candidates from the vector retriever alone
BM25_TOP_K: int = 20    # candidates from the keyword retriever alone
FINAL_TOP_K: int = 8    # chunks handed to the agent after fusion

# Reciprocal Rank Fusion: score = sum(1 / (RRF_K + rank)).
#
# NOT 60. The Cormack et al. default assumes TREC runs ~1000 deep; our pools are
# VECTOR_TOP_K/BM25_TOP_K = 20. At k=60 the rank term only spans 1/61..1/80 — a
# 1.3x spread — while appearing in both lists doubles the score, so fusion
# degenerates into counting votes and ignoring rank: a chunk both retrievers put
# last outscored one vector put first. k on the order of the pool depth keeps
# rank meaningful. Measured on the gold set, k=60 -> 10 moved hybrid R@8 from
# 0.772 to 0.783; k=1 scored higher still (0.806) but that is fitting the last
# decimal on 30 questions.
RRF_K: int = 10

# BM25's weight in the fusion depends on the *query*, not the corpus. Measured
# on the held-out test split: on queries that quote a code identifier
# ("where is parse_jwt defined") BM25 at full weight lifts MRR 0.866 -> 0.939
# over pure vector, while on natural-language queries ("how are redirects
# resolved") it drags MRR 0.649 -> 0.493 — question words match docstrings, not
# the code that answers them. Every fixed weight in between loses one side or
# the other, so hybrid_search switches on whether the query names an
# identifier. 0.0 means BM25 sits the query out entirely; that is what the
# sweep said, not a shortcut.
RRF_BM25_WEIGHT_LEXICAL: float = 1.0
RRF_BM25_WEIGHT_SEMANTIC: float = 0.0

# BM25 term saturation and length normalisation.
#
# BM25_B is 0.1, not the 0.75 default. b controls how much a short document is
# rewarded, and 0.75 is tuned for prose. Our chunker emits class *headers* as
# their own short chunks (signature + docstring, methods chunked separately),
# which is right for citations but hands BM25 a pile of stubs that outscore real
# implementations: on "where is the Basic auth header built", the HTTPBasicAuth
# and HTTPProxyAuth class headers both beat _basic_auth_str, the actual answer.
# Measured on the gold set, 0.75 -> 0.1 moved BM25 R@8 0.544 -> 0.656 and
# MRR 0.376 -> 0.440.
BM25_K1: float = 1.5
BM25_B: float = 0.1

# Dropped from both documents and queries. Question words are the point: "how",
# "where", "do", "we" are near-absent from code, so BM25 hands them a high IDF
# and lets them dominate a natural-language query. Code keywords like `for` and
# `in` are here too — they are so common in source that they carry no signal.
BM25_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by",
    "can", "did", "do", "does", "for", "from", "get", "had", "has", "have",
    "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "not",
    "of", "on", "or", "our", "out", "should", "so", "some", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "to",
    "up", "us", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
})

# Cross-encoder reranking — available as the "hybrid_rerank" retriever but NOT
# the default, because it lost on measurement. A cross-encoder reads question
# and chunk together and should beat separate embeddings, but both general-
# purpose rerankers *regressed* on code (held-out test MRR vs hybrid's 0.821):
#   cross-encoder/ms-marco-MiniLM-L-6-v2  0.792  (+0.5s/query)
#   BAAI/bge-reranker-base                0.730  (+7.5s/query on CPU)
# They are trained on web prose; retrieval-tuned code embeddings plus BM25
# already encode more of what a code query needs. Revisit only with a
# code-trained reranker, and re-run `python -m src.eval --score` before
# believing it.
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES: int = 30  # pool handed to the cross-encoder

# Chroma rejects oversized writes, so upserts and deletes are chunked.
CHROMA_BATCH_SIZE: int = 1_000

# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

MAX_REFLECTION_LOOPS: int = 2   # critic passes before answering regardless
MAX_AGENT_STEPS: int = 12       # hard stop on the langgraph state machine

# --------------------------------------------------------------------------
# run_tests
# --------------------------------------------------------------------------

# NOTE: run_tests executes code from the repository under review. A plain
# subprocess call is acceptable for this project; in production this must run
# in a sandbox (container, seccomp, no network, read-only mounts).
TEST_TIMEOUT_SECONDS: int = 300

# --------------------------------------------------------------------------
# Cloning
# --------------------------------------------------------------------------

# Shallow clones only: Q&A needs the working tree, not the history, and the
# difference on a large repo is seconds versus minutes.
CLONE_TIMEOUT_SECONDS: int = 300

# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------

API_HOST: str = "127.0.0.1"
API_PORT: int = 8000

# --------------------------------------------------------------------------
# Language registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LanguageSpec:
    """Everything language-specific the pipeline needs, in one place.

    Chunking reads the node-type tuples; run_tests reads `test_command`. Adding
    a language means installing its tree-sitter grammar wheel and adding
    registry entries pointing at a new LanguageSpec — no changes anywhere else.

    Every node tuple below was read off the real grammar, not guessed. The
    conventions differ more than you would expect: Go names a type on the inner
    `type_spec` rather than the `type_declaration`, Rust's `impl_item` has no
    name field at all, and C hides a function's name under `declarator`.
    """

    name: str                          # human-readable label
    grammar_module: str                # module shipping the compiled grammar
    function_nodes: tuple[str, ...]    # callable definitions
    class_nodes: tuple[str, ...]       # types that own members; recursed into
    test_command: tuple[str, ...]      # argv, run without a shell

    # Types that do not own methods (interfaces, enums, plain structs). Emitted
    # whole rather than recursed into.
    type_nodes: tuple[str, ...] = ()

    # Nodes that enclose a definition and belong in its chunk: decorators,
    # `export`, `template<...>`, Go's `type (...)` block. Unwrapped in a chain,
    # so `export const f = () => {}` reaches the arrow function three levels in.
    wrapper_nodes: tuple[str, ...] = ()

    # Fields tried in order when resolving a symbol's name. "declarator" is
    # followed recursively, which is how C/C++ names are reached.
    name_fields: tuple[str, ...] = ("name",)

    # Function exposing the grammar. TypeScript and PHP ship several grammars
    # in one wheel and expose none of them as plain `language`.
    grammar_entrypoint: str = "language"

    # Field holding a method's receiver, for languages that declare methods at
    # top level instead of nesting them in their type. Go only, so far.
    receiver_field: str | None = None

    # Cached because the chunker asks for this once per AST node; specs are
    # module-level constants, so nothing is retained that would not be anyway.
    @lru_cache(maxsize=None)
    def symbol_nodes(self) -> frozenset[str]:
        """Every node type that should become a chunk of its own."""
        return frozenset(self.function_nodes + self.class_nodes + self.type_nodes)


PYTHON = LanguageSpec(
    name="Python",
    grammar_module="tree_sitter_python",
    function_nodes=("function_definition",),
    class_nodes=("class_definition",),
    wrapper_nodes=("decorated_definition",),
    test_command=("python", "-m", "pytest", "-q"),
)

# `export`/`lexical_declaration`/`variable_declarator` are wrappers so that
# `export const handler = async () => {}` keeps its declaration and takes its
# name from the declarator — the arrow function itself is anonymous.
_JS_FUNCTIONS = (
    "function_declaration",
    "generator_function_declaration",
    "function_expression",
    "arrow_function",
    "method_definition",
)
_JS_WRAPPERS = (
    "export_statement",
    "lexical_declaration",
    "variable_declaration",
    "variable_declarator",
)

JAVASCRIPT = LanguageSpec(
    name="JavaScript",
    grammar_module="tree_sitter_javascript",
    function_nodes=_JS_FUNCTIONS,
    class_nodes=("class_declaration", "class"),
    wrapper_nodes=_JS_WRAPPERS,
    test_command=("npm", "test", "--silent"),
)

_TS_TYPES = ("interface_declaration", "type_alias_declaration", "enum_declaration")

TYPESCRIPT = LanguageSpec(
    name="TypeScript",
    grammar_module="tree_sitter_typescript",
    grammar_entrypoint="language_typescript",
    function_nodes=_JS_FUNCTIONS,
    class_nodes=("class_declaration", "abstract_class_declaration", "class"),
    type_nodes=_TS_TYPES,
    wrapper_nodes=_JS_WRAPPERS,
    test_command=("npm", "test", "--silent"),
)

# TSX is a genuinely separate grammar, not a flag on the TypeScript one.
TSX = LanguageSpec(
    name="TSX",
    grammar_module="tree_sitter_typescript",
    grammar_entrypoint="language_tsx",
    function_nodes=_JS_FUNCTIONS,
    class_nodes=("class_declaration", "abstract_class_declaration", "class"),
    type_nodes=_TS_TYPES,
    wrapper_nodes=_JS_WRAPPERS,
    test_command=("npm", "test", "--silent"),
)

# Go declares methods at top level with a receiver rather than nesting them in
# their type, so `receiver_field` is what recovers the `Account.Interest`
# qualification the other languages get from nesting.
GO = LanguageSpec(
    name="Go",
    grammar_module="tree_sitter_go",
    function_nodes=("function_declaration", "method_declaration"),
    class_nodes=(),
    type_nodes=("type_spec",),
    wrapper_nodes=("type_declaration",),
    receiver_field="receiver",
    test_command=("go", "test", "./..."),
)

# `impl_item` carries the type it implements under `type`, not `name`.
RUST = LanguageSpec(
    name="Rust",
    grammar_module="tree_sitter_rust",
    function_nodes=("function_item", "function_signature_item"),
    class_nodes=("impl_item", "trait_item", "mod_item"),
    type_nodes=("struct_item", "enum_item", "union_item", "type_item"),
    name_fields=("name", "type"),
    test_command=("cargo", "test"),
)

JAVA = LanguageSpec(
    name="Java",
    grammar_module="tree_sitter_java",
    function_nodes=("method_declaration", "constructor_declaration"),
    class_nodes=("class_declaration", "interface_declaration", "record_declaration"),
    type_nodes=("enum_declaration", "annotation_type_declaration"),
    test_command=("mvn", "-q", "-B", "test"),
)

CSHARP = LanguageSpec(
    name="C#",
    grammar_module="tree_sitter_c_sharp",
    function_nodes=("method_declaration", "constructor_declaration"),
    class_nodes=(
        "namespace_declaration",
        "class_declaration",
        "interface_declaration",
        "struct_declaration",
        "record_declaration",
    ),
    type_nodes=("enum_declaration", "delegate_declaration"),
    test_command=("dotnet", "test"),
)

RUBY = LanguageSpec(
    name="Ruby",
    grammar_module="tree_sitter_ruby",
    function_nodes=("method", "singleton_method"),
    class_nodes=("class", "module", "singleton_class"),
    test_command=("bundle", "exec", "rspec"),
)

# C and C++ name a function through a `declarator` chain, never a `name` field.
C = LanguageSpec(
    name="C",
    grammar_module="tree_sitter_c",
    function_nodes=("function_definition",),
    class_nodes=(),
    type_nodes=("struct_specifier", "union_specifier", "enum_specifier", "type_definition"),
    name_fields=("name", "declarator"),
    test_command=("ctest", "--output-on-failure"),
)

CPP = LanguageSpec(
    name="C++",
    grammar_module="tree_sitter_cpp",
    function_nodes=("function_definition",),
    class_nodes=("class_specifier", "namespace_definition"),
    type_nodes=("struct_specifier", "union_specifier", "enum_specifier", "type_definition"),
    wrapper_nodes=("template_declaration",),
    name_fields=("name", "declarator"),
    test_command=("ctest", "--output-on-failure"),
)

PHP = LanguageSpec(
    name="PHP",
    grammar_module="tree_sitter_php",
    grammar_entrypoint="language_php",
    function_nodes=("function_definition", "method_declaration"),
    class_nodes=("class_declaration", "trait_declaration", "interface_declaration"),
    type_nodes=("enum_declaration",),
    test_command=("vendor/bin/phpunit",),
)

# Maps file extension -> spec. Several extensions may share one spec.
# `.h` is ambiguous between C and C++; it is treated as C, whose grammar parses
# the common subset. A C++-only header still chunks, just with C node names.
LANGUAGE_REGISTRY: dict[str, LanguageSpec] = {
    ".py": PYTHON,
    ".pyi": PYTHON,
    ".js": JAVASCRIPT,
    ".mjs": JAVASCRIPT,
    ".cjs": JAVASCRIPT,
    ".jsx": JAVASCRIPT,
    ".ts": TYPESCRIPT,
    ".mts": TYPESCRIPT,
    ".cts": TYPESCRIPT,
    ".tsx": TSX,
    ".go": GO,
    ".rs": RUST,
    ".java": JAVA,
    ".cs": CSHARP,
    ".rb": RUBY,
    ".rake": RUBY,
    ".c": C,
    ".h": C,
    ".cpp": CPP,
    ".cc": CPP,
    ".cxx": CPP,
    ".hpp": CPP,
    ".hh": CPP,
    ".hxx": CPP,
    ".php": PHP,
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(LANGUAGE_REGISTRY)


def spec_for_path(path: str | Path) -> LanguageSpec | None:
    """Return the LanguageSpec for a file, or None if the language is unsupported."""
    return LANGUAGE_REGISTRY.get(Path(path).suffix.lower())
