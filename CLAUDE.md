# Codebase Q&A + PR Review Agent

## What this is
An agent that indexes a code repository (any language, via tree-sitter) and can
(1) answer questions about the codebase and (2) review a diff. Every answer is
grounded in retrieved code, and the agent checks its own work before responding.

## How you (Claude Code) should work with me
I am building this to learn, not just to ship. So after every task you complete:
1. Explain in plain language what you just built and how it fits the pipeline.
2. State the key decisions you made and the alternatives you rejected, one line
   of reasoning each.
3. Flag anything you were unsure about or that I should understand before we move on.
Keep these explanations concise and concrete. No filler, no restating the obvious.

Work one phase at a time. Do not jump ahead to later phases. After each phase,
stop so I can review, run it, and commit before we continue.

If a request would break the scope rules below, say so instead of doing it.

## Tech stack
- Python 3.11
- tree-sitter (via tree_sitter_languages) for AST-based code chunking
- sentence-transformers for embeddings
- ChromaDB for the vector store
- rank_bm25 for keyword search
- LangGraph for the agent state machine
- FastAPI + uvicorn for serving (streaming)
- pytest for tests
- Docker for deployment

## Target directory layout
```
src/
  ingest/    cloning, file walking, tree-sitter chunking
  index/     embeddings, chroma store, bm25 index
  retrieve/  hybrid retrieval + RRF fusion
  agent/     langgraph graph, nodes, tools, prompts
  review/    diff / PR review mode
  eval/      gold set, metrics, ablation runner
  serve/     fastapi app
  config.py  all model names, paths, and default params in one place
tests/
```

## Conventions
- Type hints on every function signature. Short docstrings that say what and why.
- Small, single-purpose functions. No premature abstraction or config frameworks.
- Every chunk carries its metadata: file path, symbol name, start line, end line.
- All tunable values (model names, top_k, chunk size limits, RRF k) live in
  config.py, never hard-coded across files.
- Write a couple of tests per module as you build it, not all at the end.
- Keep the vector-only, bm25-only, and hybrid retrievers separately callable. I
  need them independent for the ablation study.
- Keep everything language-agnostic through a LANGUAGE_REGISTRY in config.py that
  maps a file extension to its tree-sitter language name, the AST node types that
  count as functions and classes, and its test command. Chunking and run_tests
  read from this registry. Adding a new language must mean adding one registry
  entry, nothing else.

## Scope discipline (do not violate without telling me first)
- Build language-agnostic via the registry from the start, but implement and get
  the eval green on Python FIRST. Add JavaScript/TypeScript as the second
  language only once Python passes. Do not try to support every language at once,
  that is how the week gets eaten. Two solid languages proves it generalizes.
- Do not build the web frontend until the API and the eval both work. A CLI is
  the interface until then.
- Do not gold-plate. Prefer the simplest thing that passes the eval.
- run_tests executes repo code. Note in comments where sandboxing would be
  required in production, but a plain subprocess call is acceptable here.

## Definition of done for the whole project
Hybrid retrieval plus a working reflection loop, an eval with recall@k and answer
accuracy, ablations showing hybrid beats pure vector and reflection beats no
reflection, a deployed streaming FastAPI service, and a README with the numbers.
