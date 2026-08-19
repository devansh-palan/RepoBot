# Codebase Q&A + PR Review Agent

An agent that indexes a code repository via tree-sitter, answers questions about it,
and reviews diffs. Every answer is grounded in retrieved code, and the agent checks
its own work before responding.

Status: **complete.** Ingest, hybrid retrieval, plain RAG, the reflection
agent, diff review, the eval harness with an ablation runner, and a streaming
FastAPI service with a web UI.

## Usage

```powershell
# index a repo: builds the vector and BM25 indexes from one ingest.
# Safe to re-run — unchanged chunks hit the embedding cache.
python -m src.index <repo-path>

# retrieve. --retriever is vector | bm25 | hybrid (default) | all
python -m src.retrieve <repo-path> -q "how are gitignored files skipped" -k 5

# show both input rankings and the RRF arithmetic that fused them
python -m src.retrieve <repo-path> -q "parse a jwt" --explain

# ask a question — retrieves the top 8 chunks, then answers from them alone
python -m src.agent <repo-path> -q "how are gitignored files skipped?" --sources

# the reflection agent: retrieve -> answer -> check -> retry if ungrounded
python -m src.agent <repo-path> -q "how are redirects followed?" --trace

# single-shot RAG instead, the baseline the ablation compares against
python -m src.agent <repo-path> -q "how does chunking work?" --plain

# score retrieval against the gold set (no LLM needed)
python -m src.eval --score

# review a diff
git diff main... | python -m src.review . --diff -

# run the four-config ablation
python -m src.eval --ablate --limit 16 --balanced --out data/eval/ablation.md

# serve the API + web UI on http://127.0.0.1:8000
python -m src.serve
```

Streaming endpoint:

```bash
curl -N -s -X POST http://127.0.0.1:8000/ask   -H 'Content-Type: application/json'   -d '{"repo":"data/repos/requests","question":"How are redirects followed?","k":5}'
```

The default provider is a **local model** — the eval makes hundreds of calls
and a hosted model is not worth that. See Providers below.

## Retrieval

Three retrievers over one corpus, each callable on its own so the ablation can
measure them independently:

| | |
|---|---|
| `vector_search` | cosine nearest neighbours from Chroma |
| `bm25_search` | BM25 over code-aware tokens |
| `hybrid_search` | both, merged with Reciprocal Rank Fusion (k=60) |

**Code-aware tokenization.** `parse_jwt` and `parseJwt` both tokenize to
`['parsejwt', 'parse', 'jwt']`, so naming convention stops mattering and a
spaced query reaches either spelling. Acronyms stay whole —
`parseHTTPResponse` → `parse`, `http`, `response`. Multi-part identifiers also
emit the joined form so an exact hit outranks a chunk that merely mentions both
parts. Question words (`how`, `where`, `do`, `we`) are dropped: they are almost
absent from code, so BM25 hands them a high IDF and lets them dominate.

**Why RRF fuses ranks, not scores.** A cosine similarity of 0.72 and a BM25
score of 30.9 share no scale, and normalising them means inventing one. Ranks
are already comparable, so RRF needs no calibration and no assumption about
either score distribution:

```
score(chunk) = Σ  1 / (60 + rank in that retriever)
```

A chunk both retrievers rank mid-table beats one that a single retriever ranks
first — which is the behaviour hybrid retrieval exists for.

Worked example, `"split an identifier like parse_jwt into its parts"`:

```
 #     score  = sum of 1/(k + rank)                      chunk
 1   0.03279  1/(60+1) [bm25 #1] + 1/(60+1) [vector #1]  tokenize.py:25-34 split_identifier
 2   0.03200  1/(60+3) [bm25 #3] + 1/(60+2) [vector #2]  test_retrieve.py:90-107 test_split_identifier
 3   0.03175  1/(60+2) [bm25 #2] + 1/(60+4) [vector #4]  tokenize.py:37-52 tokenize
```

And on `"the ctest command used to run C++ tests"` — the query vector-only
could not answer at all, because the literal `("ctest", ...)` sits past the
512-token window of a long chunk — BM25 ranks the right chunk first and RRF
promotes it to the top. Note the two retrievers share *no* results there, so
RRF degrades to a round-robin interleave. That is correct behaviour with no
agreement to exploit, and it still rescues the answer.

## Supported languages

Chunking is driven entirely by `LANGUAGE_REGISTRY` in [src/config.py](src/config.py).
Each entry names a tree-sitter grammar wheel plus the AST node types that count
as functions, member-owning types, and standalone types.

| Language | Extensions |
|---|---|
| Python | `.py` `.pyi` |
| JavaScript | `.js` `.mjs` `.cjs` `.jsx` |
| TypeScript | `.ts` `.mts` `.cts` |
| TSX | `.tsx` |
| Go | `.go` |
| Rust | `.rs` |
| Java | `.java` |
| C# | `.cs` |
| Ruby | `.rb` `.rake` |
| C | `.c` `.h` |
| C++ | `.cpp` `.cc` `.cxx` `.hpp` `.hh` `.hxx` |
| PHP | `.php` |

Adding a language means installing its grammar wheel and adding a registry
entry — but read the real grammar first. Conventions vary: Go names a type on
the inner `type_spec`, Rust's `impl_item` has no name field, and C hides a
function name under `declarator`.

**Only Python has eval coverage.** The other eleven are verified structurally
(correct symbols, correct line numbers, no dropped lines) but nothing yet
measures how well their chunks retrieve.

## Indexing

Chunks are embedded with `BAAI/bge-small-en-v1.5` (384-dim, 512-token window)
and stored in a persistent Chroma collection configured for cosine distance.
The embedded text is a short context line — file path, language, kind, symbol —
followed by the code, so a question can match a path or a symbol name even when
it shares no identifiers with the body.

Model choice was measured, not assumed. `all-MiniLM-L6-v2` is the same width
and speed but truncates at 256 tokens — roughly 15 lines of Python:

| | MiniLM-L6-v2 (256 tok) | bge-small-en-v1.5 (512 tok) |
|---|---|---|
| chunks truncated | 22 / 178 (12%) | 5 / 178 (3%) |
| tokens reaching the encoder | 81% | 88% |
| functions truncated | 15 / 125 | 0 / 125 |

The remaining five are module/gap chunks, whose opening lines do not summarise
their tail. Capping gap chunk size is the next fix; BM25 should cover the rest.
Similarity scores are **not** comparable across models — bge clusters higher
than MiniLM on identical rankings, so only the eval can say which retrieves
better.

Swapping `EMBEDDING_MODEL` is safe: the cache key includes the model name, and
the collection is stamped with the model that built it and rebuilt on mismatch.
Two models can share a dimension without sharing a vector space, and Chroma
would accept the mixture silently.

Vectors are cached under `data/embeddings/` keyed on
`sha256(model_name + "\0" + embedded_text)`. Keying on content rather than file
mtime means a touched-but-unchanged file is still a cache hit, a moved function
keeps its vector, and swapping `EMBEDDING_MODEL` invalidates everything.
Re-indexing this repo with a warm cache takes ~1.5 s and never loads torch.

`src.index.search` is the **vector-only** retriever and is callable on its own,
so the ablation can measure it against BM25 and fusion independently.

## Q&A

`answer_question` is plain RAG — hybrid-retrieve, prompt, answer — and nothing
more. It is the baseline the reflection agent has to beat, so it stays
separately callable for the ablation.

The system prompt does two jobs: it forbids outside knowledge, and it makes
citation the default. Each excerpt is labelled with the exact `path:start-end`
string the model is asked to cite, so citing correctly is copying rather than
constructing.

`Answer` carries the retrieved chunks alongside the text, which makes two
things measurable without an LLM judge:

| | |
|---|---|
| `unsupported_citations` | cited locations the model was never shown — hallucination |
| `uncited_sources` | retrieved chunks the answer never used — retrieval noise |

Retrieval returning nothing short-circuits before the LLM call. There is no
grounded answer to give, and asking anyway invites the model to fall back on
general knowledge — the one thing the prompt forbids.

## Providers

The LLM sits behind a one-method wrapper (`src/agent/llm.py`). Nothing above it
imports a vendor SDK, and each provider owns its own model name, so switching is
a `--provider` flag rather than an edit.

| `--provider` | What it is | Cost |
|---|---|---|
| `local` (default) | any OpenAI-compatible `/v1` endpoint — Ollama, LM Studio, llama.cpp, vLLM | free |
| `anthropic` | the Claude API via the official SDK | per token |
| `echo` | returns the prompt instead of an answer | free, no model |

`echo` is not a test double — it is a real provider that makes the whole
pipeline runnable and inspectable with no model at all, which is the fastest way
to debug a prompt or a retrieval problem.

### Running locally

```powershell
ollama pull qwen2.5-coder:7b
python -m src.agent <repo-path> -q "how does chunking work?"
```

`qwen2.5-coder:7b` scores 88.4% on HumanEval and is still the strongest model in
its size class. Quantized it is ~4.7 GB, so on a 4 GB GPU some layers spill to
system RAM. `qwen2.5-coder:3b` fits a 4 GB GPU whole and is several times
faster — the right choice while iterating, with the 7b kept for scoring runs.

Self-hosting a frontier open model is not on the table at this scale: Kimi K3 is
2.8 T parameters and 1.56 TB of weights, needing 8x H100 minimum. If you want
that quality cheaply, point `LOCAL_BASE_URL` at a hosted OpenAI-compatible
endpoint and set `LOCAL_API_KEY` — the same provider class handles it.

## Layout

```
src/
  ingest/    cloning, file walking, tree-sitter chunking
  index/     embeddings, chroma store, bm25 index
  retrieve/  hybrid retrieval + RRF fusion
  agent/     langgraph graph, nodes, tools, prompts
  review/    diff / PR review mode
  eval/      gold set, metrics, ablation runner
  serve/     fastapi app
  config.py  all model names, paths, and default params
tests/
```

All tunable values live in [src/config.py](src/config.py). Language support is
driven by `LANGUAGE_REGISTRY` there — adding a language means adding one entry.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set `ANTHROPIC_API_KEY` in your environment before running the agent.

## Health check

Imports every module under `src/` to catch broken imports early:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_health.py -q
```

## Service

`POST /ask` streams Server-Sent Events: `meta`, then `token` (or `stage` in
agent mode), `sources`, and a final `done` carrying the answer and metrics.
`POST /review` returns structured comments. `GET /` is a dependency-free web UI.

Everything under the endpoints is synchronous and blocking, so it runs in a
worker thread via `asyncio.to_thread`; tokens arrive on a sync callback from
that thread and are handed to the event loop through an `asyncio.Queue`.

Per-request metrics separate **time to first token** from generation time,
because they are fixed in different places. A measured local run:

| | |
|---|---|
| time to first token | 8,653 ms |
| generation | 6,849 ms |
| total | 15,502 ms |
| throughput | 12.7 tok/s |

56% of the wall clock is retrieval plus model prefill. A single `total_ms`
would have pointed at the model, which is the wrong thing to optimise.

## Results

See [data/eval/ablation.md](data/eval/ablation.md), regenerated with
`python -m src.eval --ablate`.
