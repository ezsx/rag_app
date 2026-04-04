# Benchmarking custom RAG against LlamaIndex objectively

**LlamaIndex is the only credible framework comparison target**, and the benchmark reveals a striking paradox: the framework covers just **4 of 12 pipeline components** out-of-box, requiring ~800 lines of custom code on top of ~85 lines of actual framework code. The benchmark design must include a naive baseline for triangulation, use 100–150 stratified test questions (not 36), and present results as an Architecture Decision Record — framing the comparison as an informed engineering tradeoff rather than framework bashing. With the right methodology, this benchmark becomes a portfolio piece that demonstrates both deep systems expertise and evaluation rigor.

---

## LlamaIndex wins the framework selection decisively

Among all candidates — LangChain (125K stars), Haystack (24K stars), RAGFlow (70K stars), Cognita (8K), Kotaemon (25K) — **LlamaIndex (46.5K stars)** is the only framework that simultaneously satisfies three requirements: RAG-first architecture, strong Qdrant native support, and universal hiring manager recognition.

LangChain is fundamentally an agent orchestration framework, not a RAG framework. Building equivalent hybrid retrieval in LangChain requires substantially more boilerplate than LlamaIndex, where `enable_hybrid=True` on `QdrantVectorStore` is a one-liner. Haystack (by deepset) is the runner-up with excellent `QdrantHybridRetriever` and strong enterprise production readiness, but its agent capabilities are less mature and its community recognition falls below LlamaIndex outside enterprise contexts.

The specialized frameworks fail on recognition: RAGFlow is a Chinese-focused RAG *application* (not a framework for building pipelines), Cognita has only 8K stars and is unknown to hiring managers, and Kotaemon is an end-user UI with no agent support. RAGAS is evaluation-only and should be used alongside the benchmark, not as a pipeline framework. **DSPy** from Stanford is worth noting — it optimizes prompts programmatically — but it complements rather than competes with RAG frameworks.

| Criterion | LlamaIndex | LangChain | Haystack |
|-----------|-----------|-----------|----------|
| RAG-first design | ✅ Core mission | ⚠️ Agent-first | ✅ Pipeline-based |
| Qdrant named vectors | ✅ Native, async | ⚡ Generic interface | ✅ QdrantHybridRetriever |
| Hybrid BM25+dense | ✅ One-liner | ⚠️ Manual combo | ✅ Native with RRF |
| Reranking pipeline | ✅ Node postprocessors | ⚡ Via integrations | ✅ TransformersSimilarityRanker |
| Multi-query decomposition | ✅ SubQuestionQueryEngine | ⚠️ Manual | ⚠️ Custom pipeline needed |
| Agent with function calling | ✅ FunctionAgent, ReActAgent | ✅ Strong agents | ⚡ Newer, less mature |
| Hiring manager recognition | ★★★★★ | ★★★★★ | ★★★★ |

The benchmark narrative becomes compelling: *"I built a 14.5K LOC zero-framework pipeline, then implemented the equivalent in the industry's leading RAG framework to quantify exactly what the framework provides and where custom engineering delivers measurable value."*

---

## Feature parity reveals the framework covers only one-third of the pipeline

The most important finding from this research is the classification of all 12 custom pipeline components against LlamaIndex equivalents. Only **2 components are out-of-box, 2 are configurable, and 8 require custom code** — meaning the framework's actual contribution is concentrated in a narrow slice of the pipeline.

### Components LlamaIndex handles well

**Agent with tools (OUT-OF-BOX):** LlamaIndex's `FunctionCallingAgent` and `ReActAgent` support arbitrary numbers of tools via `FunctionTool` and `QueryEngineTool`. The 15-tool agent maps cleanly — define each tool function, wrap it, and pass to the agent. Streaming is built-in via `AgentStream` events.

```python
from llama_index.core.agent.workflow import FunctionCallingAgent
from llama_index.core.tools import FunctionTool

tools = [FunctionTool.from_defaults(fn=search_documents), ...]  # 15 tools
agent = FunctionCallingAgent(name="assistant", tools=tools, llm=llm)
```

**Cross-encoder reranking (CONFIGURABLE):** `SentenceTransformerRerank` wraps any HuggingFace cross-encoder. For Qwen3-Reranker-0.6B, it's a single line — though compatibility with the CrossEncoder class needs verification.

```python
from llama_index.core.postprocessor import SentenceTransformerRerank
reranker = SentenceTransformerRerank(model="Qwen/Qwen3-Reranker-0.6B", top_n=10)
```

**Hybrid retrieval (CONFIGURABLE with limits):** LlamaIndex supports Qdrant hybrid search with `enable_hybrid=True` and offers `QueryFusionRetriever` with RRF mode. However, **weighted RRF** (BM25 weight=3.0, dense=1.0) has no framework equivalent — standard RRF uses uniform weights with k=60. A custom `BaseRetriever` subclass of ~35 lines implements weighted fusion:

```python
class WeightedRRFHybridRetriever(BaseRetriever):
    def _retrieve(self, query_bundle):
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)  # top-100
        vector_nodes = self.vector_retriever.retrieve(query_bundle)  # top-20
        scores = {}
        for rank, node in enumerate(bm25_nodes):
            scores[node.node.node_id] = self.weights[0] / (self.k + rank + 1)  # w=3.0
        for rank, node in enumerate(vector_nodes):
            scores[node.node.node_id] += self.weights[1] / (self.k + rank + 1)  # w=1.0
        return sorted_top_k(scores)
```

### Components requiring significant custom code

**Seven of twelve components** have no framework equivalent whatsoever. This is the core insight — the pipeline's most distinctive engineering features are invisible to frameworks:

| Component | Custom LOC needed | Why framework can't help |
|-----------|------------------|------------------------|
| QueryPlannerService (structured SearchPlan with filters) | ~130 | SubQuestionQueryEngine has no concept of date/phrase filters |
| Dynamic tool visibility (phase-based, max 5 visible) | ~100 | No framework supports per-turn tool gating |
| LANCER nugget coverage | ~250 | Research paper from Jan 2026, zero framework support |
| ColBERT MaxSim via Qdrant multivector | ~75 | Built-in ColbertRerank runs in-process, doesn't use Qdrant's native multivector search |
| Navigation/analytics short-circuits | ~60 | No query classification bypass logic |
| Forced search deterministic fallback | ~50 | Agents don't have "force tool if unused" logic |
| Channel dedup (max 2 per source) | ~30 | No per-metadata-field deduplication |
| Round-robin subquery merge | ~25 | Only RRF or concatenation available |

**The total LlamaIndex equivalent is ~885 LOC**, of which only **~85 LOC leverages actual framework features**. The remaining ~800 LOC is custom code. The framework provides roughly **10% of the implementation** — a powerful data point for the benchmark narrative.

### Known LlamaIndex production limitations to acknowledge honestly

Multiple developer reports and analyses document these issues: **15–30% latency overhead** vs direct API calls from abstraction layers, stack traces spanning 50+ frames that make debugging "archaeological," Pydantic validation errors that obscure root causes, no built-in prompt versioning or execution logging, and a **dependency count of 40+** packages vs the custom pipeline's ~12. LlamaIndex's co-founder Jerry Liu has publicly acknowledged that coding agents like Claude Code are disrupting the framework's core value proposition. Critical CVEs have been found in LangChain (9.3/10 severity); similar supply-chain risks apply to any framework with deep dependency trees.

---

## Benchmark methodology requires 100+ questions and three systems

### Sample size: 36 questions is statistically insufficient

For a Wilcoxon signed-rank test with α=0.05 and power=0.80, a medium effect size (d=0.5) requires **27–35 pairs** — but two similar RAG systems will likely show a small effect (d=0.3), requiring **90–100 questions**. Stanford's ARES uses a minimum of 150 annotated datapoints. Microsoft's BenchmarkQED uses 200 queries across 4 classes. **Target 100–150 questions** stratified as follows:

- Factual retrieval (simple lookup): 25–30 questions
- Analytical (reasoning over retrieved info): 20–25 questions
- Multi-hop (info from multiple posts/channels): 20–25 questions
- Temporal ("latest news about X"): 15–20 questions
- Comparison ("compare X and Y approaches"): 15–20 questions
- Unanswerable/out-of-scope (hallucination resistance): 10–15 questions

Run each question through each system **5 times minimum** (Microsoft uses 6 trials) at temperature=0. Use Wilcoxon signed-rank as the primary statistical test, bootstrap confidence intervals (10K resamples) as secondary, and report Cohen's d_z for effect size. Apply Bonferroni correction (α=0.05/3=0.0167) when comparing three systems pairwise.

### The naive baseline is essential for triangulation

Include a third system — **naive RAG** (simple vector search + LLM, no agent, no reranking, ~50–100 LOC) — to create a three-point hierarchy. This triangulation dramatically strengthens conclusions: if Custom > Framework >> Naive, the sophisticated retrieval pipeline is validated. If Custom ≈ Framework >> Naive, both advanced systems work similarly and the custom pipeline's value is in other dimensions (latency, debuggability). If Custom >> Framework ≈ Naive, the framework isn't adding value over basic RAG for this use case — a much more powerful finding than a two-system comparison.

### Controlled variables checklist

Beyond same LLM/embeddings/Qdrant/data/questions (already planned), also control: **temperature=0.0** for all eval runs and all judges, **pinned LLM API version** (e.g., `gpt-4o-2024-08-06`), **same max tokens**, identical system prompts for generation, **same context window budget** (Microsoft standardizes 8K tokens), same hardware/load conditions for latency tests, and **same judge LLM** at temperature=0 for all evaluation.

### Metrics architecture

**Retrieval metrics** (computed against pooled ground-truth): Recall@5, Precision@5, MRR, nDCG@5. Create ground truth via pooling top-10 results from all three systems, then grade each document on a 0–3 relevance scale using GPT-4o as judge, with **20–30% human validation** by a Russian-speaking domain expert.

**End-to-end metrics**: Faithfulness/groundedness (LLM-as-judge), answer relevance (LLM-as-judge), factual correctness (GEval custom metric), **BERTScore using `xlm-roberta-large`** for Russian (or `ai-forever/ruBERT-large` for Russian-only evaluation), and Key Term Accuracy.

**System metrics**: TTFT and total latency at p50/p95/p99, logical SLOC via `cloc` (excluding tests/configs), stack trace depth from user query to Qdrant call, dependency count, and **"modification task" benchmark** (define 3 feature changes, measure LOC touched in each system).

### Evaluation framework: DeepEval as primary

Among RAGAS, ARES, DeepEval, TruLens, and Arize Phoenix, **DeepEval** offers the most complete ecosystem with pytest integration, custom LLM judges (critical for Russian), and the best developer experience. ARES provides unique statistical rigor via Prediction-Powered Inference confidence intervals — use it for final comparative claims but note it's explicitly English-only and requires adaptation. RAGAS now includes agentic metrics (AgentGoalAccuracy, ToolCallF1, TopicAdherence) which are directly applicable to the ReAct agent comparison.

**None of these frameworks natively "support Russian"** — they all delegate to an LLM judge. Russian support depends entirely on the judge LLM (GPT-4o handles Russian well). Provide bilingual rubrics and ask the judge to reason in Russian for consistency.

```python
from deepeval.metrics import GEval
factual_correctness = GEval(
    name="Factual Correctness (Russian)",
    criteria="""Evaluate factual correctness of the actual output vs expected output.
    Both are in Russian. Focus on: key facts, entity relationships, AI/ML terminology.""",
    evaluation_params=["actual_output", "expected_output"],
    model="gpt-4o", threshold=0.7
)
```

---

## Existing research validates the custom-over-framework approach

### The AIMultiple benchmark is the gold standard to replicate

The most rigorous public framework comparison (Jan 2026) tested 5 frameworks with identical agentic RAG workflows, same models, 100 queries × 100 runs. Key finding: **framework overhead is 3–14ms** (DSPy ~3.5ms, LlamaIndex ~6ms, LangChain ~10ms). All achieved 100% accuracy. The real differentiator is **token efficiency**: Haystack/LlamaIndex used ~1.6K tokens per query vs LangChain's ~2.4K. This means the framework choice affects cost more than quality or latency — a nuanced finding that the benchmark should explore.

### Russian-language evaluation infrastructure exists

**RusBEIR** (April 2025) provides 17 Russian IR datasets including Ria-News (news domain), directly applicable for evaluating retrieval quality. **ruMTEB** (NAACL 2025) offers 23 Russian datasets across 7 task categories with a public HuggingFace leaderboard. **MIRACL** covers Russian among 18 languages. **MIRAGE-Bench** evaluates RAG answer generation across 18 languages including Russian. These benchmarks can validate that the embedding models and retrieval strategies work well for Russian, complementing the domain-specific benchmark.

### Industry consensus: custom wins for production, especially non-English

Developer sentiment is overwhelmingly skeptical of frameworks for production. Max Woolf (BuzzFeed) wasted a month on LangChain before a custom ReAct flow "immediately outperformed" it. Multiple Hacker News threads with hundreds of upvotes criticize framework over-abstraction ("5 layers of abstraction just to change a minute detail"). A retail company rewrote from LangChain to custom after 4 months due to scaling issues. The balanced view: frameworks excel for prototyping and team onboarding, custom excels for production-critical systems with non-standard domains. **Hybrid approaches are most common** — using frameworks for prototyping, then selectively replacing with custom components.

### LANCER and nugget-based evaluation are gaining traction

The LANCER paper (Jan 2026) implements nugget coverage for retrieval sufficiency and was adopted by the TREC 2024 RAG Track. The **AutoNuggetizer** framework automates fact extraction for evaluation and shows "strong agreement at the run level between fully automatic nugget evaluation and human-based variants." TREC 2025's RAGTIME Track specifically hosts cross-language report generation including Russian. The custom pipeline's LANCER implementation is cutting-edge — no framework supports it, and it represents genuine research-informed engineering.

---

## Implementation plan: MVP in 4–7 days

### Phase 0: Skeleton and ADR (0.5 days)

Write the Architecture Decision Record and README skeleton first. The ADR format (recommended by ThoughtWorks, used by AWS/Azure) structures the comparison as: Context → Considered Options → Decision Drivers → Benchmark Results → Decision Outcome → Consequences. This framing immediately signals engineering maturity.

### Phase 1: Retrieval-only benchmark (3–5 days)

This is the highest-value MVP. Connect LlamaIndex to the existing Qdrant collection (no re-indexing needed), implement the `WeightedRRFHybridRetriever`, add cross-encoder reranking, create 50–100 golden queries with pooled ground truth, and run the shared evaluation harness.

**Critical: reuse `evaluate_agent.py`** by abstracting behind a `RetrieverInterface` protocol:

```python
class RetrieverInterface(Protocol):
    def retrieve(self, query: str, top_k: int) -> List[Document]: ...

# Both pipelines implement this interface
# Shared evaluation code runs against the interface
```

### Phase 2: Blog post with Phase 1 results (1–2 days)

Write the narrative: Problem → Hypothesis → Methodology → Results → Analysis → Decision → Lessons. Include grouped bar charts for latency (p50/p95/p99), radar charts for retrieval quality metrics, and a feature parity table. **Dedicate a full section to "Where LlamaIndex Wins"** (development velocity, ecosystem, onboarding).

### Phase 3: Agent comparison (3–4 days, optional polish)

Implement LlamaIndex agent with equivalent tools. Compare routing accuracy, tool selection, response quality. This phase surfaces the dynamic tool visibility, forced search fallback, and LANCER coverage differences — the custom pipeline's strongest differentiators.

### Code organization: separate directory in same repo

```
rag-benchmark/
├── shared/           # Interfaces, metrics, evaluation harness, golden queries
├── custom_pipeline/  # Adapter wrapping existing 14.5K LOC pipeline
├── llamaindex_pipeline/  # LlamaIndex implementation
├── naive_baseline/   # 50-100 LOC simple vector search + LLM
├── configs/          # YAML configs for reproducible experiments
├── results/          # JSON results + generated charts
├── scripts/          # run_benchmark.py, generate_charts.py
├── docker/           # docker-compose.yml for reproducibility
└── docs/             # ADR, methodology, blog post draft
```

Use YAML configs for each experiment (retrieval-only, agent, full pipeline) so results are reproducible. A Makefile with `make benchmark-retrieval`, `make charts`, `make report` streamlines the workflow.

---

## Presenting results for maximum hiring impact

The format hierarchy is: **GitHub repo with exceptional README** (primary — recruiters scan in <90 seconds) → **technical blog post** on Medium or personal site (narrative companion demonstrating communication skills) → **interactive demo** (optional but extremely high-signal, even a simple Streamlit side-by-side comparison).

A Jupyter notebook alone signals "student work." The blog + repo combination signals "production engineer who communicates." Three well-documented projects with real metrics outperform fifteen notebooks.

The benchmark's key message must balance showing engineering depth with intellectual honesty. Present a dedicated section titled something like "Where the framework shines" — acknowledging development velocity (prototype in hours vs weeks), the 160+ data loader ecosystem, easier onboarding for new team members, and community-driven improvements. Then show where custom engineering delivers measurable value: the weighted RRF fusion, LANCER coverage, dynamic tool visibility, and production-specific features that frameworks don't address.

The most powerful conclusion is nuanced: **"Framework overhead is negligible (3–14ms), but the framework covers only 10% of the engineering surface area of a production RAG system. The decision isn't framework vs custom — it's about which 90% you're willing to build yourself and whether the 10% scaffolding justifies the dependency cost."** This positions the author as someone who makes data-driven architectural decisions, which is exactly what senior engineering hiring managers value most.

### Estimated total effort

| Phase | Scope | Days | Portfolio signal |
|-------|-------|------|-----------------|
| 0 | ADR + README skeleton | 0.5 | "Thinks before coding" |
| 1 | Retrieval-only benchmark | 3–5 | "Measures and compares rigorously" |
| 2 | Blog post with results | 1–2 | "Communicates complex ideas clearly" |
| 3 | Agent comparison | 3–4 | "Understands system-level tradeoffs" |
| 4 | Full pipeline + Docker + CI | 2–3 | "Builds production infrastructure" |

**Phases 0–2 alone (4–7 days) deliver a compelling portfolio piece.** Phases 3–4 are polish that strengthens the story but isn't required for the core message to land.