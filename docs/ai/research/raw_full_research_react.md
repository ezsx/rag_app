Executive Summary

We propose integrating an Agentic ReAct-RAG pipeline into the existing rag_app system to enhance answer accuracy, especially for complex queries over Telegram channel data. The design introduces a step-by-step LLM agent that can plan queries, invoke search and reranking tools, and verify facts before finalizing answers. This agent loop builds on rag_app’s current retrieval-augmented generation: we retain the Query Planner for multi-query decomposition, the hybrid BM25+dense search with rank fusion, and the CPU-friendly reranker. By adding ReAct-style reasoning and tool use, the system can dynamically reformulate queries, pull in diverse evidence, and double-check claims. We expect improved answer quality and trustworthiness – fewer missed relevant messages and reduced hallucinations – at a modest latency cost (bounded by a few tool calls and caching). The ReAct agent will leverage Telegram-specific metadata (channels, dates, authors) to focus searches, and use verification steps to ensure answers are well-supported by cited messages. The design is mindful of performance: it maintains a CPU-only baseline but allows optional GPU acceleration (≈16 GB VRAM) for faster LLM inference. Crucially, it preserves existing API behavior (via feature flags and separate endpoints)
GitHub
GitHub
, enabling a safe rollout of the new capabilities without disrupting the current production service. Overall, this upgrade is expected to boost factual precision and user confidence in answers by orchestrating the retrieval and reasoning process in a more intelligent, multi-step manner, while keeping throughput and cost within practical limits.

Current State of rag_app

Architecture: The rag_app system currently implements a one-step RAG pipeline with a few advanced features. When a user asks a question, a Query Planner LLM (Qwen 2.5B on CPU by default) generates a structured search plan in JSON – typically 3–6 refined sub-queries plus filters like date ranges
GitHub
. This plan is produced with grammar or schema constraints for reliability
GitHub
. Next, a hybrid search executes: it runs lexical BM25 against an offline index and dense vector search via ChromaDB, then merges results using Reciprocal Rank Fusion (RRF)
GitHub
. The merged hits can be further diversified by Maximal Marginal Relevance (MMR) to reduce duplicates
GitHub
. An optional reranking stage re-scores top candidates using a cross-encoder model (BAAI/bge-reranker-v2-m3 on CPU)
GitHub
, to improve the relevance ordering. Finally, the top retrieved Telegram messages (with their metadata) are assembled into a context and fed to the main LLM (e.g. a 20B GPT-J/T model) which generates the answer. This answer can be streamed to the client in real-time via SSE
GitHub
, allowing token-by-token display. The system also supports hot-swapping models at runtime (for the LLM, embeddings, etc.) through its API
GitHub
.

Pipeline Details: In the current implementation, most of these steps happen in a straightforward sequence within the /v1/qa endpoint. The Query Planner returns a JSON plan (with normalized queries, k_per_query, and filters) which guides the retrieval stage. The Hybrid Retriever component executes BM25 and dense queries (each limited to top-100 by default
GitHub
), then applies RRF to combine scores, yielding e.g. ~60 candidates (tunable via K_FUSION)
GitHub
GitHub
. If enabled, MMR diversification (λ≈0.7) is applied to re-rank and cut down to ~60 results
GitHub
. If the reranker is on, it then processes the top 80 or so results (in batches of 16) and selects the best 5–10 passages for final answer context
GitHub
GitHub
. The chosen messages are fetched (by ID) from the Telegram collection and concatenated into a context prompt with source markers (footnote-style citations like “[1]”)
GitHub
GitHub
. The LLM’s answer is returned along with optional source references. This flow already addresses many RAG fundamentals: it uses multi-query expansion to increase recall, hybrids lexical and semantic search for robustness, and does lightweight reranking on CPU for precision. Caching is in place for plans and fusion results (via an in-memory or Redis cache) to save on repeated work
GitHub
. The Telegram-specific ingestion populates the Chroma and BM25 indexes with each message as a document, storing metadata like channel name/ID, message ID, timestamp, author, and any attachments or links (as text or placeholders). This metadata is leveraged by the planner (e.g. to filter by date or channel) and can be used in search queries or post-filtering. For example, the plan for “новости РБК за январь” (RBC news for January) includes a date filter bounding the search to Jan 2024
GitHub
.

ReAct Infrastructure (in progress): While the current QA flow is essentially single-turn, the project has laid groundwork for agent-like tool usage. There is a ToolRunner system with defined tools such as router_select (to pick BM25 vs dense vs hybrid based on query heuristics), fetch_docs (to retrieve full text by IDs), compose_context (to build the final prompt with citations), dedup_diversify (MMR-based result filtering), verify (to fact-check claims against the knowledge base), and utility tools like math_eval and time_now
GitHub
GitHub
. These tools have Pydantic schemas for input/output and run with timeouts and JSON trace logging
GitHub
GitHub
. An initial AgentService and /v1/agent API exist (behind feature flags) which implement a ReAct loop: the agent LLM receives a system prompt describing how to use tools (with the format “Thought -> Action -> Observation” and finally “FinalAnswer”)
GitHub
GitHub
. In this mode, the LLM can iteratively call the above tools – for example, first call router_select, then multi_query_rewrite or hybrid_search, etc., examine the output, and continue reasoning. The tool calls and observations are streamed as SSE events for transparency
GitHub
GitHub
. However, the current agent implementation relies on prompt-format and few-shot hints (textual ReAct format) rather than strict function calling or grammar constraints, which can be brittle with certain local models. As of now, the agent mode is experimental – the default QA uses the static pipeline described earlier. This provides clear “insertion points” for a more robust ReAct integration: we can attach the agent loop at the QA service level (or via a dedicated endpoint) so that complex queries trigger multi-step reasoning and tool use, while simpler queries could still use the direct pipeline. The aim is to enhance capabilities (multi-hop reasoning, fact-checking, etc.) without regressing baseline behavior or breaking the existing API (the new agent lives under /v1/agent, leaving /v1/qa unchanged
GitHub
).

Recommended ReAct-RAG Design

We recommend a unified ReAct-RAG architecture that orchestrates query planning, retrieval, and answer generation through an LLM-driven agent loop. The design follows a router → planner → tools → answer sequence, with well-defined tool interfaces and fallback paths for robustness. Below is the proposed flow:

flowchart TD
    A[User Query] --> B{Router Select?\n(Heuristic)}
    subgraph Plan & Retrieve
      C[Query Planner LLM] --> D[Search Tool<br/>(Hybrid/BM25/Dense)]
      D --> E[Candidate Docs<br/>(RRF + MMR)]
      E --> F{Need Rerank?}
      F -- yes --> G[Rerank Tool (CrossEncoder)]
      F -- no  --> G
      G --> H[Top-N Results]
    end
    B -- "bm25/dense/hybrid" --> C
    H --> I[Compose Context]
    I --> J{Coverage ≥ 80%?}
    J -- yes --> K[Answer LLM<br/>(Generate Final Answer)]
    J -- no --> L[Refine Search]
    L --> D  %% refine could adjust queries or k
    K --> M{Verify Answer?}
    M -- verify on --> N[Verify Tool (Fact-Check)]
    M -- verify off --> O[Return Answer]
    N --> |if low confidence| L
    N --> |if verified| O[Return Answer]


ReAct Loop Description: The process begins with a Router step (0) that quickly inspects the user query and decides which retrieval mode is best: BM25, dense, or hybrid
GitHub
GitHub
. This is a lightweight heuristic function (router_select tool) that checks the query’s characteristics (e.g. presence of keywords/operators, numeric or date terms favor BM25; very short queries favor BM25; long or abstract questions favor dense; mixed signals default to hybrid)
GitHub
. Next, the Query Planner (step 1) analyzes the question and produces a Search Plan JSON. We will maintain the current approach of using a constrained LLM prompt to output an object with fields like subqueries (3–6 search queries covering different aspects or phrasings), metadata_filters (e.g. date range or channel), and k_per_query. This can be enforced via a GBNF grammar for reliability on local models
GitHub
 or via function-call/JSON parsing on GPT-4. The planner should also handle query decomposition: for instance, if asked “What did channel X report about Y in 2022?”, it might produce subqueries for Y in channel X and apply a date filter for 2022. We’ll use the existing make_plan function (which already uses LLM + grammar)
GitHub
 and ensure it covers new metadata (like channel filters) as needed. The output plan JSON schema could be as follows:

{
  "subqueries": ["string", "..."],
  "must_phrases": ["..."],
  "should_phrases": ["..."],
  "metadata_filters": { "date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD", "channel": "..." },
  "k_per_query": 10,
  "fusion": "rrf" 
}


(This schema will be enforced via GBNF; e.g., max_subqueries=6, if the model produces fewer than 3 we can auto-complete via a small paraphraser.)
GitHub
GitHub
 After planning, the agent enters the tool execution phase (steps 2–4): it uses the plan to perform searches, fusion, and reranking. These can be done as separate tool calls or combined: for clarity and modularity, we propose a primary search tool that encapsulates multi-query retrieval and fusion, and a separate rerank tool for re-scoring. In practice, the agent could either call a single hybrid_search tool that returns final top results, or call bm25_search and dense_search individually followed by a fusion_rank tool
GitHub
. Our recommendation is to expose a search tool which takes parameters {queries: [...], filters: {...}} and internally does the BM25/dense retrieval and RRF merge (respecting the chosen route). Alternatively, the agent can sequence it: e.g. router_select returns “hybrid”, then agent calls bm25_search and dense_search (tools), then calls fusion_rank tool to combine. The end result is a list of candidates with scores and doc IDs. At that point, the agent may decide to invoke dedup_diversify (MMR) as a tool to remove near-duplicates and ensure diverse info – however, if MMR is already integrated in fusion_rank (as we plan: RRF then MMR)
GitHub
, a separate call may not be needed.

After retrieval, if a large number of candidates are present, the agent should consider reranking (step 4). This can be an automatic step: e.g., if >20 results or if the top scores are close, call the rerank tool on the top 40–60 passages. The rerank tool will use a cross-encoder (MiniLM or BGE) to score each passage against the query and return a re-ordered top-N list
GitHub
. The agent receives this and picks the top few (e.g. top 5 for context). If the retrieval yield was very small (say only 3 hits), the agent might skip reranking to save time
GitHub
. Each tool call returns a JSON result (with ok status and data); the agent appends an Observation in the prompt containing a summary or snippet of the result. (Our implementation will likely truncate the observation to avoid overloading the LLM – e.g. showing only the document IDs or brief snippets rather than full text in the agent’s working memory.)

Next is context assembly (step 5). The agent calls fetch_docs with the selected top IDs, possibly with a window parameter if we want to fetch some context around each hit (for Telegram, each “document” is typically a message; if it’s long or if the answer might require adjacent messages for context, a window of ±1 message could be fetched). The raw texts are then passed to compose_context tool, which formats them into a final prompt segment with citations
GitHub
. We will continue to use a footnote numbering scheme: e.g. [1] ...snippet... [2] ...snippet... and prepare a mapping of [1]→source metadata (channel name or ID, message ID, date). The compose step also ensures we don’t exceed the LLM context length (e.g. targeting ~1800 tokens for retrieved text out of a 4096-token context)
GitHub
GitHub
. Importantly, this step can apply the “Lost-in-the-Middle” mitigation: identify the most relevant passages and place them at the beginning (and possibly end) of the context to avoid the middle-of-context fading effect
GitHub
. It can also chunk long documents: for instance, if a Telegram post is very long, we’d break it into ~300-token chunks with overlaps and include the most relevant chunk(s) first
GitHub
. The agent does not actually see the full compose output until generation; rather, compose_context returns a JSON with prompt (the assembled context) and metadata like a citations list
GitHub
.

At this point, the agent determines if it has enough information to answer. We introduce a coverage check: the compose_context tool or the agent itself can compute a “citation coverage” metric – e.g. fraction of the question or predicted answer content that is supported by the retrieved text
GitHub
. In practice, we can approximate this by checking if all key entities or terms from the query have appeared in the context, or by a rough LLM evaluation of context sufficiency. If coverage is high (e.g. ≥ 0.8)
GitHub
GitHub
, the agent proceeds to give the final answer. If coverage is low, the agent can attempt a refinement loop: go back to step 2 (retrieval) for one more round. This could involve using a broader search: e.g. increasing k_per_query, generating a couple of new subqueries (perhaps using a multi_query_rewrite tool to paraphrase the query differently
GitHub
), or switching strategy (if route was dense, try hybrid or vice versa). For example, the agent might think: “Not enough info found; let’s search the web or extend date range.” In our design, we limit to 1 extra round to avoid long cycles
GitHub
. The second round might use a fallback strategy like dense-only with a higher k, since it’s likely something was missed in first pass
GitHub
. After that, whatever is found will be used. (If still insufficient, the agent will conclude it cannot answer confidently.)

Finally, the Answer Generation occurs (step 6): the agent, having gathered context, now either directly produces the FinalAnswer in the ReAct loop (if we use the prompting style with Thought/Action/Observation, the final step is the model writing the answer). In the function-call variant, we might have the LLM return a JSON like {"answer": "...", "sources": [1,2]} as per a schema
GitHub
GitHub
, but with our footnote approach, it’s easier to have the model just write a textual answer referencing [1], [2], etc. The answer should include citations for each factual claim, ideally. Our agent system prompt explicitly tells it to end with FinalAnswer: and the answer text
GitHub
. We will enhance this prompt to remind the model to use citations and to only answer if supported by the retrieved info. The answer is then streamed to the user.

Verification (step 7): As an optional post-processing (or as part of the agent loop), we include a verify tool that performs fact-checking. The verify tool takes a claim (text) and runs an internal search to see if it’s supported, returning a boolean or confidence score with some evidence
GitHub
. In practice, this uses a simplified retrieval of top-3 passages for the claim and checks if any contain a match. We envision two uses of this: (a) Agent self-check – after generating an answer (before finalizing), the agent could call verify on one or more key statements or on the answer as a whole. For example, if the answer says “Alice won the competition in 2021”, the agent might call verify("Alice won the competition in 2021"). If verify returns low confidence (e.g. verified: false or below a threshold), the agent knows that claim wasn’t found in the docs and can adjust or refuse. (b) Final answer verification – the system can, outside the agent, run a verification pass on the final answer. Since we have the full context used, a simpler method is to ensure faithfulness: check that each sentence of the answer has an overlapping n-gram with some source text. We will implement a check such as: for each sentence or fact in the answer, confirm it can be found (or inferred) from the retrieved documents; if not, we either remove that part or add a disclaimer. This verification step helps prevent hallucinations by effectively doing a “cited passage coverage” test
docs.aws.amazon.com
docs.aws.amazon.com
. If the answer cannot be well supported, the agent will respond with an uncertainty (e.g. “I couldn’t find information on that”) or a refusal, rather than risk a confident fabrication. This addresses one key limitation of naive RAG – lack of validation
weaviate.io
weaviate.io
.

Tool Contracts & Timeouts: Each tool in the ReAct loop has a defined JSON schema for inputs and outputs, ensuring consistency and security. For example:

Tool: search – Input: {"queries": [str], "filters": { "date_from": str, "date_to": str, "channel": str }, "k": int}. Output: {"hits": [ { "id": str, "score": float, "snippet": str, "source": {...} } , ... ]} (sorted by score). This may encapsulate multi-query fusion internally; alternatively, separate tools bm25_search, dense_search, fusion_rank can be called in sequence
GitHub
.

Tool: rerank – Input: {"hits": [ {id:str, text:str, ...} ], "top_n": int}. Output: {"hits": [ {...} ]} (reordered list with updated scores). The reranker will likely ignore the text beyond a certain length (256 tokens per passage) for efficiency
GitHub
.

Tool: verify – Input: {"query": str, "claim": str, "top_k": int} – it can optionally take the original user query for context or just verify the claim standalone. Output: {"verified": bool, "confidence": float, "evidence": [str], "documents_found": int}
GitHub
. In our case, evidence might be short snippets from any supporting docs.

All tools follow the unified format already defined in ToolRequest/ToolResponse (with ok status and error messages if any)
GitHub
GitHub
. They each have a default timeout (we will use ~4–5s per tool call as a safeguard)
GitHub
GitHub
. The AgentService will run each tool via the ToolRunner in a background thread, so if a call exceeds the time limit or fails, the agent gets an error observation (and can decide to either retry or give up)
GitHub
GitHub
. We will keep step limits (max ~3–4 tool uses per query) to control latency
GitHub
. If the agent hits the max steps or a timeout, it will fall back to returning whatever partial answer it has or trigger the old QA flow.

Integration into FastAPI: We will introduce a new asynchronous endpoint (already outlined as /v1/agent/stream) for the ReAct pipeline. The existing qa_service remains for single-turn answers, while the agent_service will encapsulate the above logic. According to our plan (and the ADR decisions), we will not shoehorn this into the old /qa to avoid risk
GitHub
. Instead, clients can opt-in to the agent via the new endpoint. Internally, both share components: the AgentService will use the same dependencies (LLM, retriever, planner) from core.deps
GitHub
. We’ll add any missing pieces: e.g. ensure multi_query_rewrite and search tools are registered in deps.get_agent_service() similar to existing ones
GitHub
. The environment config (.env) will get new flags such as ENABLE_AGENT=True, AGENT_MAX_STEPS=4, etc., and possibly a TOOLS_ALLOWLIST to restrict which tools the agent can use (for safety). The AgentService will also respect the active collection (it can use the collection field from AgentRequest
GitHub
 to filter searches to a specific Telegram collection if provided, defaulting to whatever is selected globally or the “current” collection in use).

Telegram-Specific Adjustments: In designing the agent, we tailor it to Telegram data characteristics. We define document chunking: each Telegram message is typically stored as one “document” (if very long, it might be chunked by the ingestion pipeline). The agent’s retrieval should treat each message as an independent unit (unless threads are explicitly linked; Telegram doesn’t have true threads, but replies could be chained – currently we don’t have linking of messages beyond maybe including the replied-to message text during ingestion). We include temporal filtering: when the query or plan specifies a date range, the search tool will apply that on both BM25 and Chroma queries (Chroma supports metadata filters, and BM25 can filter by a date field post-retrieval or via an index that stores date). If a user question implies a time window (“in January 2022” or “last week”), the planner or a time_normalize tool should convert that to exact dates
GitHub
. Our tools list already includes temporal_normalize (dateparser-based) to handle phrases like “last month”
GitHub
. The agent can call this tool early on to get date_from/date_to values, which then feed into the search. We also ensure channel filtering: if the query says “in @channelName”, the planner or agent could add a filter for that specific collection or channel metadata. The underlying store has collections (each representing a channel or group of channels). We’ll support that by either switching the active collection or filtering on a channel_id field.

When composing context from Telegram messages, we will include each message’s metadata in the citation (e.g. “[1] {ChannelName}, {Date}” in the sources list) to give user proper source info. We also address forwards/duplicates: Telegram often has forwarded messages or different channels posting the same content. Our deduplication (via MMR or explicit duplicate removal by content hash) will ensure we don’t feed the LLM repeated text
GitHub
GitHub
. If two hits are essentially the same text, we keep only one, or if one is a forward of the other, we prefer the original source channel perhaps. The verify step also helps catch if the answer would double-count a fact from two identical sources.

Retrieval & Rerank Improvements

To maximize the retrieval quality in this new setup, we will tweak and extend the current hybrid search configuration. First, multi-query generation: the Query Planner already provides multiple subqueries, which is a strength. We will ensure it produces at least 3 distinct phrasings (in Russian or English as needed) and at most 6
GitHub
. If the planner yields too few or very similar ones, we can enable the multi_query_rewrite tool to generate additional paraphrases
GitHub
. For example, if the queries are not diverse (e.g. all contain the same rare term), the agent might call multi_query_rewrite with the original question to get 1–2 alternative formulations (using a tiny grammar that outputs an array of strings)
GitHub
GitHub
. These can help capture different aspects. We will use this sparingly (trigger only if needed, to save time).

Hybrid retrieval parameters: We will keep RRF (Reciprocal Rank Fusion) as the primary fusion method since it robustly combines BM25 and dense scores
GitHub
. The RRF k value (the ranking constant in formula) can be set around 60 (as currently) – this essentially means we consider up to ~60 results from each source for fusion
GitHub
. We will increase the initial pool size a bit: for instance, retrieve top-50 from BM25 and top-50 from Chroma (so HYBRID_TOP_BM25=50, HYBRID_TOP_DENSE=50 for round 1). After RRF merging, we expect ~60–80 combined results. We’ll then apply MMR diversification to select the top 40 or so for consideration
GitHub
. MMR uses cosine similarity on embeddings; we set λ = 0.5 for a balanced relevance/diversity trade-off
GitHub
GitHub
 (this is slightly lower than the current 0.7, to encourage more diversity). The parameter MMR_TOP_N (the number of final results after MMR) will be around 40. This ensures the reranker (if used) gets a manageable number. These values can be fine-tuned during testing – we’ll prepare a small table of recommended settings:

Parameter	Proposed Value	Notes
FUSION_STRATEGY	rrf	Use RRF by default (robust merge)
GitHub
.
K_FUSION (per src)	50 (BM25), 50 (dense)	Increase initial candidates per source.
RRF k constant	60	RRF scoring constant (already ~60)
GitHub
.
ENABLE_MMR	true	Diversify results post-fusion.
MMR_LAMBDA	0.5	Balance novelty vs relevance
GitHub
.
MMR_OUTPUT_K	40	Pass top-40 to next stage (or answer if no rerank).
ENABLE_RERANKER	conditional (auto)	Use if >20 results or as needed.
RERANKER_MODEL	MiniLM-L6 or BGE v2	Use a fast cross-encoder model on CPU
GitHub
.
RERANKER_TOP_N	40	Number of candidates to rerank (batchable).
RERANKER_BATCH	8–16	Batch size for reranker model for speed.

(These values ensure a good recall while keeping processing feasible. RRF and MMR values are informed by recent RAG literature: 4–6 queries with RRF + MMR (λ ~0.5) is a proven approach to cover diverse facets
GitHub
.)

We’ll implement query routing heuristics as outlined (maybe already implemented in router_select): e.g. if the planner outputs any must_phrases (exact keywords that must appear), we might force a BM25 subquery for each of those to ensure they appear in results. If the query has advanced syntax or exact quotes, use BM25 only. Otherwise hybrid is default
GitHub
. These rules will be encoded in the router tool and can be adjusted based on query logs.

Query expansion techniques: In cases where the initial retrieval fails (low coverage), we want fallback strategies. One promising method is HyDE (Hypothetical Document Embeddings)
docs.haystack.deepset.ai
 – here the LLM would generate a hypothetical answer or document given the query, and then we embed that and search. This can help when queries are abstract or when the vector model doesn’t capture the query well. We could implement HyDE as follows: if after one round, coverage < 50%, call the LLM with a prompt like “Draft an answer to the question as if you knew the info.” (one short paragraph). Use that text as a new query (for dense retrieval primarily)
docs.haystack.deepset.ai
. This might surface documents that directly match an expected answer phrasing. We will include this as an automatic fallback in the second round of retrieval if needed. Another expansion is to generate synonyms or related terms for key entities (especially for Russian, to handle morphological variants). The planner might already output some in should_phrases, but we can enhance the BM25 query by OR-ing those terms. We will also consider SPLADE (a sparse model that expands the query into weighted terms) – however, implementing SPLADE is non-trivial and may require a heavy model. As a simpler approach, we might integrate key term weighting: e.g., if the planner identifies a rare term or name, ensure the BM25 query boosts that term strongly (maybe using BM25’s query-time boosts or by duplicating the term in the query string).

Reranker enhancements: Currently BGE v2-m3 is used, but we have evidence that a well-tuned MiniLM cross-encoder can perform better and faster for reranking
GitHub
. We will likely switch to the MS MARCO MiniLM-L6-v2 model
GitHub
, which is 6-layer and can run very quickly on CPU (with possible quantization or ONNX acceleration)
GitHub
. This model is known to improve ranking quality significantly over raw vector similarity. If multi-lingual support is needed (since Telegram data can be Russian, English, etc.), we might use the multilingual MiniLM variant or mpnet. Cohere Rerank is another alternative – it’s powerful (Cohere’s reranker 3.0 can boost search accuracy by reordering based on deep semantic alignment
aws.amazon.com
) and is multilingual
aws.amazon.com
, but using it would incur external API calls and cost. As a compromise, we stick to open models for now. For GPU deployments, we could consider larger cross-encoders or even ColBERT-based reranking: e.g. a ColBERT model could tokenize documents and allow fine-grained scoring. However, given a single 16GB GPU, a simpler approach is to maybe use OpenAI’s function calling to have GPT-4 judge the relevance of top passages (but that is expensive). So, our primary plan: use MiniLM for rerank on up to 40 passages, which should add only ~100-200ms on CPU. The impact of reranking is substantial: studies show adding a reranker can improve retrieval hit-rate by 8–11% for models like BGE
blog.lancedb.com
, moving truly relevant results to the very top. We will incorporate a check such that if the top-1 retrieved passage is already very high similarity and clearly answers the question, we might skip rerank to save time (i.e., if the highest BM25 score is above some threshold or if dense cosine > 0.9, often it’s a direct hit). Otherwise, rerank ensures the final context is as on-point as possible. The target is to feed the LLM 5 highly relevant passages rather than 5 somewhat relevant ones, which should increase answer precision.

To further optimize latency, we will use asynchronous execution for retrieval steps: BM25 and vector search can run in parallel, as can fetch_docs for multiple IDs
GitHub
. The agent loop will be orchestrated to overlap where possible (though the LLM itself will run sequentially due to thread safety). By carefully tuning these retrieval parameters and adding fallback expansions, we expect to maximize recall and diversity of information gathered, which directly contributes to answer completeness.

Verify Stage – Fact-Checking & Self-Consistency

In the final stage of the pipeline, we aim to validate the answer’s faithfulness to avoid hallucinations and unsupported claims. We propose a two-pronged verification approach integrated into the ReAct loop:

Self-Verification Tool Calls: The agent can proactively verify intermediate facts during its reasoning. For example, suppose the question is: “Did person X announce project Y on their Telegram channel?” The agent might find a message and form an answer “Yes, on Jan 5, X said they are working on Y.” Before finalizing, the agent can use the verify tool on that statement. Under the hood, verify will perform a mini-search (possibly restricted to the Telegram data) for evidence of the claim
GitHub
. It returns a confidence score (say based on BM25 score of the best match) and some snippets. If verified=true or confidence > threshold (e.g. >0.6), the agent proceeds. If not, the agent realizes the info isn’t directly supported – it might then decide to either: (a) search again for that specific claim (maybe it missed it, so do a new targeted query), or (b) reformulate the answer to be more cautious (“There’s no record of X explicitly announcing Y.”). This aligns with a ReAct “ask for verification” step: the agent essentially queries its own knowledge base for each important assertion. This behavior can be encouraged in the prompt instructions (like “Before giving final answer, verify each factual claim by search”). Using the verify tool in-loop thus adds a layer of confirmation.

Post-answer Consistency Check: After the answer is produced (especially if not using the agent loop for final output, e.g. in a synchronous /qa mode), we implement a final consistency scan. One tactic is self-consistency: if time permits, have the LLM generate the answer multiple times with slight variations or different chain-of-thought seeds, to see if it converges on the same answer
weaviate.io
. Divergent answers indicate uncertainty. However, this is costly and not feasible for real-time. Instead, we focus on citation consistency: ensure every part of the answer can be traced to the retrieved sources. We will programmatically ensure that for each citation [n] used, the referenced document indeed contains the text or fact that it’s next to in the answer. And vice versa: each key fact without a citation triggers a warning. Our compose_context already tries to enforce at least one citation per substantive sentence
GitHub
GitHub
. We can also compute citation precision and coverage metrics on the answer, similar to Amazon Bedrock’s evaluation: citation precision measures how many cited passages were actually relevant, and citation coverage measures how much of the answer is supported by citations
docs.aws.amazon.com
docs.aws.amazon.com
. In a live system, we can’t compute these fully without ground truth, but we approximate by checking overlap between answer and sources. If coverage is low (e.g. answer contains names or numbers not present in any source text), that’s a red flag. The system can react by appending an “according to available data” qualifier or by refusing to answer fully.

In summary, the verify stage will be tightly integrated such that the agent returns not just an answer but also an indicator of answer confidence. If the answer fails verification, the agent can output an alternative response: for instance, an apology and “I could not find information to answer that” (with no misleading content). The ReAct framework allows for this gracefully: the agent’s final step could simply be a refusal if tools didn’t yield sufficient info. We will encode triggers for refusal (e.g. if after second retrieval round coverage <0.5, or if verify finds 0 evidence for the main question topic, then final answer = “I’m sorry, I cannot find that information.”). This ensures we prefer no answer over a wrong answer – a critical aspect for user trust.

Additionally, verification helps with duplicate/conflict handling: if two sources say different things (e.g. two channels give different dates for an event), the agent should notice and either reconcile if possible or at least present one with caveat. This can be done by the agent analyzing the retrieved texts (in Observation steps) – essentially a mini “cross-check.” While full automated contradiction detection is complex, we can prompt the LLM to be vigilant: e.g. “If sources conflict, note this in the answer.” The verify tool might also be extended (as a future idea) to cross-verify multiple references (like ensure at least 2 independent sources agree on a fact).

Finally, we consider self-consistency in terms of reasoning: the agent’s chain-of-thought is not exposed to the user (we suppress printing Thought: in final answer)
GitHub
, but internally we log it. We can analyze these logs offline to see if the agent changed its mind mid-way often – if so, it might indicate uncertainty. Tuning the prompt and tools can reduce dithering. We set a step limit (max_steps=4), so the agent won’t loop endlessly
GitHub
. In case it doesn’t reach confidence by then, it will fallback. All these measures together aim to ensure that the answer delivered has high faithfulness (no unsupported info) and the system can gracefully handle unanswerable queries. This approach follows industry best practices where the LLM+RAG output is vetted by either another LLM or by deterministic checks for factual alignment
docs.aws.amazon.com
docs.aws.amazon.com
.

Integration Plan (Code & Deployment)

To implement this design, we propose a series of pull requests, each focusing on a component, to incrementally merge into the rag_app codebase. Below is the plan outlining changes to specific files/modules:

PR 1: Agent Service & API Integration
Files: src/services/agent_service.py (new or expanded), src/api/v1/endpoints/agent.py (new), src/core/deps.py (modify), src/core/settings.py (modify).
Description: Introduce the ReAct AgentService class orchestrating the loop (as per the pseudocode logic in this design). Implement streaming via SSE of AgentStepEvents
GitHub
GitHub
. Define config flags in settings (e.g. ENABLE_AGENT, AGENT_MAX_STEPS, tool timeouts, etc.). In deps.py, instantiate AgentService with the LLM factory, ToolRunner, QAService, etc.
GitHub
. Add new FastAPI endpoint /v1/agent/stream that accepts AgentRequest and uses AgentService to stream results
GitHub
. Ensure CORS and auth are consistent with other endpoints. This PR establishes the separate agent pipeline without altering existing QA endpoints (per non-invasive principle)
GitHub
. Fallback: AgentService should catch exceptions or timeouts and call the QAService as backup to return a normal answer
GitHub
.

PR 2: Tool Implementation and Enhancement
Files: src/services/tools/*.py (new: search.py, possibly bm25_search.py, dense_search.py; updated: router_select.py, dedup_diversify.py, verify.py), src/services/query_planner_service.py (minor), src/core/deps.py (register tools).
Description: Implement the search tool as described – likely as a wrapper that calls the HybridRetriever. It should accept multiple queries and filters, run the retrieval (parallel BM25 & dense), fuse results (RRF), then (optionally) call dedup_diversify (MMR) on them
GitHub
. This can reuse code from the /v1/search endpoint or HybridRetriever class. Implement bm25_search and dense_search tools if we decide to expose them separately (they would call the underlying adapters). Ensure each tool returns JSON data conforming to schemas. Update router_select.py if needed to refine heuristics
GitHub
. Extend verify.py: possibly incorporate the simplified logic of using the main retriever to fetch top-3 docs for the claim and set a confidence (e.g., based on BM25 score of top result)
GitHub
. Minor update to QueryPlanner if we want it to handle channel filters – e.g. if collection specified in AgentRequest, pass that context to planner to maybe include a channel in the plan. In deps.py, register the new tools with ToolRunner (like tool_runner.register("search", search_func)). Use wrappers for tools that need dependencies, similar to how verify_wrapper is done
GitHub
. Write unit tests for these tools (e.g., ensure search returns expected hits given a test index).

PR 3: Prompt & Grammar Tuning
Files: src/utils/prompt.py (new templates), src/utils/gbnf.py (if new grammars added), src/core/settings.py (add model context settings), possibly model loader config for grammar usage.
Description: Develop the system prompt for the agent (bilingual if needed) instructing the ReAct format and tools usage (as in the code snippet we saw, but updated to include only allowed tools and any additional rules)
GitHub
GitHub
. Create a GBNF grammar for the SearchPlan JSON if not already done (there’s likely one in memory; ensure it matches the JSON schema with any new fields)
GitHub
. Also consider a grammar or function schema for the agent’s final answer – e.g., a simple JSON with answer and sources – if we decide to use structured output for final answer. Alternatively, since we use footnote format, final answer can remain free-form text (with post-validation). Update the planner’s prompt to better handle Telegram context (for example, example: “If user asks for last month, include metadata_filters with that range”). Include few-shot examples in planner prompt illustrating date parsing and multi-query. The JSON-schema for plan is already likely defined; just verify MAX_PLAN_SUBQUERIES etc. are set appropriately (env var). Verify that when grammar mode is on (for llama.cpp), no stop tokens are set (per recommendations)
GitHub
GitHub
. This PR focuses on model prompt adjustments to ensure the agent and planner behave as expected. Test the planner output on various sample queries and adjust until JSON parses reliably. Test the agent’s chain on a couple of scenarios (maybe using GPT-4 in the loop initially to validate the logic before trying local models).

PR 4: Evaluation Toolkit
Files: scripts/evaluate_agent.py (new), docs/ai/eval_plan.md (new documentation for evaluation process), datasets/eval_questions.json (new).
Description: Create a small evaluation set of queries relevant to our Telegram data. This might include 10-20 questions that cover: date-range queries (“What did channel X post last week about topic Y?”), multi-hop questions (“Find when X mentioned Y and what was said”), fact-check style (“Is it true that … ?”), and a few simple ones. For each, if possible, have a known correct answer or at least a description of what to look for. Implement evaluate_agent.py to run both the old QA and new Agent on these queries and collect metrics. Metrics to calculate: Recall@5 (did the retrieved context contain the answer?), Precision of answer (perhaps manual check or using the verify tool on the answer), Citation coverage (fraction of answer’s statements supported by sources), and latency. We can set threshold goals: e.g., require that for 80% of queries, the agent’s answer has full support (coverage ~100%) whereas baseline might be lower; or that if baseline missed an answer, agent finds it. Also measure any increase in latency (aim <2x baseline median, e.g., if baseline ~2s, agent ~4s for complex query). Define “go/no-go” criteria: e.g., if agent answers are more accurate on at least 70% of eval questions and not significantly worse on any, we proceed. Document these in eval_plan.md. This PR is about ensuring we have a way to quantitatively and qualitatively verify the improvements before flipping the switch.

PR 5: Performance & Optimization
Files: src/core/settings.py (tweak thread counts or GPU usage), src/utils/ranking.py (if adding ONNX model loading for reranker), Dockerfile or compose if needed to include new dependencies (e.g. dateparser for temporal_normalize, any new model files).
Description: Address any performance bottlenecks discovered. For instance, enable model quantization or GPU acceleration: we can allow LLM_GPU_LAYERS env to load some layers of the model on GPU if available (already in settings)
GitHub
. Possibly switch to using a faster embedding library for MMR similarity (maybe precompute embeddings for retrieved docs to avoid re-embedding during MMR each time – though likely negligible cost for 40 docs). Ensure ToolRunner’s thread pool doesn’t conflict (we might raise max_workers). Implement caching for tool results where sensible: e.g. cache recent search results for identical subqueries (with TTL 5 min)
GitHub
. This prevents repeating work if agent asks the same thing again in verify or second round. Confirm memory usage is within limits (the new agent will keep some conversation state but we clear it each query except the Thought/Action history within one query). Update documentation in README or new docs to reflect the new capabilities and how to enable/use them.

Each PR will include tests: unit tests for tools, integration test for a full agent query (could simulate with a smaller model). We will use feature flags so that the agent mode can be toggled easily (and default it to off initially in .env). Deployment-wise, since we are not changing existing endpoints, rolling this out should not require downtime. We’ll run the new container alongside, test the /v1/agent endpoint, and once confident, we can suggest clients to use it for complex queries. In production, we might keep both: use agent for queries detected as complex and fallback to direct QA for simple ones (this could even be automated via the router or query planner detecting complexity).

Evaluation Plan

Dataset: We will curate a small evaluation dataset from a few representative Telegram channels (for example, official news channel, a tech updates channel, etc.). For each, we’ll come up with questions that test various aspects: (1) straightforward fact lookups (“When did channel ABC first mention keyword X?”), (2) questions requiring multi-step reasoning (“What did person P say about topic Q and what was the context?” – requiring gathering multiple messages), (3) temporal queries (“Summarize what happened in channel XYZ in March 2023”), (4) verification challenges (“Did channel A confirm the release of product Z?” where the correct answer might be “No, they denied it” – testing that the system doesn’t hallucinate a yes), and (5) cross-channel or cross-source queries if applicable. We expect ~20 questions with a short reference answer or description for each.

Metrics: We will use a combination of automatic and manual metrics. Key ones include:

Answer Accuracy/F1: Does the answer contain the correct information? (Manual judgment for each question, since we often don’t have a single exact answer string.)

Faithfulness (No Hallucination): We’ll use the citation precision metric – count of answer statements that are actually supported by provided sources
docs.aws.amazon.com
. Ideally, precision = 100% (everything stated has evidence). We’ll flag any hallucinated info.

Coverage/Recall: Using the citation coverage concept
docs.aws.amazon.com
 – did the answer use most of the relevant info from the sources? And did the retrieval bring in the needed sources? We can measure if the known relevant message for the question was retrieved in top-5. If not, that’s a miss.

Conciseness & Correctness: We can measure answer length vs expected. If the agent tends to verbose or wander, that’s an issue. We want it to be to the point and correct. Possibly use a "helpfulness" or coherence criterion in judging output (subjective).

Latency: measure the average and P95 time for the agent pipeline vs baseline QA. E.g., if baseline answers in ~2s and agent in ~5s for complex queries, that might be acceptable given the improved results. But if agent sometimes spikes to 15s due to tool timeouts or multiple steps, that’s a concern. We set a soft goal that typical agent query < 10s and virtually all < 30s
GitHub
.

Procedure: For each question in the eval set, run the baseline QA (with include_context=true to see what it used) and the agent (with streaming disabled for measurement, or measure until final answer token). Compare results side by side. We expect improvements especially on multi-step and verification questions. For example, baseline might give an incomplete answer or say “I’m not sure”, whereas agent finds the info through iterative search. We will tabulate results. If any regression is found (e.g. agent misses something baseline got, or agent gives a wrong answer), analyze why (was it a tool failure? planning issue? etc.) and adjust prompts or logic.

We’ll also evaluate the user experience via a small A/B test: have a few team members use the system interactively in agent mode vs normal, to ensure the step-by-step doesn’t introduce odd delays or content. Since the agent can output intermediate steps to SSE (like “Thought” and “Observation”), we will likely filter those for end-users (maybe only send final answer and possibly some live progress indicator). The logs however will contain the full trace for debugging.

Success Criteria: We will consider the ReAct-RAG integration ready to enable by default if:

On our eval set, the agent answers at least 70% of questions correctly/supportively, vs baseline perhaps 50% (for complex ones).

The hallucination rate (answers with unsupported claims) drops significantly (ideally to zero obvious hallucinations in the test).

Latency remains within 2x of baseline on average, and all tool calls succeed within timeouts in >90% of cases (we will monitor the logs for timeouts and errors).

No major new errors introduced (crashes, etc.). Memory overhead should be acceptable (maybe +10–20% for running two LLMs – planner and main – concurrently)
GitHub
.

We’ll document these results. If some criteria are not met (e.g., agent is too slow or occasionally wrong), we can keep the feature optional (behind a flag or only for certain queries) until improvements are made.

Risks and Rollback Strategy

Integrating an agentic system introduces complexity, so we identify risks and mitigation:

Performance Degradation: Each agent query could use multiple LLM calls (planner + answer, maybe rewrite or verify) and multiple retrieval ops. This could tax the CPU and increase response times. If under load the latency becomes unacceptable, we have a feature toggle to disable the agent and revert to the simpler QA path. Our design includes graceful degradation: e.g., if the planner fails or times out (we set ~8s timeout)
GitHub
, we can fall back to a single-query search using the raw user query. If a tool call times out, the agent gets an error and we have it either try a simpler approach or finish early with whatever info it has
GitHub
. We also limit rounds to prevent runaway latency (max 2 rounds). We will implement a “kill switch” env var (like ENABLE_AGENT) so ops can turn it off without redeploying, in case of any incident. Additionally, if the agent is on and experiences heavy load, the system can temporarily route queries to QAService (maybe based on a simple check like queue length or an environment health metric) – basically a dynamic fallback to ensure uptime.

Tool Failures & Errors: Since tools execute code (search, etc.), a bug or exception in a tool could derail the agent. We log every tool invocation with success/failure
GitHub
GitHub
. If a particular tool is causing issues (e.g. web_search if internet is down), we can disable it via an allowlist in the AgentRequest or config (there’s tools_allowlist in schema)
GitHub
. The agent prompt can also list only safe tools. In worst case, we catch exceptions in AgentService and return a fallback answer or error message gracefully to the user. This prevents the entire app from crashing if, say, translate tool had an issue.

Quality Risks (Agent mistakes): The agent might sometimes pick a wrong tool or ask irrelevant queries (especially depending on LLM behavior). Our use of a deterministic planner mitigates this for search queries. But the agent could still do something silly like ask the math_eval tool when not needed, or misinterpret a date. We guard this by carefully writing the system prompt with tool descriptions and by limiting tool set (no harmful tools). Also, we plan to initially use high-accuracy LLM (possibly GPT-4) for the agent to validate the approach, and then fine-tune the prompt/grammar for local models. If a pattern of errors emerges (like always doing an unnecessary second search), we can adjust logic (for example, have the agent code itself decide to skip some tool calls based on simple heuristics instead of leaving all decisions to LLM). Essentially, we keep a bit of rule-based control as backup to the agent’s reasoning – e.g., enforce that after 2 unsuccessful searches, stop. These are our guardrails to avoid infinite loops or nonsense actions.

Resource Usage and Costs: Running two LLMs (planner and main) means more memory and CPU. On CPU-only deployment, if both run simultaneously it could slow things. We have set it so that the planner (3B model) runs first quickly and finishes before the main 20B LLM starts generating answer. So they don’t use CPU at the same time heavily
GitHub
. We also allow partial offload to GPU if available to speed up the 20B model. The reranker and embedding models are also running – but those are small and fast. We will monitor system metrics. If memory becomes an issue (20B model might be ~10GB, 3B ~3GB, plus indexes), we might unload the planner model from memory when not in use (since it’s quick to reload or run in 4-bit). We’ll also make sure that the agent’s additional context (thoughts/actions) doesn’t blow up the token count – by not appending entire documents into the “Observation” (we’ll only show concise observations). If needed, we can shorten or drop the Thought text in the prompt history to save tokens (since we trace it in logs anyway).

User Perception and SSE: With the agent, intermediate steps might be visible if not handled properly. We plan to emit SSE events for thought, tool_invoked, observation, etc. for transparency
GitHub
GitHub
, which can be useful for debugging or power-users. However, general end-users might be confused by the thought process. We likely will filter SSE events on the client side: e.g., only display the final answer, perhaps with a loading spinner during the reasoning. We can also provide an option to see the reasoning (for developers). The risk is minimal since it’s just a UI/UX decision.

Security: We must ensure no prompt injection lets a user commandeer the agent. The agent system prompt explicitly sets the format and rules. We also use a security filter on user queries (security_manager.validate_input) to strip dangerous content
GitHub
. Tools like math_eval are sandboxed and only allow safe operations
GitHub
. We will maintain a strict schema for tool inputs to avoid any SQL or code injection (the grammar excludes problematic characters)
GitHub
GitHub
. If any such attempt is detected, the agent should refuse. These measures keep the system robust.

Rollback: If any serious issue arises in production (e.g., agent answers are consistently wrong or timeouts flood the logs), we can instantly switch off the agent via environment (since existing /qa endpoint is intact, we just direct users to use that). The code is structured so that even if /v1/agent fails, it doesn’t affect /v1/qa. So rollback is simply to stop using the new endpoint or set ENABLE_AGENT=false to even disable the planner and agent internally. In case we had replaced the QA with agent logic, we’d instead have included a runtime flag to bypass agent and use legacy pipeline as fallback. But given our plan, such a situation is handled by keeping them separate. We will also keep monitoring logs and possibly implement an automatic cutoff: e.g., if agent error rate > X, the service can temporarily route queries to QAService.

Overall, the integration is done in a controlled, opt-in manner, with multiple safety nets (caches, timeouts, step limits, and feature flags). The existing functionality remains an immediate fallback path, which ensures we can recover from any unforeseen problems quickly, maintaining service continuity.

Appendices

A. Sample Prompt Schema (Query Planner)
(An excerpt of the JSON Schema for SearchPlan and its GBNF grammar)
GitHub

// JSON Schema for SearchPlan (simplified)
{
  "type": "object",
  "properties": {
    "normalized_queries": {
      "type": "array", "minItems": 3, "maxItems": 6,
      "items": { "type": "string", "minLength": 1 }
    },
    "must_phrases": {
      "type": "array",
      "items": { "type": "string" }
    },
    "should_phrases": {
      "type": "array",
      "items": { "type": "string" }
    },
    "metadata_filters": {
      "type": "object",
      "properties": {
        "date_from": { "type": "string", "format": "date" },
        "date_to": { "type": "string", "format": "date" },
        "channel": { "type": "string" }
      },
      "additionalProperties": false
    },
    "k_per_query": { "type": "integer", "minimum": 1, "maximum": 50 },
    "fusion": { "type": "string", "enum": ["rrf","mmr"] }
  },
  "required": ["normalized_queries"]
}


GBNF Example:

root ::= "{" ws "\"normalized_queries\"" ws ":" ws "[" query_list "]" // ...other fields...
query_list ::= string (ws "," ws string)*
string ::= "\"" utf8safe* "\""
utf8safe ::= [\u0020-\u007E\p{L}\p{Nd}] - ["\\<>;`$]  // allowed chars


(This grammar ensures the LLM outputs a JSON with queries and optional filters, avoiding problematic characters in strings
GitHub
.)

B. Example Agent Dialogue
(This illustrates how the agent might operate on a real query, showing Thought/Action/Obervation, though in practice we might not display all to end-user.)

User query: “Какие важные объявления делал @durov в Telegram в августе 2023?” (What important announcements did @durov make on Telegram in August 2023?)

Thought: I should find posts by Durov in Aug 2023 and see which are major announcements. Likely need date filter and search the channel @durov.

Action: router_select {"query": "дуров август 2023 анонсы"}

Observation: {"route": "bm25", "reasons": ["contains date"]}

Thought: Router suggests BM25 (the query has a date). I’ll plan a search query specifically for August 2023 in Durov’s channel.

Action: query_plan {"query": "важные объявления дуров август 2023"}

Observation: {"normalized_queries": ["дуров август 2023 важные анонсы", "дуров объявления август 2023"], "metadata_filters": {"date_from":"2023-08-01","date_to":"2023-08-31"}, "k_per_query": 10, "fusion": "rrf"}

Thought: The plan gave two queries and an August date filter, using RRF fusion. Let’s retrieve messages.

Action: search {"queries": ["дуров август 2023 важные анонсы","дуров объявления август 2023"], "filters": {"date_from":"2023-08-01","date_to":"2023-08-31","channel":"durov"}, "k":10}

Observation: {"hits": [ {"id":"durov/1234","score":11.5,"text":"...Durov announced ... Wallet...","date":"2023-08-13"}, {"id":"durov/1250","score":9.7,"text":"...introduced Stories...","date":"2023-08-23"}, ... ] }

Thought: I got some hits: one on Aug 13 about Wallet, one on Aug 23 about Stories feature. These sound like major announcements. I should fetch details.

Action: fetch_docs {"ids": ["durov/1234","durov/1250"]}

Observation: {"docs": [ { "id":"durov/1234", "text": "On August 13, Pavel Durov announced the launch of a self-custodial Wallet feature ...", "metadata": {...}}, { "id":"durov/1250", "text": "On August 23, Pavel Durov introduced Telegram Stories for all users ...", "metadata": {...}} ] }

Action: compose_context {"docs": [docs above], "max_tokens_ctx":1800}

Observation: {"prompt": "[1] On August 13, Pavel Durov announced the launch of ...\n[2] On August 23, Pavel Durov introduced Telegram Stories ...", "citations": [ {"id":"durov/1234","index":1}, {"id":"durov/1250","index":2} ] }

Thought: I have two relevant announcements from that period. Coverage looks good (two major announcements). I will now answer summarizing these, with references.

FinalAnswer: Pavel Durov made several major announcements in August 2023. For example, on August 13, 2023 he announced a self-custodial Wallet feature for Telegram users【1】. Later in the month, on August 23 he introduced Telegram Stories for all users as a new functionality【2】. These were among the important updates he shared in that timeframe.

(The final answer provides the info with citations [1] and [2] linking to the fetched messages.)

C. Sources and References

rag_app README (2025) – Project overview of Query Planner, Hybrid search (BM25 + Chroma with RRF/MMR), reranker, SSE streaming, and a roadmap for ReAct tool integration
GitHub
GitHub
. (Describes existing features and hints that Planner+Hybrid+Reranker are the foundation for adding tools like search() and verify().)

ReAct/RAG Research Survey (Sep 2025) – Internal research document summarizing best practices for agentic RAG on CPU
GitHub
GitHub
. Covers use of structured outputs (GBNF grammars for LLM), multi-query strategies (4–6 paraphrases, RRF fusion)
GitHub
, MMR for diversity
GitHub
, and recommends a CPU cross-encoder reranker for top-20 results
GitHub
. Also emphasizes caching and step limits for robust agent loops
GitHub
.

Agent Tools Documentation – Explains the design of the tool subsystem for the agent
GitHub
GitHub
. It lists available tools and their JSON I/O schemas (e.g. router_select returns a route, compose_context returns a prompt with citations, verify returns a boolean and confidence)
GitHub
GitHub
. This shows how the system isolates tool execution and logs each action, which we leverage in our design.

ReAct Agent ADR (Dec 2024) – Architecture Decision Record for introducing the ReAct Agent API
GitHub
GitHub
. It confirms the creation of a separate AgentService with step-by-step reasoning, streaming events, and fallback to the QA service
GitHub
. Key principles: non-intrusive (existing QA unchanged), reuse of components, observability via JSON traces, and ability to disable via config
GitHub
GitHub
. This guided our integration approach (new endpoint, reuse dependencies, feature flag).

ReAct Playbook v1 – Early internal playbook of ReAct+RAG patterns (2024)
GitHub
GitHub
. Stresses a “lightweight cycle” with fixed 3 iterations max and that the planner should generate all subqueries in one go to minimize LLM calls
GitHub
GitHub
. It also details the SearchPlan JSON format (with fields for queries, filters, etc.)
GitHub
 and suggests grammar constraints for plan and final answer
GitHub
GitHub
. We followed these guidelines for structured outputs and limited steps.

Weaviate Blog – What is Agentic RAG (Nov 2024) – Explains the concept of agentic RAG and motivations
weaviate.io
weaviate.io
. Highlights that vanilla RAG does one-shot retrieval and lacks validation, whereas agentic RAG introduces an agent that can choose tools, formulate queries, and iterate until satisfied
weaviate.io
weaviate.io
. This bolsters our rationale for adding an agent: to allow multiple knowledge sources and self-checking of retrieved context.

Haystack Docs – HyDE (2023) – Documentation on Hypothetical Document Embeddings
docs.haystack.deepset.ai
. Describes how HyDE prompts an LLM to generate a fake document from the query, embeds it, and uses that to retrieve real docs, which can improve recall in zero-shot scenarios
docs.haystack.deepset.ai
docs.haystack.deepset.ai
. We cite this as a potential strategy when normal retrieval fails to find relevant info.

AWS Blog – Cohere Rerank for RAG (Sep 2024) – Discusses using Cohere’s reranker to improve search result quality
aws.amazon.com
. It explains that rerankers assign a relevance score by deeply comparing query and document, often pulling up a result that might have been low-ranked by a simpler similarity measure
aws.amazon.com
aws.amazon.com
. This supports our inclusion of a cross-encoder reranker to boost answer precision by reordering top candidates.

LanceDB Benchmark (2023) – Experiment comparing rerankers
blog.lancedb.com
. Found that applying a reranker (like Cohere) after vector search gave ~8% improvement for a BGE embedding retriever
blog.lancedb.com
. Also noted hybrid search with reranker greatly improved accuracy (90%+ recall) compared to no rerank. This empirical evidence underlies our decision to include a rerank stage for quality.

AWS Bedrock Knowledge Base Eval (Aug 2023) – Defines evaluation metrics for RAG systems
docs.aws.amazon.com
docs.aws.amazon.com
. Introduces citation precision and citation coverage: precision = fraction of cited info that is actually relevant, coverage = fraction of answer supported by citations
docs.aws.amazon.com
docs.aws.amazon.com
. We intend to use these metrics to assess and enforce answer faithfulness (e.g. requiring high coverage). This reference provides clear definitions to guide our verification and evaluation strategy.