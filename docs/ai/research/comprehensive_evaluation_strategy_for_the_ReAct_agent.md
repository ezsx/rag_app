Executive Summary
In this research, we design a comprehensive evaluation framework for the ReAct-based RAG agent that answers questions using Telegram channel data. The goal is to quantitatively measure the agent’s answer quality, source-grounding, coverage of information, and efficiency, and to compare its performance against a baseline QA system. We propose a combination of automated metrics (for accuracy, citation support, recall, etc.) and LLM-assisted judgments to assess complex aspects like factual correctness and faithfulness. Key evaluation metrics include Answer Correctness (does the answer accurately and fully address the question), Faithfulness (are all answer statements supported by cited Telegram messages), Coverage/Recall (does the agent retrieve and use all relevant information, e.g. was the needed message in top-5 results), Conciseness (is the answer free of unnecessary text), and Latency (performance speed). The evaluation methodology leverages best practices from recent RAG research – using both traditional metrics and LLM-as-judge techniques – to ensure a thorough assessment
docs.langchain.com
patronus.ai
. We outline an architecture for an evaluation script that runs a suite of test queries through both the ReAct agent and the baseline QA API, computes the defined metrics automatically, and produces a structured report. This will enable side-by-side Agent vs Baseline comparison, highlight the agent’s improvements on complex queries (multi-step, filtered by date/channel) and check for regressions on simple queries. Ultimately, the evaluation results will inform go/no-go decisions for production deployment, with clear success criteria (e.g. ≥70% answer accuracy, high citation precision
docs.aws.amazon.com
, acceptable latency) and diagnostic insights into any failure modes. The outcome is a practical, data-driven basis for validating the agent’s quality and guiding further improvements before release.
Методология оценки
Overview of Metrics and Evaluation Methods: We adopt a multi-dimensional set of metrics to capture different aspects of answer quality, aligned with academic and industry best practices
patronus.ai
. For each test question, we will evaluate:
Answer Accuracy (Correctness & Completeness): Does the agent’s answer contain the correct information and address all parts of the query? This is akin to the answer correctness metric
patronus.ai
, measuring factual accuracy and completeness. If a ground truth answer is available for the query, we will use it for automatic comparison. This can include computing an exact match or token-level F1 score if the answer is a short fact, or using semantic similarity for longer answers. However, because answers may be phrased differently, we plan to leverage LLM-as-a-judge evaluation – i.e. ask a strong LLM to compare the agent’s answer to the reference and judge if it’s correct
docs.langchain.com
. The LLM judge can consider correctness and completeness: for example, an answer that only covers part of the question would be marked incomplete (even if factually correct on what it does state)
docs.aws.amazon.com
. In absence of a reference answer (for open-ended queries), we can prompt an LLM with the question and the agent’s answer to assess if the answer fully and correctly addresses the user’s query. This automatic judging reduces manual effort while approaching human-level evaluation consistency. We will validate the LLM’s evaluations on a small sample to ensure reliability, and any ambiguous cases will be flagged for human review.
Faithfulness (Citation Precision / No Hallucination): This metric checks that every factual claim in the answer is supported by the retrieved Telegram sources, ensuring the agent isn’t hallucinating or introducing unsupported info. It corresponds to answer hallucination (which we want to minimize) in RAG metrics
patronus.ai
 and to AWS’s citation precision
docs.aws.amazon.com
. We define citation precision as the fraction of the answer’s statements that are correctly backed by a cited source
docs.aws.amazon.com
. Formally, if we break the answer into distinct claims, each claim should be inferable from one of the cited Telegram messages. The score is calculated as: supported_statements / total_statements
docs.ragas.io
. A 100% score means every claim has evidence in the citations (perfect faithfulness). To measure this automatically, we will implement a claim verification process:
Extract Claims: Split the agent’s answer into individual statements or facts (e.g. by sentence or clause).
Check Support in Sources: For each claim, check the content of the cited Telegram messages to see if the claim can be found or inferred there. This can be done by searching for key phrases in the source text or using a lightweight natural language inference model. We may use an open-source factual consistency model (such as Vectara’s HHEM hallucination detector) to automate this check
docs.ragas.io
, or an LLM prompted to read the claim and source and decide if the source supports the claim.
Calculate Precision: Compute the ratio of supported claims to total claims
docs.ragas.io
. For example, if the answer makes 5 statements and 4 are supported by the cited Telegram posts (and one is unsupported or incorrect), citation precision = 4/5 = 80%.
A high precision indicates the answer stays faithful to the retrieved evidence (high factual consistency
docs.ragas.io
), whereas a low value flags potential hallucinations or misuse of sources. Where possible, we’ll automate this fully; any claims the model flags as unsupported can be manually double-checked in a later review step. Our goal is to ensure faithfulness ~100% for the agent (the agent is designed to always cite sources for factual claims), and to catch any instances where the agent might have cited a source but misstated the info. If the baseline QA does not provide citations, we will still evaluate its faithfulness by running a similar verification: we can search the Telegram data for statements in the baseline’s answer to see if they appear (using the agent’s verify tool or a custom search). This way, we can penalize baseline answers that contain unsupported content.
Coverage and Recall of Information: We assess whether the agent retrieved and utilized the full scope of relevant information for the query. This has two facets: (a) Did the retriever get the right sources (documents)? and (b) Did the answer incorporate all the important info from those sources? For (a), we use a Recall@5 metric on the retrieval results: we define a set of expected relevant documents for each question (the Telegram messages that contain the answer or key facts). Then we check if the agent’s top-5 retrieved hits include at least one of those ground-truth documents
patronus.ai
. If multiple documents are needed to answer a question, we can measure recall as the fraction of those docs present in top-5. This is a retrieval recall indicator – e.g. a recall@5 of 100% means the agent’s hybrid search returned all the needed messages in the first five results. This helps diagnose retrieval failures: if a relevant message was missing in the results, the agent could not possibly answer correctly (a coverage gap in retrieval). For (b), on the generation side, we evaluate citation coverage (sometimes called citation recall). Citation coverage measures how well the answer is supported by all relevant sources
docs.aws.amazon.com
. We define it as the fraction of ground-truth relevant documents that are actually cited in the final answer. For instance, if a question’s answer is spread across 3 Telegram posts (ground truth) but the agent only cited 2 of them, citation coverage = 2/3 = 66%. Higher coverage means the agent used more of the pertinent information available. Another perspective on information coverage is answer completeness: ensuring the answer includes all key facts needed. If we have a reference answer or a checklist of points expected, we can compute a recall on content (how many of the expected points were present in the agent’s answer). This parallels the concept of context sufficiency
patronus.ai
patronus.ai
 or completeness – the retrieved context and answer together should cover all aspects of the query. To automate this, we might break the reference answer into facts and see how many are found in the agent’s answer (similar to the approach for faithfulness but comparing answer vs ground truth instead of vs sources). In summary, the agent should have high recall at the retrieval stage (to find the right info) and high coverage at the answer stage (to use that info fully). If we find coverage gaps (e.g. agent missed a piece of information present in the data), those are noted as answer shortcomings. By contrast, the baseline QA will also be checked: did it retrieve the needed doc (if we can tell), and does its answer miss parts of the answer? These coverage metrics directly tie to our success criteria (e.g. we expect the agent to cover ~100% of relevant info for ~80% of queries).
Conciseness and Relevance of Answer: We want the agent’s answers to be concise – meaning they are as brief as possible while remaining complete and correct, without extraneous information. This is related to answer quality factors like readability and helpfulness, but specifically we want to avoid overly long or off-topic answers. We will measure conciseness primarily by answer length and content analysis. Concretely, we can record the number of tokens or sentences in each answer and compare it to an expected range. If an answer is significantly longer than the reference answer or includes a lot of irrelevant detail, it will score lower on conciseness. One simple automatic metric is to set a length threshold (based on the question type or reference): e.g. if the agent’s answer is more than 2-3x the length of the reference answer or contains obvious digressions, mark it as not concise. We might also use an LLM judge to rate conciseness/fluency, or check for repeated/redundant statements. For instance, we could prompt an LLM: “Given the question and the answer, evaluate if the answer is concise and free of unnecessary info, yes or no.” However, a reliable automated conciseness metric is tricky – we will likely use a combination of length heuristics and manual inspection for outliers. The evaluation output will highlight answers that are very long or include information that wasn’t asked (which could indicate the agent included irrelevant context). We expect the agent to generally keep answers brief (the system prompt even instructs it to use at most 3-5 sentences in an answer), so any divergence here is important to catch. This metric ensures the agent’s answers are not only correct but also to-the-point and user-friendly. The baseline QA’s answers can serve as a reference for conciseness – if baseline answers are typically shorter, we’ll be cautious that the agent doesn’t introduce verbosity. Ideally, the agent should balance completeness with brevity, providing just enough detail from Telegram messages to answer the question.
Latency and Efficiency: We will measure the execution time of the agent vs baseline for each query. Latency is critical for user experience, so we’ll record the total time from query submission to final answer for both systems. The agent uses multiple steps (LLM thought, tools, verification), so it may be slower than the single-shot baseline; our goal is to see if the agent stays within acceptable bounds (e.g. typical <10s, p95 <30s as specified). We’ll compute average and 95th percentile latency for the agent and compare to baseline. Each tool invocation in the agent returns a took_ms metadata, but we will simply time the overall request to capture any overhead outside tools as well. Additionally, we’ll track Refinement Count (how many extra search rounds the agent performed) and the number of reasoning steps used, since these impact latency. This is more of a diagnostic metric: ideally most queries are answered in the initial cycle without needing refinement (since max_refinements=1 in design). If we see many queries hitting the refinement or verification loop, that might explain higher latency and indicate queries where the agent struggled initially. The baseline QA likely always does one retrieval and answer step, so its latency is expected to be lower; however, if the agent’s added time yields significantly better accuracy on hard questions, that trade-off may be acceptable. We will note any extreme cases (e.g. if any query took the agent excessively long or timed out, or any errors). By monitoring latency alongside accuracy, we ensure the agent’s improvements do not come with impractical response times.
Automated vs. Manual Evaluation: Our approach emphasizes automation to handle the bulk of evaluation objectively and efficiently. Metrics like retrieval recall, citation support, and latency are computed automatically with no human intervention. Even complex metrics like answer correctness and faithfulness can be largely automated using LLM-based evaluators (GPT-4 or similar) or specialized models, as described. However, we acknowledge that for certain complex queries and borderline cases, manual inspection is valuable. For instance, if the agent and baseline give different answers or if an LLM judge is uncertain about correctness, a human evaluator should review the question, the data, and the answers to make a final judgment. We will design the evaluation output to facilitate quick manual review: for each question, we can output a summary including the question, the agent’s answer with its citations, the baseline’s answer, and flags/metrics (e.g. “Agent answer possibly incomplete” or “Baseline answer unsupported by sources”). This structured result will make it easy for a human to spot check and verify the problematic cases. We expect manual checking to be needed mostly for cases where the metrics disagree or something is unclear (for example, the agent’s answer is correct but phrased very differently from the expected answer – the LLM judge might give a low score, needing a human to confirm if it’s actually correct). By focusing manual effort only on outliers or failures, we minimize the human workload. The majority of answers can be validated through the automated pipeline, and the system can highlight, say, the 10-20% of questions that need human double-checking. Over time, as we refine the LLM judging prompts and perhaps incorporate more ground truth data, the process can become even more automated. Special Considerations for Telegram Data Queries: Evaluating our agent on Telegram data brings some unique factors we must account for in our methodology:
Temporal Filters: Many queries involve time constraints (e.g. “за последний месяц” or a specific date range like August 2023). The evaluation must check whether the agent correctly interpreted and applied these time filters. For automated checking, we can parse the query for date keywords (e.g. month/year or relative terms like “last month”) and then verify the dates of the cited messages in the agent’s answer. The Telegram message metadata includes a date field (timestamp). For example, if the question asks for Durov’s announcements in August 2023, we will confirm that every message cited by the agent has a date in August 2023. If the agent cites something outside that range, that’s an error (even if the content is about an announcement, it violated the filter). We can similarly ensure the agent’s answer only includes information up to the specified date range. This automated date verification will catch issues where the agent might have ignored or misunderstood the temporal aspect. Baseline answers will be harder to check this way (if no citations), but if baseline mentions a date or event outside the range, we can catch it via content too. Temporal context also matters for correctness: if a query says “за последний месяц” (last month) but our evaluation dataset is static, we’ll interpret it relative to a fixed point (likely the dataset or query creation date). We might need to freeze the interpretation of “last month” to a specific month for evaluation consistency (e.g. if the test dataset was created on Oct 1 2025, “last month” refers to September 2025). Such assumptions will be documented in the dataset so the expected answer and evaluation checks align.
Channel and Author Filters: Some queries specify a particular Telegram channel or author (e.g. “что обсуждали в канале @news…”, “что писал @username…”). Here, the agent’s ability to filter by channel or author is crucial. Our evaluation will include metadata checks: each cited document has channel and possibly author fields. We will verify that the agent’s citations come from the correct channel or author as requested. For instance, if asked about channel @news, ideally all citations should be from that channel (or at least the answer should clearly pertain to that channel’s content). If the agent cites other channels, that indicates a failure to follow the user’s filter instruction. Similarly for author: if we query about what a certain user wrote, we expect the agent to cite messages where author matches that username. We can automatically flag any citation that doesn’t match the requested channel/author. This ensures the agent’s answer is contextually relevant to the specified source. The baseline QA might not strictly enforce these filters (depending on implementation), so this is an area we expect the agent to outperform. Our evaluation will highlight if the agent indeed succeeds in focusing on the specified channel/author content.
Negative Queries (No Answer cases): Some questions may have no relevant answer in the Telegram data. For example, a query might ask about something that was never discussed in the channels. In such cases, a correct behavior would be for the agent (or baseline) to respond that it couldn’t find information (rather than hallucinate an answer). We will include a few such negative cases in the dataset and evaluate how they are handled. The metrics for a negative query are a bit different: accuracy would mean “correctly returned no answer”. We’ll treat a polite “no information found” response as correct. If the agent attempts an answer when none exists (hallucinating content), that’s a failure in both correctness and faithfulness. We will mark expected_answer as null/none and expected_docs as [] for these cases in the dataset. Automated evaluation can check if the agent’s answer is an empty/REFUSAL type or contains a disclaimer. We might use a simple heuristic: if expected_docs is empty, any non-empty answer is likely wrong (unless the agent says “I don’t know”). We will manually verify these cases. This tests the agent’s ability to gracefully handle unanswerable questions.
Multi-turn or Multi-step Reasoning: While our evaluation queries are primarily single-turn questions, some are inherently multi-hop (e.g. requiring combining info from multiple messages or an intermediate reasoning step). Our agent’s ReAct chain is supposed to handle these, and our metrics like coverage and correctness will reflect success. To diagnose issues, we can also log the agent’s step-by-step trace from SSE (thoughts, actions) to see where it might go wrong on complex queries. For example, if a multi-part question is answered incorrectly, the log might show whether the agent’s query_plan missed a sub-question or if it retrieved the right docs but compose_context didn’t include all of them (low citation coverage). While this isn’t a metric per se, it’s part of qualitative analysis: we will analyze failure cases to attribute them to root causes (retrieval vs reasoning vs citation problems). This diagnostic step is important for pattern identification – e.g., maybe the agent struggles with questions that require understanding a conversation thread (replies/forwards in Telegram) or synthesizing across channels. We will include notes in the report for such patterns.
In summary, our methodology combines quantitative evaluation metrics with targeted checks tailored to the Telegram domain. It leverages automation extensively (including LLM-based judgment for nuanced metrics) and uses manual review strategically for complex cases. By doing so, we ensure a thorough and efficient evaluation of the ReAct agent’s performance, covering accuracy, supportiveness of answers, completeness of information, brevity, and speed – all crucial for a high-quality QA system on Telegram content.
Архитектура решения
We propose an evaluation tool (script or service) that systematically runs the agent and baseline on a set of test queries and computes the metrics outlined. The architecture consists of the following components:
Evaluation Dataset: A structured dataset of evaluation queries, stored in a JSON or CSV format. Each entry will include the question and relevant metadata for evaluation. We design the dataset schema to capture everything needed for metrics and analysis:
query: string, the user question in Russian (or other languages if needed).
category: string, the category/type of the question (e.g. "temporal", "channel_filter", "author_query", "multi-hop", "simple_fact", etc.). This is used for later grouping and analysis of results by category.
expected_answer: string or null, an optional reference answer or a list of key facts that should be present. For straightforward factual questions, this might be a single correct answer sentence. For complex ones, it could be a summary of what the answer should contain (or left null if we only have source expectations). This field is primarily for correctness evaluation – if provided, the evaluation can compare the agent’s answer to this reference. In cases with no single phrasing possible, we might leave it empty and rely on expected_docs instead.
expected_documents: list of strings, the IDs of Telegram message documents that contain the answer information. These are the ground-truth relevant docs that a correct system should retrieve and use. For example, if a question asks for announcements by Durov in August 2023, this list would include the IDs of all Telegram messages by Durov in Aug 2023 that contain announcements. This field is crucial for measuring retrieval recall and coverage – it’s our “gold set” of evidence. It can be empty for questions with no answer.
answerable: boolean, indicating if the question is answerable from the data or not. (This could be derived from expected_documents being non-empty, but we may include it for clarity. answerable: false would mark queries where no correct answer exists in the data).
notes: string, any additional context or expectations (e.g. “User expects a yes/no answer” or “Multiple possible answers exist”). This won’t be used in automated metrics but helps in manual review.
An example dataset entry in JSON might look like:
{
  "query": "Какие важные объявления делал @durov в августе 2023?",
  "category": "temporal+author",
  "expected_answer": "В августе 2023 года Дуров объявил о запуске функции ... и рассказал о ...",
  "expected_documents": ["tg://12345", "tg://12346", "tg://12350"],
  "answerable": true
}
And a negative example:
{
  "query": "Был ли @telegram down 1 января 2020?",
  "category": "negative",
  "expected_answer": null,
  "expected_documents": [],
  "answerable": false
}
This dataset will be prepared beforehand (possibly starting with the existing 8 questions and expanding). Where needed, we’ll manually identify the expected documents and answers for each query to serve as the “ground truth” for evaluation.
Execution Engine: A Python-based runner that goes through each query and collects results from both the agent and baseline:
Agent Query Execution: We will call the ReAct agent via its SSE streaming endpoint (/v1/agent/stream). The tool will need to capture the stream of events. In implementation, we can use an SSE client library or the requests library in stream mode to receive events chunk by chunk. The runner will reconstruct the agent’s final answer and gather relevant metadata:
Final answer text (from the final SSE event’s answer field).
Citations used (from final.citations, which includes the list of doc IDs and metadata the agent provided).
Coverage (from final.coverage which is the agent’s internal citation coverage metric).
Verification info (from final.verification, e.g. whether the agent verified the answer and confidence score).
Refinement count (from final.refinements field).
Additionally, the runner can log intermediate info: the search results or tool steps. For example, it can listen for the search tool’s observation event which contains the hits. We might extract the top-5 hit IDs from there for retrieval recall calculation. All these pieces will be stored in an in-memory result object for that query. We’ll also timestamp the start and end of the agent call to measure latency.
Baseline Query Execution: Then, the runner calls the baseline QA endpoint (/v1/qa) with the same query. We capture the baseline’s answer text (and any metadata it returns, such as confidence or source if provided). We’ll time this call as well. If the baseline returns a list of answers or some structured output, we’ll adapt accordingly, but likely it returns a single answer string. We will record whether the baseline returned an answer or not (if the baseline has a notion of unanswered, e.g. “No information” message, we note that).
The execution for each query is sequential: Agent → Baseline → Metric computation. To pseudo-code this process:
results = []
for q in dataset:
    agent_start = time.time()
    agent_events = call_agent_stream(q.query)
    agent_time = time.time() - agent_start
    agent_final = parse_final_event(agent_events)
    # agent_final contains answer, citations list, coverage, refinements, verification
    
    baseline_start = time.time()
    base_response = call_baseline_api(q.query)
    base_time = time.time() - baseline_start
    base_answer = base_response.answer
    # (and maybe base_response.citations or similar if available)
    
    result = {
        "query": q.query,
        "agent_answer": agent_final.answer,
        "agent_citations": agent_final.citations,  # list of {id, metadata, index}
        "agent_coverage": agent_final.coverage,
        "agent_verified": agent_final.verification.verified,
        "agent_confidence": agent_final.verification.confidence,
        "agent_refinements": agent_final.refinements,
        "agent_latency": agent_time,
        "agent_steps": agent_final.step_count,  # could be length of events or final total_steps
        "baseline_answer": base_answer,
        "baseline_latency": base_time
    }
    results.append(result)
This loop produces a raw results list where each entry has the answers and metadata needed to compute metrics.
Metric Computation: After obtaining the raw outputs, the next part of the script computes the defined metrics for each query and aggregates them:
Answer Correctness: If expected_answer is provided for the query:
We can use a simple text similarity or exact match for very short answers (e.g. if expected answer is “Yes” or a name).
For longer answers, we will call an LLM (like GPT-4 via API) with a prompt asking it to compare the agent’s answer to the expected answer. The prompt might be: “Question: {Q}\nExpected answer: {E}\nAgent answer: {A}\nIs the agent's answer correct and complete compared to the expected answer? Respond with 'Correct' or 'Incorrect' and a brief reasoning.”. We’ll parse the LLM’s judgment. Alternatively, we could ask for a score (e.g. 0-1 or 0-10) for graded evaluation
docs.langchain.com
, but a binary or categorical output (correct, partially correct, incorrect) might suffice. If no expected_answer is given, we might skip this or rely on faithfulness as a proxy (since if the agent uses correct sources faithfully, it’s likely correct).
We will likely store a boolean or graded score for agent correctness and baseline correctness. For baseline, if expected_answer exists, do the same comparison. If the baseline or agent said “I don’t know” for an answerable question, that’s incorrect (though arguably better than a wrong answer – we might distinguish “no answer” vs wrong answer in analysis).
For multi-fact answers, we might also compute a recall/precision against expected facts list (if we structured expected_answer as a set of facts). In such a case, we could count how many expected facts were present in the agent’s answer (this contributes to completeness), and how many extra incorrect facts were added (affecting precision). This is similar to computing an F1-score on the set of correct facts.
Faithfulness (Citation Precision): Using the agent’s answer and agent_citations (which contain the actual text of sources or at least IDs we can fetch text for), we’ll implement the earlier described algorithm:
Split agent_answer into statements (likely by sentence or semicolon). For each statement, gather all citation indices that the agent attached to it (if the answer is written with [1], [2] references, we know which statement each reference belongs to from the context string). In the agent’s output format, the answer might include numbers like “[1]” which correspond to the citations list. We can map statements to citations by position.
For each statement, check the corresponding source text (we might retrieve the full text of the Telegram message via fetch_docs if not already present) to see if the statement’s content appears or is logically implied. We might do a substring match for names/dates or use a language model to check. Another method: use the verify tool itself in a meta way – for each statement, call the verify tool with claim=statement (though that might be expensive and possibly redundant since agent’s verify was on the whole answer). Instead, a small entailment model or embedding similarity might do. As cited from RAGAS, an exact method is: “if all claims can be inferred from context, faithfulness = 1”
docs.ragas.io
. We can also use the HHEM model as they suggest
docs.ragas.io
: feed the statement and context to it to get a classification of supported/not-supported.
Sum up supported statements and divide by total statements
docs.ragas.io
 to get a precision score. We will likely convert that to a percentage. For evaluation reporting, we might simply mark each answer as “faithful” or not based on whether this score is 100% or below. But a granular score is useful to identify partial hallucinations. We’ll capture this for the agent. For baseline (which has no citations), we can attempt a similar approach by treating the entire baseline answer as a claim and searching the knowledge base for each statement. Essentially, run a mini retrieval for each baseline statement and see if the Telegram data contains it. If not, that statement is unsupported. This approximates baseline’s faithfulness.
Additionally, we will compare the agent’s citations against the expected_documents list to compute Citation Accuracy: are the sources the agent cited actually relevant/correct? Ideally, the agent should cite documents that are in the expected_documents (the ground truth sources). If the agent cited something outside that set, it might have used a less relevant source or even a wrong one. We can measure citation precision as (number of cited docs that are relevant / total cited docs). For example, agent cited 3 messages, but only 2 of those were in the expected set → citation precision 66%. This differs slightly from faithfulness-by-claims (because it’s about the documents themselves, not the claim content), but it’s another view of citation quality
docs.aws.amazon.com
. We will include this as well, as it reveals if the agent brings in unnecessary sources. A high score means the agent’s references are all pertinent.
Coverage & Recall metrics:
Retrieval Recall@5: Using the expected_documents for the query, check if each expected doc ID appears in the top-5 results from the agent’s search. We will have to get the agent’s search hits. Since the agent uses a hybrid search with RRF, the SSE observation for the search action might list hits (with their IDs and scores). We will parse out the first 5 IDs from that observation. If the SSE doesn’t give full info, another approach is to directly call the search tool in evaluation (bypassing the agent) to replicate the agent’s search results. Because the agent’s search is deterministic given the same query and filters, we could call the same retriever (via an API or Python call if available) with route and filters the agent decided. But the simpler way is likely capturing the agent’s first search event. Assuming we have the list: we then compare it with the expected_documents set. We compute recall@5 = (count of expected_docs present in top5) / (count of expected_docs). Additionally, we can mark a binary success if at least one expected doc is in top5 (often used definition of recall@5 for single-answer questions). This will be done for the agent (and potentially for baseline if baseline had a retrieval phase – but if not accessible, we skip baseline for this metric). We’ll aggregate the recall@5 across queries to see the retriever’s effectiveness.
Citation Coverage (Document level): From the agent’s citations, compute what fraction of expected_documents were cited. Essentially this is recall on the evidence usage. For example, expected_documents = [A, B, C], agent_cited = [A, C] → doc coverage = 2/3. If the query required multi-doc reasoning, we want this high. If the query only needed one doc and agent cited it, that’s 1/1. We will measure this for each query (if expected_docs defined). Low coverage indicates the agent didn’t use some relevant source (possibly it missed it in retrieval or chose not to include due to token limit or an error in compose_context).
Information Coverage (Answer completeness): If we have a list of expected key points in expected_answer or notes, we can attempt to mark each as present/absent in the agent’s answer. This is similar to how we verify claims, but now checking the answer contains all needed facts. For instance, expected_answer might list two distinct announcements; we check the agent’s answer and find it only mentioned one – that’s incomplete. We might quantify this as a recall of facts (e.g. 1/2 = 50% completeness). This likely requires some manual labeling of what the key facts are, so it might be semi-automatic (the LLM judge for correctness can inherently judge completeness qualitatively
docs.aws.amazon.com
). We will mainly rely on the LLM’s assessment or manual judgment for nuanced completeness, and use this metric as supportive evidence.
Agent’s internal coverage: Note that the agent already reports citation_coverage (docs cited / docs fetched) in each compose_context. We will collect the final coverage value (after any refinements) from final.coverage. This is an internal measure (e.g. agent might retrieve 5 docs and only cite 4, giving 80% coverage). While not as “gold-standard” as the above metrics, it’s informative to see how much of what agent fetched was used. If internal coverage is low, it implies either many irrelevant docs were fetched or some relevant ones got dropped. We can include average internal coverage in the report. Ideally, internal coverage should be high due to the refinement logic (agent tries to achieve ≥80%
docs.aws.amazon.com
). We will verify that indeed for most queries it hits that threshold, and note those where it didn’t (they likely triggered a refinement).
Conciseness: We will compute a simple length-based metric for conciseness:
Count the number of tokens or characters in the agent’s final answer. We might also count sentences.
Compare this with either a predefined ideal length or the baseline’s answer length. For example, if baseline answered in 2 sentences but the agent took 6 sentences, and those extra sentences don’t add new info, that might be a conciseness issue. We can define a heuristic such as: If agent_answer_length > baseline_answer_length * 1.5 (150%) and correctness is similar, then agent is less concise. Or if agent’s answer exceeds, say, 1000 characters (arbitrary threshold) it’s probably too long for a QA answer.
We will also detect if the agent included extraneous info. One automatic way: see if the agent’s answer includes content that isn’t directly asked for (we might parse the question for keywords and see if the answer introduces a completely unrelated topic). This is hard to generalize, but if our faithfulness check is high (all content is from sources) yet the answer is long, it could still be on-topic but maybe overly detailed. This might require a qualitative judgement. Possibly the LLM judge can be extended to rate “conciseness” or we can simply note the length.
For reporting, we might give an average answer length for agent vs baseline and flag the outlier cases. E.g. “Agent answers average 3 sentences (~50 words), baseline average 2 sentences (~30 words).” If conciseness is a concern, we can tighten prompts or logic. But at least we’ll identify any instances where the agent rambles.
Latency and Performance: We calculate:
Agent latency per query (recorded earlier) and similarly baseline latency.
Then compute aggregate stats: mean, median, and 90th or 95th percentile of agent latency. Compare with baseline’s. We’ll also note the distribution of agent steps (how many reasoning steps on average).
We’ll check how many queries triggered refinement_count = 1 (or more, if in future allowed) – ideally this should be a small fraction if most answers had sufficient coverage initially. If many refinements occur, it may indicate queries where initial search missed info.
Possibly measure throughput: queries per minute the agent can handle vs baseline (though since we do one at a time in eval, we may not explicitly measure concurrency, but we can extrapolate if needed).
All these help ensure the agent meets the performance criteria. If any queries took exceptionally long (maybe one with max steps), we identify them in the report (with their step count and which tool took time).
Result Aggregation and Output: After computing metrics for all queries, the tool will compile a comprehensive evaluation report. The output format will be designed for clarity:
We will likely produce a tabular summary (CSV or Markdown table) where each row is a query and columns include: Query ID, Category, Agent Correct?, Baseline Correct?, Agent Faithfulness (%), Agent Coverage (%), Agent Answer Length (chars), Agent Latency (s), Baseline Latency (s), Comments. For example:
Query	Category	Agent Correct?	Baseline Correct?	Faithfulness	Coverage	Agent vs Base Latency (s)	Notes
@durov announcements Aug 2023	temporal	Yes	Partial (missed 1)	100%
docs.ragas.io
3/3 docs (100%)	8.5 vs 3.2	Agent cited all sources, baseline missed one announcement.
Channel @news crypto discussion	channel	Yes	No (hallucinated)	90%	2/2 docs (100%)	10.1 vs 2.8	Baseline gave incorrect info; agent correct with sources.
Telegram new features last month	temporal	Mostly (missed minor detail)	Mostly	100%	4/5 docs (80%)	12.0 vs 5.0	Agent missed 1 feature (refinement limit), baseline had similar miss.
Unanswerable query example	negative	Yes (said no info)	Yes (no answer)	N/A	N/A	3.0 vs 2.5	Both correctly indicated lack of info.
Simple fact question X	simple	Yes	Yes	100%	1/1 doc (100%)	5.0 vs 2.0	Both correct; agent took longer.
(Citations like
docs.ragas.io
 in the table above would not actually be in the final report text – they are just here in this explanation to indicate a reference. In the real output, we might not include reference codes like that; instead, we would include them in footnotes or separate details.) This table gives a high-level view for stakeholders to scan. It highlights where the agent outperformed (e.g. second query: baseline hallucinated, agent correct) and where the agent underperformed or was slow. The “Notes” can briefly explain any interesting points or reasons for failure as determined by our analysis.
We will also produce aggregated metrics:
e.g. “Agent accuracy: 75% of queries answered correctly (6/8 correct fully, 2 partially correct) vs Baseline accuracy: 50% (4/8).”
“Average citation precision (faithfulness): 0.95 (i.e. 95% of answer statements supported by sources). Citation coverage: agent cited on average 90% of relevant docs. Recall@5: 100% (all expected docs were found in top-5 for all queries) – indicating strong retrieval.”
“Latency: Agent avg 8 sec (p95 12 sec) vs Baseline avg 3 sec. Agent used 1 refinement in 2/8 queries.”
These can be listed as bullet points or a short paragraph in the report. We will clearly state if the agent met the success criteria (≥70% correct answers, etc.) and where it did not.
Detailed per-question report: For deeper analysis, we will generate a section or separate file with each query’s details:
Question, agent answer, baseline answer.
The agent’s cited Telegram messages (we could include the actual excerpt of each citation for context).
The metrics for that query: whether correct, which expected docs were found or missed, any unsupported statements.
If something went wrong, a brief diagnosis: e.g. “Agent failed because it missed a relevant message in search” or “Agent’s answer included a detail not in sources (hallucination about X)”.
This serves as a diagnostic log for developers to inspect specific cases. For example: Query 2: “Что обсуждали в канале @news о криптовалюте?”
Agent: “В канале @news недавно обсуждали криптовалюту – [Bitcoin] рост цены и [новые регуляции] в Европе【1】. Также упоминались перспективы [DeFi] проектов【2】.”
Citations: [1] (NewsChannel, 2025-10-01: post about Bitcoin price), [2] (NewsChannel, 2025-09-20: post about DeFi regulations).
Evaluation: Correct – agent summarized two relevant discussions with sources. Faithfulness 100% (both claims supported by [1] and [2]). Coverage 100% (expected both posts and both cited). Baseline: “Они обсуждали недавний рост биткойна.” (only mentioned one aspect, no source). Baseline missed the second part (partial answer).
Query 3: “Какие изменения произошли в политике канала Z?”
Agent: … etc. (This level of detail would be provided for each query.)
This part can be in textual form or as a structured JSON for further analysis. It’s mainly for internal debugging and understanding failure modes.
Analysis and Decision Support: Finally, the evaluation report will include a summary analysis and recommendations:
Highlight the agent’s strengths: e.g. “Agent answered 2 queries that baseline could not, thanks to multi-step search and proper use of channel filters.”
Highlight any regressions: e.g. “On simple fact queries, agent was correct in all cases, matching baseline, though with slightly longer answers.” Perhaps “Agent tended to add more context than baseline, which is relevant but maybe not needed (conciseness could be improved).”
Note patterns: “All temporal queries were handled correctly with proper date filtering by agent. The only failures were in a multi-hop query where the agent missed one document due to the single refinement limit.”
Go/No-Go Criteria Check: We explicitly check if the success criteria are met:
Accuracy: say agent got 6/8 = 75% correct (above 70% threshold – criterion met)
patronus.ai
.
Citation precision: suppose average ~95%, with all answers having full support (criterion: ideally 100%, but 95% might be acceptable if the unsupported piece was minor).
Coverage: maybe agent achieved full coverage on 80% of queries (exactly meeting the 80% criterion).
Baseline comparison: agent outperformed baseline on all “complex” queries (baseline only managed 50% of them) – criterion met.
Latency: if agent’s p95 was 12s (well under 30s) – criterion met.
We will list these and give a go recommendation if all key criteria are satisfied, or no-go if there are serious issues (with suggestions on what to fix).
This analysis section is part of the output to guide stakeholders in understanding the results and deciding on deployment.
Technical Implementation Notes: The evaluation tool will likely be a Python script or notebook. It could also be built as a small FastAPI service if we wanted a web UI for uploading a dataset and getting a report, but initially a script is simpler. It will integrate with the existing system by calling the public API endpoints of the agent and baseline (ensuring we use proper auth if needed). We’ll reuse the agent’s capabilities (like document fetching or even the verify tool if it helps in evaluation). For example, we could use the agent’s verify tool to double-check baseline answers by feeding the baseline answer as a claim to verify against the index – this gives a confidence of how supported it is, similar to how the agent verifies itself. This is a clever reuse to automatically flag baseline hallucinations. We will make sure the code is modular: separate functions or classes for loading dataset, running agent, running baseline, computing each metric, and generating the report. This modularity allows extending or tweaking metrics easily. For instance, if later we want to add a new metric (say, Logical Coherence or Harmlessness), we can add a function and integrate it with minimal changes. The architecture also considers error handling: if the agent fails on a query (e.g. returns an error or falls back due to max_steps), we capture that and mark the result appropriately (agent_answer might be “ERROR” or fallback flag true). Those will be counted as incorrect for accuracy. Similarly, if baseline returns nothing or an error, we handle that. The script will log these occurrences.
Детальная спецификация метрик
Let’s detail each metric with definitions, calculation methods, and examples:
Answer Accuracy (Correctness & Completeness): Definition: The degree to which the agent’s answer is factually correct and answers all aspects of the question. We consider an answer “accurate” if it contains the right information with no significant errors or omissions.
Measurement:
– If a ground truth answer or list of expected facts is provided, we treat this as the gold standard. We can calculate a Precision, Recall, F1 on content: e.g. Precision = (# of correct facts in agent answer / total facts stated in agent answer), Recall = (# of correct facts in agent answer / total facts in ground truth). A simple exact match is too strict for long answers, so instead we often use an LLM to compare. We will prompt an LLM with the question, agent answer, and reference answer to get a judgment. Alternatively, use a rubric as in the Databricks example, where correctness is scored 0-3
databricks.com
databricks.com
. For simplicity, we might map to: Correct, Partially Correct, or Incorrect. Partially correct means some aspects answered but something missing or slightly wrong.
– If no reference answer is available, accuracy is harder to measure automatically. In this case, we lean on faithfulness and recall as proxies: if the agent cited the right sources and didn’t hallucinate, and if it had high coverage of those sources, likely the answer is accurate. But to be safe, we might use an LLM judge that has been given the context of the top retrieved documents: essentially ask “Given these source documents, is the answer correct and answering the question?”. This blends correctness with faithfulness in a single check
docs.langchain.com
.
Example: For question “What new features did Telegram introduce last month?”, suppose the expected key points are: {“Usernames without SIM”, “Ability to convert video messages to text”}. If the agent’s answer mentions both, it’s complete. If it only mentions one, it’s partially correct (missing one feature). If it mentions something irrelevant or wrong, it’s incorrect. We’d reflect that in a score or label. An LLM might evaluate: “The answer covers one feature correctly but misses the second – so it's incomplete.”
Automation: Using LLM-as-judge for correctness is a known approach to avoid purely manual grading
docs.langchain.com
. We will likely use GPT-4 (if available) or another high-quality model. If that’s not feasible offline, an alternative is embedding similarity: embed agent answer and expected answer and compute cosine similarity – not very precise, but can signal if they share content. Another rule-based check: look for key phrases from expected answer in the agent answer. For instance, does the agent answer contain “SIM” or “video message” in the above example? If one is missing, that indicates missing content. Such keyword overlap can serve as a partial automated check for completeness.
Faithfulness (Citation Precision / Hallucination Rate): Definition: The proportion of the answer’s content that is supported by the cited source documents. In other words, it measures how factual and grounded the answer is. A perfectly faithful answer has no hallucinated details – every claim can be verified in the provided citations
docs.ragas.io
.
Formula: We use the formula from RAGAS’s definition of faithfulness
docs.ragas.io
:
Citation Precision (Faithfulness)
=
#
 of claims in answer supported by sources
#
 of total claims in answer
.
Citation Precision (Faithfulness)= 
# of total claims in answer
# of claims in answer supported by sources
​
 .
This yields a value between 0 and 1 (or 0–100%). We can also express the complement as a Hallucination rate = 1 - precision (fraction of claims not supported).
Method: We will implement a claim check as described. For each answer, identify the sentences or clauses that assert a fact. For each claim:
Find if any cited document contains that fact. This can be done by a direct string search for unique terms (names, numbers) or using a semantic check. For semantic check, we can use an entailment model or an LLM: “Source: [excerpt]. Claim: [sentence]. Does the source support the claim? yes/no.” If yes for at least one source, the claim is considered supported. If the agent provided multiple citations per claim, we check each until one supports it (only need one good source per claim).
Count supported vs total.
Example: Agent answer: “Pavel Durov announced the launch of Stories in August 2023 [1]. He also mentioned an upcoming username auction platform [2].” Suppose citation [1] is a Telegram post where Durov indeed announced Stories in August 2023, and [2] is a post talking about usernames. If both statements align with their sources, then faithfulness = 2/2 = 100%. If the second statement was not actually in the cited post [2] (maybe the post was about something else), then faithfulness = 1/2 = 50% – meaning half the answer is unsupported (low precision).
Automation: We can automate with high confidence using a combination of string matching and an NLI model (natural language inference). There is mention of an open classifier (HHEM) that flags hallucinated content by checking answer vs context
docs.ragas.io
 – we can integrate that for a robust approach. If that model says a claim is a hallucination, we mark it unsupported. Alternatively, use the agent’s verify tool with the claim as a query; if verify finds a doc with high confidence, that implies support, if not, likely unsupported. We will try to minimize manual work here – ideally this is fully automated. We’ll double-check a few by hand to ensure the automation is working (especially if the answer is paraphrased relative to source – the model should catch that as supported).
Additionally, we will measure Citation correctness: fraction of cited sources that were truly needed. If agent cites an irrelevant message, that would indicate an issue. But given the agent’s design, usually it cites what it used. We have ground truth doc list; any cited doc not in that list might be extraneous or even wrong. That metric (relevant citations / total citations) is analogous to precision in information retrieval terms
docs.aws.amazon.com
. We want it as close to 1.0 as possible – meaning the agent isn’t citing unrelated stuff.
Information Coverage (Recall) Metrics:
Document Retrieval Recall@5: Definition: The proportion of relevant documents that appear in the top-K retrieval results (K=5 in our case). It indicates the retriever’s ability to find the necessary information quickly. A recall@5 of 1 (100%) means all needed docs were found within the first 5 hits. If recall@5 is low, the agent might require more steps (or miss out entirely).
Calculation: 
Recall@5
=
∣
{
expected docs
}
∩
{
top 5 retrieved docs
}
∣
∣
{
expected docs
}
∣
.
Recall@5= 
∣{expected docs}∣
∣{expected docs}∩{top 5 retrieved docs}∣
​
 . For single-answer queries where only one doc is expected, this is essentially 1 if the doc is in top5, 0 if not. We will compute this for each query using the expected_documents list and the captured top5 IDs. Then we can average across all queries to get an overall recall@5. If multiple docs are expected, the formula above applies (e.g. expected 3, found 2 in top5 → recall@5 = 0.67).
Example: If a query needs docs [A, B] and the agent’s search results (rank 1-5) IDs are [X, A, Y, Z, B], then recall@5 = 2/2 = 100% (both A and B were found, albeit not rank1). If only A was found, recall@5 = 1/2 = 50%.
Note: Because our agent uses a hybrid search with BM25 + dense, and we give it the query as is, we expect it often finds relevant stuff if present. If we see recall@5 less than 100%, that’s a sign to improve the index or query planner. This metric requires the ground truth doc IDs – we will ensure those are correct via manual curation.
Citation Coverage (Evidence Recall): Definition: The fraction of ground-truth relevant documents that were actually cited in the final answer. This reflects if the agent incorporated all the needed sources when formulating the answer – a measure of answer’s completeness regarding source usage
docs.aws.amazon.com
.
Calculation: 
Citation Coverage
=
∣
{
expected docs
}
∩
{
cited docs
}
∣
∣
{
expected docs
}
∣
.
Citation Coverage= 
∣{expected docs}∣
∣{expected docs}∩{cited docs}∣
​
 . If expected_docs = [] (no known relevant doc, as in unanswerable or trivial cases), we define coverage as N/A. We can aggregate similarly to see on average what percent of relevant sources are being used.
Example: expected docs [A, B, C], agent cited [A, C] => coverage = 2/3 ≈ 66%. Ideally this is 100% for answerable questions (the agent used all the key evidence). A less than full coverage could mean the agent answered with partial info (e.g. ignored B). In context, maybe B had some detail agent omitted. That might correlate with an incomplete answer.
Answer Completeness (Fact Recall): Definition: Fraction of expected key facts that appear in the answer. This is similar to citation coverage but at the information level rather than document level. We will use this mostly qualitatively. If we have an enumerated expected answer (e.g. “the user should mention features X, Y, Z”), we can see how many of X, Y, Z appear in the agent’s answer.
Calculation: 
Answer completeness
=
# of expected facts present in answer
# of expected facts total
.
Answer completeness= 
# of expected facts total
# of expected facts present in answer
​
 . This requires us to define the set of expected facts, which might come from the reference answer or manual annotation. It might not be automated unless we code specific keyword checks.
Example: If expected facts = {Bitcoin price increase, new crypto regulation, DeFi projects} and agent answer mentions two of those, completeness = 2/3.
In practice, the LLM judge we use for correctness can inherently cover completeness, so this metric may not need separate calculation; it’s more of an interpretation aid for us. But if needed, we can do a quick keyword matching for critical concepts.
Conciseness: Definition: Measures whether the answer is presented in a brief, focused manner without unnecessary content. A concise answer is typically short and directly answers the question, whereas a non-concise answer might include irrelevant details, overly long explanations, or repeated information.
Possible Metrics: There isn’t a single formal formula for conciseness, but we can define a few indicators:
Answer Length: The simplest indicator. We will record the length of each answer (in characters or tokens). We can set a guideline that, for our domain, an ideal answer is e.g. 1-3 sentences (maybe 20-100 words depending on question). If an answer is, say, 300 words, that’s likely too long. We can derive a score inversely proportional to length beyond a threshold. For instance, Score 1.0 if <=100 words, then linearly decay to 0 by 300 words. This is somewhat arbitrary, so we might not formalize a score but simply flag long answers.
Redundancy Check: We can check if the answer has repeated phrases or sentences (which would indicate it’s rambling). For example, if the agent’s answer restates the same point twice, or contains filler like “It is noteworthy that… (repeating)”, we can detect duplicate n-grams or excessive filler phrases.
Irrelevant content: Hard to measure automatically, but if our faithfulness check finds a statement supported by a source that isn’t directly relevant to the question, that might be extraneous. Alternatively, an LLM can be asked: “Did the answer include information not asked by the question?” and give a yes/no.
Approach: We will likely combine the above into a qualitative judgment. As part of the LLM judge prompt, we can include a criterion for brevity (some evaluations do this by asking for a rating on “readability/conciseness”). For example, instruct the judge: “Rate if the answer is concise (no irrelevant info, answered in as few words as needed).” This could yield a score or just a comment. In automation, we might not fully trust an LLM to score conciseness, but we can use it to catch glaring issues. The main automated metric will be answer length vs expected.
Example: For a question that expects a one-sentence answer, if agent outputs a paragraph with background info, it’s not concise. Suppose baseline answered in 10 words and agent in 50 words, and those extra 40 words were just context or quotes that the user didn’t ask for – that’s a conciseness problem. We’d note that as such. On the other hand, if the agent answer is longer because it lists multiple items (which were actually asked), that’s fine (complete but still concise relative to requirement). The trick is distinguishing useful detail from fluff. We will rely on human interpretation for final judgment here, but automated flags (like length outlier) will point us where to look.
Latency & Throughput: Definition: Time taken by the system to produce the answer, measured in seconds. We consider both average latency and tail latency (p90/p95). Also, related metrics like the number of reasoning steps or tool calls, which affect latency.
Measurement: Straightforward – we measure real clock time for each query for both agent and baseline:
Latency (seconds)
=
t
response_end
−
t
request_start
.
Latency (seconds)=t 
response_end
​
 −t 
request_start
​
 .
We’ll do this on the client side to include network overhead (since in production that matters). After all queries, compute mean, median, and percentile stats for agent vs baseline.
Additionally:
Step Count: We can parse the final event’s total_steps for the agent (or count step_started events). This tells us how many Thought/Action loops it went through.
Refinements Count: Already recorded per query (0 or 1 typically). We can sum how many queries required refinement and the average refinements used.
Tool usage patterns: We might log how many times each tool was invoked on average (e.g. search was always used, maybe rerank used X% of time, verify used Y%). This is more of a diagnostic performance metric – if verify is used every time and maybe adds 2s, it’s expected overhead.
Targets: We have threshold of 10s typical, 30s worst-case from requirements. So in analysis, we will check what fraction of queries exceeded 10s, and if any exceeded 30s.
Example: If agent latencies: [5s, 8s, 12s, 9s, 7s, 11s, 10s, 6s] for 8 queries, average ~8.5s, p95 ~12s. Baseline latencies: [2s, 3s, 4s, 2.5s, 3s, 3.5s, 4s, 2.2s], average ~3s. We will note this difference. The example shows agent is slower but within acceptable range (no timeouts). If one query had say 25s (maybe hitting a slow tool or borderline case), we’d highlight that as needing optimization.
There’s no complex formula here, just direct measurement and summary.
Additional Qualitative Metrics: (if relevant, though not explicitly asked, we mention for completeness)
Context Relevance: How relevant were the retrieved documents to the query? (We can measure this via average retrieval score or an LLM judge rating the context quality
docs.langchain.com
). A high context relevance means the search results made sense. If our recall is good, relevance likely is too. We might not separately quantify this unless a case arises where agent retrieved a lot of irrelevant stuff (which would show as low internal coverage).
Logical Coherence: Are the agent’s answers logically structured and not contradictory? Probably fine given it’s one paragraph answers; we won’t measure unless obvious issues.
Comparison vs Baseline metrics: We might include a metric of Relative Gain: e.g. how many queries agent got right that baseline got wrong (which is basically what our comparison will list). This isn’t a single number metric but we can count: agent answered +X more questions correctly than baseline. Or Error overlap: out of the ones agent got wrong, how many did baseline also get wrong (if baseline got some right that agent missed, that’s regression count). These help identify if agent introduces any new errors.
In summary, each metric defined will be calculated for each query (where applicable) and then aggregated. We will produce a scorecard of these metrics:
Answer Accuracy (perhaps as % of queries correct, or average correctness score),
Faithfulness (average citation precision %),
Coverage (average citation coverage %, and overall recall@5 %),
Conciseness (maybe average answer length and any flags),
Latency (average and p95 in seconds),
plus a comparison of baseline vs agent on key metrics like accuracy.
We will preserve citation references in our evaluation report whenever we quote or refer to specific findings in research. For instance, if we mention how we measure faithfulness, we’ll cite the formula source
docs.ragas.io
, or if we refer to AWS’s recommendation of using both precision and coverage for citations, we cite that
docs.aws.amazon.com
. These citations lend credibility and traceability to the methodology choices.
План реализации
To implement this evaluation system, we propose the following steps and priorities:
Define Evaluation Queries and Ground Truth (Week 1):
– Expand the dataset beyond the initial 8 questions to cover diverse scenarios (temporal filters, channel-specific queries, author-specific, multi-hop, simple factual, unanswerable, etc.). Aim for perhaps 20-30 queries for a more robust evaluation (still manageable manually).
– For each query, manually gather the expected documents (search in Telegram data or from domain knowledge) and formulate reference answers or key points. This step requires domain understanding but is critical to have reliable evaluation data.
– Mark each query with metadata (category, answerable or not).
– Outcome: eval_dataset.json ready with all necessary fields filled. This is the “gold standard” our metrics will use.
Set Up the Evaluation Script Environment (Week 1):
– Ensure we have access to the agent’s streaming API and baseline API in a test environment. Prepare authentication (if needed) and any config (like base URLs).
– If using any external LLMs (OpenAI GPT-4) for judgment, obtain API access and ensure we respect rate limits/cost. Alternatively, set up a local LLM or the Qwen model itself for judging (though Qwen might not be as reliable for self-evaluation). We might try a smaller open model for initial development and reserve GPT-4 for final runs for quality.
– Install or prepare any libraries: e.g. SSE client, requests, numpy/pandas for data handling, possibly ragas or similar if we leverage it (though we may just implement our own due to customization needs). Also, if using the Vectara HHEM model, get that ready (could be a HuggingFace model to load).
Implement Core Evaluation Logic (Week 2):
– Dataset loader: Code to read the dataset JSON and iterate through entries.
– API caller for agent: Using SSE, handle streaming. (We might create a helper that collects events and returns the final event plus any intermediate we want. Or simpler, since agent’s final answer only comes at the end, we might ignore intermediate events except search results. But capturing search results is needed for recall@5. We can either parse the observation text for search or possibly modify agent API to return hits – but we’ll likely just parse the SSE text.)
– API caller for baseline: straightforward HTTP request and response handling.
– Metric calculations: Implement functions for each metric:
evaluate_correctness(agent_answer, expected_answer, context_docs) – if expected_answer given, possibly call LLM or compute similarity. Could also incorporate context_docs (the ground truth docs) if we want to check completeness with them. The function could return a tuple (correct: bool, score: float, notes: str).
evaluate_faithfulness(agent_answer, citations) – implement claim splitting and support check. This is a substantial piece: might incorporate an entailment model. For now, perhaps implement a simple version that checks if all content words in answer appear in the citations text (as a rough proxy), then later integrate a proper model. Also returns (precision_score, unsupported_claims_list).
evaluate_recall(search_hits, expected_docs) – returns recall@5.
evaluate_coverage(cited_docs, expected_docs) – returns fraction of expected docs cited.
evaluate_conciseness(agent_answer, baseline_answer, question) – measure lengths, maybe run a brief check for redundancy. Could return a qualitative label or a score. Possibly something like: if len(agent_answer) <= len(baseline_answer)*1.2 -> “Concise”, else if <= *1.5 -> “Acceptable”, else “Too long”. This is heuristic.
These functions use data from the agent’s results and dataset to compute values.
– Storing results: For each query, create a result entry with all raw and computed fields (like earlier pseudocode, but now also fill metrics). We might define a Result data class for clarity.
Automate LLM Judging (Week 2-3):
– Integrate GPT-4 or another model for correctness and possibly conciseness evaluation. This involves constructing prompts. We should also do a few-shot prompting if needed to calibrate the judge (maybe provide one example of a correct vs incorrect answer in the prompt). However, given a small eval set, zero-shot with careful instruction might suffice.
– Ensure this step is toggleable (maybe make LLM-judging optional via a flag) in case we want to run the evaluation without external calls at times. We’ll likely first run everything with automated metrics and then add LLM eval for final scoring.
Run Evaluation and Compute Metrics (Week 3):
– Execute the script on the full dataset. Catch any errors (e.g. if agent fails on a query due to unforeseen input issues). We might adjust the agent’s max_steps or other settings if needed for test (the default 8 steps should be fine given we don’t expect extremely complex dialogues).
– Gather all results. Compute aggregate metrics easily via Python (e.g. use pandas to average).
– Generate the output tables and charts. We can use Markdown for tables and maybe matplotlib or similar for any plot (like a bar chart of accuracy or a latency distribution histogram). If this is to be a static report, including a few simple charts for latency or accuracy comparison might be nice. However, since the final output may just be text/Markdown, we’ll primarily ensure the tables are clear. If needed, we could embed an image of a chart (but the instructions said no need to search for images; we can programmatically generate a chart of metrics distribution if it adds value – perhaps not essential here).
– Ensure that all citations to sources (like the ones we used from AWS, RAGAS, etc.) are included appropriately in the textual report (so that our evaluation methodology references are there for the reader’s benefit).
Analyze Results and Draft Findings (Week 3-4):
– Interpret the numeric results. Write the summary of findings (this will mirror some content in the executive summary and analysis above, but now with actual numbers from our run). E.g. “Agent answered 18/25 questions correctly (72% accuracy), baseline 12/25 (48%). The agent’s mistakes were mostly partial answers missing one detail, whereas baseline’s mistakes included 5 hallucinations without sources. Citation precision for agent was ~98%, meaning almost every claim was source-supported
docs.ragas.io
. One answer had a minor unsupported guess (noted in report). The retriever recall@5 was 92% – in two cases, a relevant message was ranked lower, which led to a missed detail. Latency: agent average 9s vs baseline 3s; all agent responses under 20s. …” and so on.
– Formulate recommendations based on this: e.g. if conciseness was an issue, suggest adjusting the prompt to be more brief. If a certain category was problematic (say, multi-hop requiring >1 refinement, which we limited), suggest allowing more refinement or improving query planning. These recommendations tie the evaluation back to actionable improvements.
Iterate and Finalize (Week 4):
– If some metrics are not meeting targets, consider a quick fix and re-run on affected queries (for example, if a particular query failed due to a bug, maybe adjust and test again). However, since this is evaluation, we might not change the agent here, just note it. If we do fix agent, we would re-run evaluation to see impact.
– Finalize the evaluation report document with all sections (executive summary, methodology, results, etc. as structured above). Make sure it’s comprehensive and clear for stakeholders.
– Package any evaluation code for reproducibility (so it can be run as part of CI or on new model versions easily). Possibly write a README for how to run it.
Prioritization (MVP vs Later Enhancements):
For the MVP (Minimum Viable Product) of the evaluation tool, the critical metrics to implement first are:
Answer accuracy determination (even if just via a rough method or minimal LLM usage).
Faithfulness (no hallucination) check, because ensuring citations support answers is crucial for trust.
Retrieval recall@5, since that directly affects whether the agent can find answers at all.
Latency measurement, to ensure performance is within limits.
Comparison with baseline on correctness (to prove the value-add of the agent).
Conciseness and some of the finer-grained metrics (like context precision or answer completeness scoring) can be added once the above are in place. They are slightly lower priority because an agent that is correct, faithful, and finds info is already in a good state – conciseness, etc., are more polish (unless extreme verbosity is observed). Dependencies:
Access to the running agent and baseline services (ensure the environment is similar to production for realistic results).
Access to a powerful LLM for evaluation (not strictly required if we avoid it, but recommended for accuracy assessment – possibly we can use our own Qwen or another open model with a judge prompt if API use is restricted).
The Telegram data index should be the same as used by agent; if we run evaluation in a staging environment, ensure the data is current. If the dataset queries things like “last month”, make sure it aligns with the data’s time frame.
Python libraries for SSE and any ML model (like transformers for HHEM model if we use it). Possibly ragas library if we want to leverage it (it could simplify some calculations if we configure our data into their format, but integrating might take extra time – writing our small functions might be faster given the custom needs).
CI/CD Integration and Monitoring (Future work):
– Once the evaluation suite is stable, integrate it into the CI pipeline. For example, we can add a test stage that runs a subset of eval queries on every code change (to catch major regressions quickly – maybe not all queries to save time, but a representative few). For full evaluation, we might schedule it as a nightly or weekly job that outputs a report to the team.
– Also plan for production monitoring: we can reuse some metrics definitions (like faithfulness) to monitor live traffic. E.g., for each user query, log the verification confidence
docs.aws.amazon.com
 or run a lightweight hallucination check in the background. Over time, aggregate these to see if the agent’s live performance drifts (like if a new data source is added and it starts hallucinating more). The evaluation tool’s code can be adapted into a monitoring script that samples random real queries and evaluates them with the same metrics (though without ground truth, it would rely on LLM judging and the agent’s own signals).
By following this implementation plan, we will create a robust evaluation harness that not only produces a one-time report for go/no-go decision, but can be reused continuously to track the agent’s performance over time and after any updates. The plan emphasizes getting core metrics working first (to have a meaningful MVP result), then layering on enhancements like better automated judging and integration into development workflows.
Примеры кода и данных
To illustrate the evaluation setup, here are some simplified examples of data structures and pseudocode:
Dataset JSON Structure Example:
[
  {
    "id": 1,
    "query": "Какие важные объявления делал @durov в августе 2023?",
    "category": "temporal+author",
    "expected_answer": "В августе 2023 года Павел Дуров объявил о запуске NFT-аукционов для имён пользователей и о введении истории в профилях.",
    "expected_documents": ["telegram-1001", "telegram-1005"],
    "answerable": true
  },
  {
    "id": 2,
    "query": "Что обсуждали в канале @news о криптовалюте?",
    "category": "channel",
    "expected_answer": "В канале @news обсуждали рост курса Bitcoin и новые регулирования криптовалют в Европе.",
    "expected_documents": ["telegram-2008", "telegram-2010"],
    "answerable": true
  },
  {
    "id": 3,
    "query": "Был ли сбой Telegram 1 января 2020?",
    "category": "negative",
    "expected_answer": null,
    "expected_documents": [],
    "answerable": false
  }
]
In the above, telegram-1001 etc. would correspond to actual message IDs in the index. We might include the channel name in metadata too, but that can be fetched from the index if needed.
Pseudocode for Metric Calculation: (assuming we have agent_answer, agent_citations_texts, expected_answer, expected_docs for a query)
def assess_answer(agent_answer, agent_citations_texts, expected_answer, expected_docs):
    metrics = {}
    # 1. Correctness (using expected_answer if available)
    if expected_answer:
        result = llm_judge_correctness(question, expected_answer, agent_answer)
        metrics['correct'] = result['correct']  # True/False
        metrics['correctness_score'] = result.get('score', 1.0 if result['correct'] else 0.0)
    else:
        metrics['correct'] = None  # or use faithfulness as proxy
    # 2. Faithfulness (citation precision)
    claims = split_into_statements(agent_answer)
    supported = 0
    unsupported_claims = []
    for claim in claims:
        if claim.strip() == "": 
            continue
        if is_supported_by_any(claim, agent_citations_texts):
            supported += 1
        else:
            unsupported_claims.append(claim)
    if claims:
        precision = supported / len(claims)
    else:
        precision = 1.0
    metrics['faithfulness'] = precision  # e.g. 0.8
    metrics['unsupported_claims'] = unsupported_claims
    # 3. Citation coverage (recall of expected docs in answer)
    cited_doc_ids = [c['id'] for c in agent_citations]  # assume we stored agent_citations as list of dicts
    if expected_docs:
        found = len(set(expected_docs) & set(cited_doc_ids))
        metrics['citation_coverage'] = found / len(expected_docs)
    else:
        metrics['citation_coverage'] = None
    # 4. Retrieval recall@5
    # Assuming we stored top5_hits in agent_result
    if expected_docs:
        found = len(set(expected_docs) & set(agent_top5_hits))
        metrics['retrieval_recall@5'] = found / len(expected_docs)
    # 5. Conciseness
    agent_len = len(agent_answer.split())
    metrics['answer_length'] = agent_len
    if expected_answer:
        expected_len = len(expected_answer.split())
        metrics['length_ratio_vs_expected'] = agent_len / expected_len if expected_len>0 else None
    if baseline_answer:
        base_len = len(baseline_answer.split())
        metrics['length_ratio_vs_baseline'] = agent_len / base_len if base_len>0 else None
    # We can also add a simple check:
    metrics['concise'] = (agent_len < 100 and (metrics['length_ratio_vs_baseline'] or 1) < 1.5)
    return metrics
The actual implementation might be more complex (especially is_supported_by_any which could use embeddings or model), but this shows the structure.
Prompt Template for LLM Judge (Correctness):
System: "You are a judge for question answering. Evaluate the answer given the question and the expected correct answer."  
User: 
Question: "<user question>"  
Ground Truth Answer: "<expected_answer>"  
Answer to Evaluate: "<agent_answer>"  
Instructions: Determine if the answer to evaluate is correct and complete. Say "Correct" if it fully answers the question with no errors, "Partially Correct" if it has some correct info but is incomplete or slightly wrong, or "Incorrect" if it fails to answer or has wrong info. Provide a brief explanation."
We would feed this to GPT-4 and parse the output. If it says "Correct", we mark correct=True. If "Partially Correct", we could mark correct=False but maybe count it separately (like a half point in accuracy). For simplicity, we might treat partially correct as 0.5 in an accuracy score or just as a category.
Sample Output Report Excerpt (Markdown): Overall Results:
Agent Accuracy: 75% (18/24 queries answered correctly) – exceeds the 70% target. Baseline Accuracy: 50% (12/24). The agent outperforms baseline, especially on filtered and multi-step questions.
Faithfulness (Citation Precision): 96% of the agent’s answer statements are supported by sources on average. No major hallucinations observed – all answers had supporting citations
docs.ragas.io
. Baseline, lacking citations, hallucinated on 4 queries (e.g. gave an answer not found in data).
Coverage: Agent used 90% of relevant documents on average (Citation Coverage). In 2 cases, minor relevant docs were left out due to context limit. Retriever Recall@5 is 92% – in two queries, a relevant message was ranked lower than 5, indicating possible room for retriever improvement.
Conciseness: Agent answers averaged 60 words, vs baseline 40 words. Most agent answers were concise and on-topic, but 2 answers were somewhat lengthy (included extra background). These could be trimmed; however, they were still correct. No extraneous completely irrelevant info was observed.
Latency: Agent mean latency 8.4s, p95 12.3s, max 15s. Baseline mean 3.1s. Agent is slower due to multi-step processing, but within acceptable range (under 30s) for all queries. 2 queries triggered the refinement step (each took ~12-15s).
By Category:
Temporal queries (5 queries): Agent answered 5/5 correctly, applying date filters properly. Baseline got 3/5 (failed when date filtering was needed).
Channel-specific (5 queries): Agent 4/5 correct (one partial miss), Baseline 2/5. Agent consistently cited posts from the specified channel, adhering to the constraint. Baseline often gave generic info not confined to the channel.
Author-specific (3 queries): Agent 3/3 correct, using the author’s messages. Baseline 1/3 (in two cases it didn’t find anything and just gave no answer or a wrong guess).
Complex/multi-hop (e.g. needing multiple pieces, 3 queries): Agent 2/3 correct. One failure was due to not using a second refinement – it cited one message but missed the second. Baseline 0/3 on these (couldn’t handle multi-part reasoning).
Simple factual (6 queries): Agent 5/6 correct, Baseline 6/6 correct. On straightforward questions, baseline matched or slightly outperformed agent in one case (agent made a minor error where baseline was directly correct). This suggests the agent has a small regression on an easy query, possibly due to over-complicating it – worth reviewing agent’s thought process for that case.
Example Case Analysis:
Query: "Какие важные объявления делал @durov в августе 2023?"
Agent: Answered with two announcements (“NFT username auctions” and “Stories feature introduction”) with citations【1】【2】. Correctness: Both are indeed major August 2023 announcements – correct and complete. Faithfulness: 100% (it quoted the announcements exactly from Durov’s posts). Coverage: Used 2/2 expected posts. Baseline: Gave only one announcement and missed the second, with no citation. So baseline was partially correct.
Query: "Что писал @username о проекте Y?" (author-specific)
Agent: Provided a summary of what that user wrote about project Y, citing the user’s message【3】. It correctly filtered by author. Baseline: responded with a generic answer not actually from that user (hallucinated that “they praised project Y’s features”, which wasn’t in data). Agent clearly outperformed here, demonstrating the effectiveness of the agent’s author filter tool.
Query: "Расскажи о новых функциях Telegram за последний месяц"
Agent: Listed three new features introduced in the last month, each with a citation to an official Telegram channel post【4】【5】【6】. The answer was a bit long (3 sentences, ~80 words) because it enumerated multiple features, but it was accurate. Faithfulness: 1 claim about each feature, all supported (100%). Baseline: Only mentioned one feature (usernames without SIM), missing others. Agent’s recall of multiple facts gave it an edge, albeit at cost of length. This showcases agent’s strength in questions requiring aggregation of info.
(The numbers [1], [2] etc. here would correspond to actual source references in the full report, with footnotes or a listing of source titles/IDs, but for brevity we describe them in text.)
These examples and structured outputs show how the evaluation results can be presented in a clear manner, combining quantitative scores with qualitative explanations. They also serve as templates for how to structure the data (JSON input, intermediate metrics calculations, and final report content).
Рекомендации и лучшие практики
Based on our research and the needs of this project, here are recommendations and best practices for conducting and maintaining the evaluation:
Automate as Much as Possible: Leverage tools like LLMs and existing evaluation frameworks to reduce manual effort. Using an LLM as a judge is a powerful technique to automatically evaluate answer quality on complex, open-ended queries
aws.amazon.com
docs.langchain.com
. This avoids needing human reviewers for each run. However, always verify the LLM evaluator on some examples to ensure it aligns with human judgment. Where available, use domain-specific checking tools (e.g. the hallucination detector model for factual consistency) to bolster automation
docs.ragas.io
. The more we can trust automated metrics, the faster we can iterate and the more often we can run evaluations (even on every code change if fully automated).
Maintain Reproducibility: Fix the random seeds and environment for the evaluation runs. The agent has some nondeterminism (LLM generation with temperature). For fair comparison, you might run the agent with a fixed random seed or temperature=0 for evaluation purposes. If that’s not possible, consider running multiple trials and averaging, though that’s costly. At least, document which model version and data snapshot was used, so results are comparable over time. Store the evaluation dataset and perhaps the agent’s outputs for each query (in a results JSON) – this acts as a record to compare with future versions. When the agent is updated, you can re-run on the same dataset and directly diff the outputs to see improvements or regressions.
Gold Standard and Continuous Updates: As Patronus AI suggests, “Establish a gold standard early”
patronus.ai
 and keep evolving it. Our initial eval dataset is small; as the system handles more queries in beta, we should collect real user questions and build them into the eval set (with ground truth answers determined). Continuously enrich the dataset to cover new edge cases or important scenarios. This helps the evaluation stay relevant and prevents overfitting to the initial questions. In addition, update the expected answers if the definition of correctness changes (for example, if policies change on how to handle no-answer cases, reflect that in expected outputs).
Use Multi-faceted Metrics: No single metric gives the full picture. We use a combination to catch different failure modes
patronus.ai
. For instance, a high accuracy score is meaningless if the answers aren’t faithful (the agent could be guessing correctly). Likewise, a perfectly faithful answer might be incomplete. By monitoring precision and recall aspects (both in retrieval and answer content)
docs.aws.amazon.com
, we ensure both quality and coverage. Continue using both citation precision and coverage together for evaluation, as AWS Bedrock recommends, to get a complete view of source usage
docs.aws.amazon.com
. Also consider user-facing metrics: an answer could be correct but if it’s too long (low conciseness) or too slow (high latency), user experience suffers. So maintain those as part of the eval criteria and weight them in decisions.
Minimize Manual Review with Smart Filtering: When presenting results for manual analysis, make it easy to focus on problematic cases:
Sort or highlight queries where agent and baseline differ in outcome (those are key to inspect).
Highlight any query where the agent’s verification confidence was low, or our metrics flagged hallucination. These likely need human attention to decide if the answer was actually wrong.
If using LLM judges, highlight any “Partially correct” judgments; those often require a nuanced human call or might indicate missing info.
Provide the sources and answers side by side for those cases so a human can quickly verify. E.g., list the claim and the source text snippet, so one can see if they match.
This targeted approach means a reviewer might only need to look at, say, 5 out of 25 questions, rather than all 25, saving time.
Ensure Evaluation Reflects Real Use: Tailor the dataset and metrics to what matters in production. For example, if in real usage most questions are about recent news, make sure the eval set has such examples and that we evaluate the handling of date ranges correctly. If the agent will be expected to always provide sources, weight faithfulness heavily (which we do). If certain failure types are absolutely unacceptable (e.g. hallucinating an answer to a sensitive question), you might have a hard fail criterion for any hallucination occurrence. Our go/no-go criteria already incorporate some of this (like requiring all answers to be source-verified ideally). Continuously align metrics with product goals – e.g., if users care more about completeness than brevity, we tolerate a slightly longer answer if it’s thorough, but if brevity is a priority for UX, enforce conciseness more strictly.
Integrate Evaluation into Development Workflow: Use the evaluation tool not just as a one-time gating mechanism but as a development aid:
After any significant change to the agent (model update, prompt change, tool improvement), run the eval and see the impact on metrics. This guards against unintended regressions in one area while improving another.
Possibly incorporate a lightweight version of the eval in unit tests (for example, have a few crucial queries that must always remain correct – like regression tests).
Set up alerts if evaluation metrics drop below thresholds (in CI or nightly builds). For instance, if accuracy falls below 70% or latency spikes, flag it before the change is merged or deployed.
Use evaluation results to do error analysis with the dev team. The structured logs showing where the agent failed (and why, if we have diagnostic info) are extremely useful to guide debugging. Maybe the query planner misparsed a query – we see that in the thought process. Or the reranker chose a suboptimal doc – perhaps we then tweak reranker settings. Essentially, evaluation should feed a loop of continuous improvement.
Production Monitoring Using Evaluation Metrics: Once in production, it’s important to keep an eye on real performance:
We can log the agent’s actions and final outcomes for real queries (with user permission/anonymization). We can sample these logs and run a similar evaluation routine (without ground truth, we rely on verification confidence and maybe occasional human labeling). This could yield ongoing metrics like “estimated correctness” and “verification confidence distribution” in the wild. If we see verification confidence dropping over time or certain question types causing low confidence often, that signals a drift or new type of query that needs attention.
Consider implementing a feedback loop: if users can rate answers or if we can detect when the agent says “I’m not sure”, feed those back into an evaluation dataset.
The evaluation tool could be run periodically on a rolling set of recent real queries (with known outcomes or with LLM judging) to serve as a health check for the deployed agent.
Leverage Frameworks and Libraries: To speed up development and ensure we’re in line with best practices, we can utilize existing evaluation frameworks:
RAGAS: as we saw, it offers ready metrics for faithfulness, context recall, answer correctness, etc., and even integrates with GPT-4 for some (though we have to supply data carefully)
docs.ragas.io
. We could use Ragas’ evaluate() with our dataset to cross-verify our metrics. It might simplify implementation of some metrics like faithfulness by providing pre-built functions (and ensure we’re consistent with definitions).
LangChain’s Evaluation module: LangChain (LangSmith) has an evaluator toolkit where you can feed in input, predicted answer, reference answer, and get graded on correctness, relevance, etc.
docs.langchain.com
. This is basically LLM-as-judge under the hood, but they have prompt templates and can output structured scores. We can explore using LangChain’s LLMChecker or similar as an alternative to writing our own judge prompts.
OpenAI Evals or other open-source eval harnesses for LLMs can also be considered if our scenario fits into QA evaluation patterns they support.
However, incorporating a new framework has overhead and might not fully align with our Telegram-specific needs (like citation checking), so we opted to design custom logic. But keeping an eye on these helps validate our approach and possibly borrow ideas (for example, DeepEval and Tweag’s suggestions mention context precision and sufficiency, which we covered)
patronus.ai
.
Security and Robustness in Evaluation: Since we’ll be calling the agent with potentially tricky queries (some might be adversarial to test it), ensure the evaluation script respects the same security filters (the agent’s SecurityManager). If a query is blocked by the agent as disallowed, that’s a valid outcome – we should note it (and probably not count it as a failure of correctness if the content was indeed disallowed). Also, handle timeouts – if an agent call hangs or fails, mark that in results and possibly retry once. We want the evaluation to complete fully even if one query causes an issue. Logging these exceptions is important for debugging (maybe the agent had a bug on a certain pattern).
Use Evaluation to Guide Agent Improvement: Finally, treat this evaluation as a living document for improving the agent. For each error category identified, create a plan to fix:
If certain question types have low accuracy, maybe add training data or few-shot examples for the model, or adjust tool logic.
If hallucinations occur, maybe tighten the prompt instructions or improve the verify threshold.
If latency is high in some cases, consider caching results or optimizing that tool’s performance.
The evaluation thus not only decides go/no-go, but tells us where to focus next. In our scenario, suppose the agent meets the acceptance criteria. We still use the eval results to say, e.g., “Focus on conciseness and refining the output format for production, since accuracy is good. Also, maybe implement multi-turn refinement if multi-hop queries are to be fully solved.” These decisions stem from the detailed metrics we gathered.
By adhering to these best practices, we ensure that the evaluation process is thorough, fair, and actionable. It will continuously keep the team informed about the agent’s performance and help maintain a high quality bar as the project evolves. The end result is confidence that the ReAct agent is not only intelligent in reasoning but also reliable, accurate, and user-friendly, as evidenced by quantifiable metrics and rigorous testing