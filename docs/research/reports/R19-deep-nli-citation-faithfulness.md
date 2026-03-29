# NLI citation faithfulness for Russian RAG: a concrete implementation plan

**The Hybrid approach (C) — Qwen3 for claim decomposition plus XLM-RoBERTa-large-xnli for NLI verification — is the clear winner for your hardware and language constraints.** This architecture delivers independent faithfulness verification at ~83.5% NLI accuracy on Russian, fits within your VRAM budget (~1.12 GB FP16), and avoids the circular dependency of using the same LLM as both generator and judge. Start with eval-only integration in `evaluate_agent.py`, targeting a baseline faithfulness score of **0.85** and iterating toward 0.90+. The 0.92 threshold is achievable but ambitious — treat it as a stretch goal that requires co-optimizing retrieval and generation.

---

## Why Approach C wins: the three-way comparison

Your three candidate architectures differ fundamentally in independence, accuracy, and operational cost. After analyzing Russian NLI benchmarks, VRAM constraints, and latency profiles, Approach C dominates.

**Approach A (LLM-only)** uses Qwen3-30B-A3B for both decomposition and verification. This is the fastest path to a prototype — no new model needed, and Qwen3 handles Russian exceptionally well (ranked #1 open-source LLM for Russian by SiliconFlow 2026). However, it perpetuates the exact problem you identified: the same model that generated potentially hallucinated claims now judges them. LLM-as-judge achieves only **ρ = 0.55 Spearman correlation** with human faithfulness judgments, and cross-lingual LLM judging shows Fleiss' κ ≈ 0.30 across languages. For factual consistency specifically, this approach has a fundamental conflict of interest.

**Approach B (dedicated NLI model only)** eliminates LLM dependency entirely but requires a separate claim decomposition strategy. No dedicated NLI model handles Russian claim extraction natively — they only classify premise-hypothesis pairs. You would need to either manually segment sentences (losing atomic granularity) or use a separate extraction model that doesn't exist for Russian.

**Approach C (Hybrid)** leverages each component's strength: Qwen3-30B-A3B performs claim decomposition in Russian (where LLMs excel — structured extraction, not judgment), and a dedicated NLI model performs binary entailment classification (where encoder models are fast, independent, and well-benchmarked). The NLI model acts as an **orthogonal verification signal** — it has never seen the generation prompt, has no access to the agent's reasoning, and operates purely on textual entailment.

| Criterion | A: LLM-only | B: NLI-only | C: Hybrid |
|-----------|-------------|-------------|-----------|
| Russian NLI accuracy | ~85-88% (estimated, LLM-based) | 83.5% (XLM-R XNLI) | 83.5% (XLM-R XNLI) |
| Independence from generator | ❌ Same model | ✅ Fully independent | ✅ Verification independent |
| New VRAM required | 0 GB | ~1.12 GB (FP16) | ~1.12 GB (FP16) |
| Latency per answer | 2-4s (2+ LLM calls) | ~0.2s NLI only (no decomposition) | 1.5-3.5s (1 LLM call + NLI batch) |
| Claim decomposition quality | ✅ Excellent (Qwen3 Russian) | ❌ No native solution | ✅ Excellent (Qwen3 Russian) |
| Circular bias risk | ⚠️ High | ✅ None | ✅ Low (decomposition only) |
| Operational complexity | Low | Medium | Medium |
| Recommended | No | No | **Yes** |

---

## The Russian NLI model landscape: what actually works

The critical question — which NLI model handles Russian faithfulness verification best — has a nuanced answer depending on your priorities.

**XLM-RoBERTa-large-xnli** (`joeddav/xlm-roberta-large-xnli`) achieves **83.5% accuracy on the XNLI Russian test set** and **0.798 on TERRa** (Russian SuperGLUE textual entailment). With 560M parameters at ~1.12 GB in FP16, it fits comfortably on your RTX 5060 Ti alongside existing models. This is the strongest general-purpose multilingual NLI model for Russian, with proven zero-shot classification capabilities (demonstrated with Russian examples in the model card). FP16 inference is fully supported.

**cointegrated/rubert-base-cased-nli-threeway** is the only Russian-native NLI model purpose-built for 3-way classification (entailment/contradiction/neutral — exactly what faithfulness verification needs). At just **180M parameters (~0.36 GB FP16)**, it's 3× smaller than XLM-R-large. It achieves **0.90 ROC AUC for entailment on TERRa dev** and was trained on ~1.9M NLI pairs auto-translated to Russian. The trade-off: training data is machine-translated from English, which can introduce artifacts, and it lacks multilingual fallback.

**mDeBERTa-v3-base-xnli-multilingual-nli-2mil7** scores **80.3% on XNLI Russian** with 279M parameters, but has a critical practical issue: **FP16 inference is broken** due to a known mDeBERTa bug, forcing FP32 deployment at ~1.12 GB — the same size as XLM-R-large in FP16 but with lower accuracy.

**DeBERTa-v3-large-mnli-fever-anli-ling-wanli is English-only and cannot process Cyrillic text at all.** The model author explicitly states it requires machine translation to English for non-English use. There is no multilingual DeBERTa-v3-large — only the base size exists multilingually.

MiniCheck, AlignScore, and LettuceDetect — the leading dedicated faithfulness checkers — **all lack Russian support**. MiniCheck is English-only (trained on English ANLI + synthetic data). AlignScore uses English RoBERTa. LettuceDetect covers 7 languages via EuroBERT but not Russian, though EuroBERT's pre-training included Russian, making adaptation feasible through fine-tuning on translated RAGTruth data.

| Model | Russian XNLI | TERRa | Params | FP16 Size | 3-way NLI | FP16 OK |
|-------|-------------|-------|--------|-----------|-----------|---------|
| XLM-RoBERTa-large-xnli | **83.5%** | 0.798 | 560M | 1.12 GB | ✅ | ✅ |
| ruBERT-base-nli-threeway | — | 0.90 AUC | 180M | 0.36 GB | ✅ | ✅ |
| mDeBERTa-v3-base-xnli | 80.3% | 0.783 | 279M | ❌ broken | ✅ | ❌ |
| DeBERTa-v3-large-mnli | ❌ English-only | — | 434M | 0.87 GB | ✅ | ✅ |

**Recommendation: Deploy XLM-RoBERTa-large-xnli as primary, with ruBERT-base-nli-threeway as a lightweight fallback.** XLM-R-large provides the highest benchmarked accuracy on Russian NLI. If VRAM becomes tight or you need faster inference, ruBERT-base is 3× smaller with competitive Russian performance. Run an A/B comparison on your first 30 golden questions to determine which model produces more reliable faithfulness scores for your specific news domain.

---

## The decompose-then-verify pipeline in detail

The implementation follows a three-stage architecture: decompose claims from the Russian answer, verify each claim against cited documents via NLI, then aggregate scores.

### Stage 1: Claim decomposition via Qwen3-30B-A3B

Qwen3 is trained on 36 trillion tokens across **119 languages** with Russian explicitly supported and ranked as the best open-source LLM for Russian. No specialized Russian claim extraction models exist — LLM prompting is the only viable path. Use this Russian-language prompt:

```
Разбей следующий ответ на независимые атомарные утверждения.
Каждое утверждение должно:
- Содержать ровно один проверяемый факт
- Быть полностью понятным без контекста (раскрой все местоимения)
- Быть сформулировано как утвердительное предложение

Классифицируй каждое утверждение:
- "verifiable" — требует подтверждения документом
- "common_knowledge" — общеизвестный факт
- "meta" — структурное/связующее высказывание

Верни JSON: {"claims": [{"text": "...", "type": "verifiable"}, ...]}

Ответ: {answer}
```

This prompt resolves pronouns, excludes meta-statements from verification, and flags common knowledge to avoid false penalties. Expected output: **5-12 claims per typical RAG answer**, with decomposition taking **1-3 seconds** on Qwen3-30B-A3B (3B active parameters via MoE handle short structured generation efficiently).

Known failure modes for non-English decomposition include: vague anaphoric claims that lose context, redundant overlapping claims, and incomplete extraction that misses implicit facts. Mitigate these by including 2-3 few-shot examples in Russian within the prompt.

### Stage 2: NLI verification

For each verifiable claim, run NLI against every cited document. The XLM-RoBERTa-large-xnli model classifies each (document, claim) pair as entailment/neutral/contradiction:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class NLIVerifier:
    LABELS = ["contradiction", "neutral", "entailment"]
    
    def __init__(self, model_name="joeddav/xlm-roberta-large-xnli", device="cuda:1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device).eval()
        self.device = device
    
    def verify_batch(self, claims: list[str], documents: list[str], 
                     batch_size: int = 16) -> list[dict]:
        pairs = [(doc, claim) for claim in claims for doc in documents]
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            inputs = self.tokenizer(
                [p[0] for p in batch], [p[1] for p in batch],
                padding=True, truncation=True, max_length=512,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = logits.softmax(dim=-1).cpu()
            for j, (doc, claim) in enumerate(batch):
                p = probs[j]
                results.append({
                    "claim": claim, "document": doc[:100],
                    "label": self.LABELS[p.argmax().item()],
                    "scores": {l: p[k].item() for k, l in enumerate(self.LABELS)}
                })
        return results
```

For a typical answer with **8 claims and 4 cited documents = 32 NLI pairs**, batched GPU inference on the RTX 5060 Ti takes approximately **30-160ms total** — negligible compared to the decomposition step.

### Stage 3: Score aggregation with strict and lenient modes

Each claim receives a score based on its best entailment result across all cited documents:

```python
def aggregate_faithfulness(claim_results: list[dict], mode: str = "lenient") -> dict:
    claim_scores = {}
    for r in claim_results:
        claim = r["claim"]
        if claim not in claim_scores:
            claim_scores[claim] = {"best_entailment": 0, "max_contradiction": 0}
        claim_scores[claim]["best_entailment"] = max(
            claim_scores[claim]["best_entailment"], r["scores"]["entailment"])
        claim_scores[claim]["max_contradiction"] = max(
            claim_scores[claim]["max_contradiction"], r["scores"]["contradiction"])
    
    scores = []
    for claim, s in claim_scores.items():
        if s["best_entailment"] > 0.5:       # Supported
            scores.append(1.0)
        elif s["max_contradiction"] > 0.5:    # Contradicted
            scores.append(0.0)
        else:                                  # Neutral/extrinsic
            scores.append(0.5 if mode == "lenient" else 0.0)
    
    return {
        "faithfulness": sum(scores) / len(scores),
        "supported": scores.count(1.0),
        "contradicted": scores.count(0.0),
        "neutral": len(scores) - scores.count(1.0) - scores.count(0.0),
        "total_claims": len(scores)
    }
```

The **lenient mode** (0.5 for neutral claims) is the recommended default — it avoids penalizing common knowledge and extrinsic-but-correct facts. **Strict mode** (0.0 for neutral) is appropriate for high-stakes evaluation where every claim must be document-grounded.

---

## Formulas: four metrics you need to track

Your system should compute four distinct metrics that capture different failure modes:

**Faithfulness (Factual Precision against context)**:
`Faithfulness = (Σ claim_score_i) / N_claims` where each score is 1.0 (entailed), 0.5 (neutral, lenient), or 0.0 (contradicted). This measures whether the generated answer stays within the bounds of retrieved documents.

**Citation Recall** (ALCE formula):
`Citation_Recall = (1/N) × Σ_i 𝟙[NLI(concat(C_i), a_i) = entailment]`
Fraction of statements where the concatenation of all their cited passages entails the statement. This measures whether every claim has adequate citation support.

**Citation Precision** (ALCE formula):
`Citation_Precision = 1 - (|irrelevant citations| / |total citations|)`
A citation is irrelevant if removing it doesn't change the entailment judgment and it alone doesn't support the claim. This measures whether superfluous citations are attached.

**Factual Correctness** (reference-based, your existing metric):
`Factual_Correctness = claims_matching_expected_answer / total_expected_claims`
This is your current evaluation signal — keep it. It measures whether the answer is actually right, independent of grounding.

The critical insight: **faithfulness and factual correctness are positively correlated but not interchangeable.** ALCE benchmark results show models can be factually correct but poorly grounded (using parametric knowledge without citations), or well-cited but wrong (if retrieved sources are incorrect). High faithfulness with low correctness signals retrieval problems. Low faithfulness with high correctness signals the model is ignoring context. Track both.

---

## VRAM budget and deployment architecture

Your RTX 5060 Ti (16 GB) currently uses ~5 GB, leaving **~11 GB free**. XLM-RoBERTa-large-xnli in FP16 requires **~1.7 GB including runtime overhead** (1.12 GB weights + ~500 MB CUDA context and activation buffers). This leaves **~9.3 GB free** — more than enough headroom.

For eval-only mode, however, **CPU deployment is the pragmatic choice**. The LLM decomposition step on the V100 takes 1-3 seconds and dominates total latency regardless of NLI inference speed. Running NLI on CPU via ONNX Runtime with INT8 quantization delivers **~40-80ms per claim pair** — adding only ~1.3-2.6 seconds for 32 pairs sequentially, or ~400ms with batching. This keeps both GPUs free for their primary workloads.

The recommended deployment path:

```
Phase 1 (Eval-only, CPU):
  evaluate_agent.py → Qwen3 decomposition (V100) → NLI verification (CPU ONNX INT8)
  Latency: ~2-4s per answer (async, no user impact)
  VRAM impact: 0 GB additional

Phase 2 (Runtime monitoring, GPU):  
  After calibrating thresholds on golden set, add NLI to RTX 5060 Ti (FP16)
  Latency: ~1.5-3.5s added to response pipeline
  VRAM impact: ~1.7 GB on 5060 Ti

Phase 3 (Runtime enforcement):
  Low-faithfulness answers trigger re-generation or claim filtering
  Requires fallback strategy when faithfulness < threshold
```

For ONNX export and INT8 quantization:
```bash
pip install optimum[onnxruntime]
optimum-cli export onnx --model joeddav/xlm-roberta-large-xnli \
  --task zero-shot-classification onnx_xlmr_nli/
# Then quantize with AutoQuantizationConfig.avx512_vnni()
```

---

## The 0.92 faithfulness target: calibration against reality

Published RAGAS evaluations and production benchmarks provide clear calibration points. Elastic Labs reports **average faithfulness of 0.75** in their demo pipelines. Production guidance from multiple sources converges on **0.85 as a solid target** and **0.90+ as well-grounded**. The Vectara hallucination leaderboard shows even frontier models (GPT-5, Claude Sonnet 4.5) hallucinate **>10%** of the time on enterprise documents.

For your specific setup — Russian news domain, 13K documents, ReAct agent with 13 tools — several factors affect the achievable score:

- **Retrieval quality matters enormously.** Your Strict Recall@5 of 0.342 means roughly two-thirds of relevant documents aren't retrieved in the top 5. Claims the agent makes based on documents outside the top-5 will appear as "neutral" to the NLI verifier, dragging down faithfulness scores. Improving retrieval directly improves measurable faithfulness.
- **News domain is favorable.** News articles are factual, well-structured, and contain explicit claims — ideal for NLI-based verification. This is easier than creative writing or technical reasoning.
- **Russian NLI accuracy ceiling of ~83.5%** means roughly 16.5% of individual NLI judgments may be incorrect. At the answer level, errors partially cancel out across multiple claims, but this introduces noise in the metric.
- **Agent tool calls add complexity.** A ReAct agent with 13 tools may synthesize information across multiple documents and tool results, producing more extrinsic claims than a simple RAG pipeline.

**Realistic trajectory: Start measuring at whatever baseline emerges (likely 0.70-0.80), target 0.85 within the first optimization cycle, and treat 0.90+ as a quality milestone.** Reaching 0.92 consistently will require improving retrieval recall alongside generation faithfulness — these are coupled problems.

---

## Integration with existing compose_context.py

Your `compose_context.py` already computes a 6-signal composite coverage score — a retrieval quality signal. The NLI faithfulness score is a complementary **generation quality signal**. Together they form a complete picture:

```python
# New file: verify_faithfulness.py
class FaithfulnessVerifier:
    def __init__(self, nli_model_path, llm_client):
        self.nli = NLIVerifier(nli_model_path)
        self.llm = llm_client  # Qwen3 for decomposition
    
    def evaluate(self, answer: str, cited_docs: list[str], 
                 coverage_score: float, mode: str = "lenient") -> dict:
        # Stage 1: Decompose
        claims = self.llm.decompose_claims(answer)
        verifiable = [c for c in claims if c["type"] == "verifiable"]
        
        # Stage 2: NLI verify
        nli_results = self.nli.verify_batch(
            [c["text"] for c in verifiable], cited_docs)
        
        # Stage 3: Aggregate
        faithfulness = aggregate_faithfulness(nli_results, mode)
        
        # Combined signal
        return {
            "faithfulness_score": faithfulness["faithfulness"],
            "coverage_score": coverage_score,  # from compose_context
            "combined_quality": 0.6 * faithfulness["faithfulness"] + 0.4 * coverage_score,
            "claim_details": faithfulness,
            "total_claims": len(claims),
            "verifiable_claims": len(verifiable),
            "common_knowledge_claims": len([c for c in claims if c["type"] == "common_knowledge"])
        }
```

In `evaluate_agent.py`, add faithfulness alongside existing metrics:
```python
# After running each test case
verification = verifier.evaluate(
    answer=agent_response.text,
    cited_docs=[doc.content for doc in agent_response.citations],
    coverage_score=agent_response.coverage
)
metrics["faithfulness"] = verification["faithfulness_score"]
metrics["citation_grounding"] = verification  # Full detail
```

---

## Step-by-step implementation plan

**Week 1 — Foundation:**
1. Install `joeddav/xlm-roberta-large-xnli` and `cointegrated/rubert-base-cased-nli-threeway`
2. Write `NLIVerifier` class with batch inference support
3. Test both models on 10 manually constructed Russian (claim, document) pairs from your Qdrant corpus to validate NLI quality on your actual news data
4. Choose the model that produces more intuitive entailment judgments on your domain

**Week 2 — Decomposition pipeline:**
5. Develop and iterate the Russian claim decomposition prompt for Qwen3-30B-A3B
6. Test on 10 answers from your golden set, manually verify claim quality
7. Handle edge cases: very short answers (1-2 claims), very long answers (15+ claims), answers with code or lists
8. Implement the `verifiable` / `common_knowledge` / `meta` classification

**Week 3 — Eval integration:**
9. Implement `FaithfulnessVerifier` class
10. Integrate into `evaluate_agent.py` — run on full golden_v1 (30 questions)
11. Establish baseline faithfulness score
12. Compare lenient vs strict mode scores
13. Export ONNX INT8 model for CPU deployment if GPU contention is an issue

**Week 4 — Analysis and calibration:**
14. Analyze results: which claims fail verification? Are they genuine hallucinations or NLI errors?
15. Manually review the 10 lowest-faithfulness answers to calibrate threshold
16. Compare faithfulness scores with your existing factual correctness scores (1.79/2) — measure correlation
17. Set production threshold based on empirical distribution
18. Document failure modes and create a report on model accuracy for your specific domain

---

## Risks and how to mitigate them

**Risk 1: NLI accuracy on Russian news is lower than XNLI benchmarks.** XNLI uses formal, well-edited text; Telegram channel content may be informal, contain jargon, or use slang. Mitigation: benchmark on 50+ real examples from your corpus before trusting scores. If accuracy drops below 75%, consider fine-tuning XLM-R-large on a small set of manually annotated (claim, news_article, label) triples from your domain — even 500-1000 examples can significantly improve domain-specific performance.

**Risk 2: Claim decomposition introduces errors that propagate.** Research shows decomposition benefits weaker verifiers but can hurt stronger ones. Over-decomposition creates redundant or trivially true claims that inflate scores; under-decomposition misses important sub-claims. Mitigation: validate decomposition quality manually on 20-30 examples. Use the DecMetrics framework (completeness, correctness, semantic entropy) to measure decomposition quality.

**Risk 3: Neutral claims dominate, making the score uninformative.** If the agent frequently makes claims that are correct but not explicitly stated in documents (extrinsic knowledge), the faithfulness score in strict mode will be artificially low. Mitigation: start with lenient mode (neutral=0.5), separately track the ratio of entailed/neutral/contradicted claims, and use the `common_knowledge` filter in decomposition to exclude obviously general facts.

**Risk 4: 512-token NLI context window truncates long documents.** Your Qdrant points may contain documents longer than 512 tokens. XLM-RoBERTa-large has a 512-token limit — longer documents get truncated, potentially losing the relevant passage. Mitigation: chunk documents to 400 tokens with overlap before NLI, and take the maximum entailment score across chunks. Your existing chunking in Qdrant likely already handles this.

**Risk 5: Latency makes runtime deployment impractical.** The 1.5-3.5 second overhead for decomposition + verification may be unacceptable for SSE streaming responses. Mitigation: keep faithfulness as eval-only initially. For runtime, consider a lightweight "red flag" check — skip full decomposition and run NLI directly on each sentence of the answer against cited documents. This is less precise but 3-5× faster.

---

## Conclusion

The path to reliable faithfulness verification in your Russian RAG pipeline is concrete and achievable with your existing hardware. The Hybrid approach (Qwen3 decomposition + XLM-RoBERTa-large NLI) resolves the fundamental limitation you identified — LLM-as-judge cannot distinguish hallucination from grounded fact because it sees the same context. An independent NLI model breaks this circular dependency.

Three insights emerged from this research that should guide your implementation priorities. First, **your retrieval recall (0.342) is a bigger faithfulness bottleneck than generation quality** — claims scored as "neutral" by NLI may simply reflect documents the retriever didn't surface, not hallucination. Improving Strict Recall@5 will mechanically improve measurable faithfulness. Second, **the claim decomposition step matters more than the NLI model choice** — errors in decomposition propagate directly into score distortion, while the top Russian NLI models differ by only ~3 percentage points. Invest heavily in prompt engineering and manual validation of claim extraction. Third, **LettuceDetect adaptation for Russian is a compelling medium-term option** — EuroBERT was pre-trained on Russian, and the Turkish adaptation recipe (translate RAGTruth, fine-tune) provides a clear template for building a purpose-built Russian hallucination detector that operates at token level without needing claim decomposition at all.