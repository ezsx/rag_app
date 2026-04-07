# Robustness Testing Artifacts

NDR/RSR/ROR robustness study (2026-04-02). Methodology: Cao et al. 2025 (adapted).

Bypass pipeline: direct Qdrant + LLM, controlled k and ordering.
151 LLM calls, BERTScore F1 + Claude judge scoring.

**Results**: NDR 0.963, RSR 0.941, ROR 0.959, Composite 0.954.
**Key finding**: BERTScore failed as robustness proxy (underestimated by 0.145).

**Total local files**: 10. **Committed samples**: judge_ndr_rsr_ror_final.json, ndr_rsr_ror_report.md.
