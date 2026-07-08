# Retrieval and Generation QA Report

Date: 2026-07-08  
Target: remote host-run API used by Streamlit (`ai-server`, `http://127.0.0.1:9230`)  
Collection under test: `rag2/rag_documents_v2`  
Test type: live end-to-end chat stream through `/api/agent/conversation/chat`

## Executive Summary

The new collection is active and retrieval is generally useful, especially for broad standards questions and the newly ingested DOCX electrical-design material. Retrieval passed or partially passed 7 of 8 probes, with one notable miss when the query explicitly asked for "North America Electrical Design Series" terminology.

Generation quality is less stable. Answers are often well grounded when the final answer is produced, but the current model configuration spends a large amount of the output budget on reasoning. One test returned no final answer at all, and another answer was cut off mid-sentence. The main production risk is not the collection switch; it is answer completeness under `reasoning=True` with the current output cap.

Overall result:

| Area | Result |
|---|---:|
| Retrieval pass or partial pass | 7 / 8 |
| Retrieval strong pass | 6 / 8 |
| Generation pass or partial pass | 6 / 8 |
| Generation strong pass | 4 / 8 |
| Hard generation failures | 1 / 8 |

## Scoring Rubric

| Score | Meaning |
|---:|---|
| 5 | Strong: relevant sources, grounded answer, citations support claims |
| 4 | Good: usable with minor omissions or framing issues |
| 3 | Partial: useful, but incomplete, overbroad, or citation discipline is uneven |
| 2 | Weak: some relevant material, but misses the user intent |
| 1 | Poor: mostly irrelevant or materially unsupported |
| 0 | Failure: no answer, hallucinated answer, or unusable result |

## Test Results

| ID | Probe | Retrieval | Generation | Notes |
|---|---|---:|---:|---|
| R1 | Fire alarm notification appliance placement or spacing | 4 | 5 | Retrieved NFPA 72, NFPA 1, and DOCX fire-alarm chunks. The final answer correctly said the retrieved sources did not contain notification-appliance spacing rules and did not invent horn/strobe requirements. |
| R2 | Grounding and bonding requirements | 5 | 3 | Retrieved highly relevant NEC, CEC, IEEE, and DOCX grounding chunks. Final answer was accurate and cited, but ended mid-sentence after hitting the output cap. |
| R3 | Electrical equipment access or working clearance | 4 | 0 | Retrieved relevant IEEE, NEC, CEC, NFPA 70E, and maintenance-clearance context. Generation consumed the output budget in reasoning and returned an empty final answer. |
| D1 | North America Electrical Design Series terminology for grounding, bonding, overcurrent | 2 | 2 | Retrieval returned mostly IEEE/NEC/CEC definition chunks instead of the DOCX series. The answer incorrectly framed the requested series as absent from the provided sources, even though the collection does contain those DOCX chunks. |
| D2 | Q&A examples for electrical room design or distribution equipment | 5 | 3 | Retrieval strongly hit new DOCX chunks from Volume 3 and NA training material. The answer gave a useful summary but admitted no explicit Q&A examples, then added an uncited "Beyond the sources" paragraph. |
| G1 | Electrical design package review checklist | 5 | 5 | Strong result. Retrieval hit design-review summary, transcript, terminology, and Q&A chunks. Final answer was concise, source-backed, and directly useful. |
| G2 | Normal vs emergency vs standby power comparison | 4 | 4 | Retrieved relevant IEEE 446, IEEE 493, NFPA 110, and NEC context. Final table was grounded and explicitly identified missing information. |
| N1 | NFPA 25-2020 sprinkler inspection frequency negative control | 3 | 4 | Retrieval found NFPA 1/NFPA 13 references discussing NFPA 25 scope, but not NFPA 25 frequency tables. Answer correctly stated the frequencies were missing, though it added general unsourced guidance under "Beyond the sources." |

## Key Findings

1. The active retrieval collection is the new Milvus target.

The QA run exercised the same host API used by Streamlit. Earlier runtime checks confirmed the API is using `MILVUS_DB=rag2` and `RAG_COLLECTION_NAME=rag_documents_v2`.

2. Retrieval quality is good for standards and broad engineering topics.

Grounding/bonding, electrical-room design, design review, emergency/standby power, and fire alarm questions generally retrieved relevant chunks. The new DOCX chunks are visible in live retrieval, especially for electrical-room and design-review prompts.

3. Retrieval is weaker when the query depends on collection-specific naming.

The phrase "North America Electrical Design Series" did not reliably pull the DOCX series chunks for the grounding terminology test. The retriever preferred standards documents with stronger semantic overlap. This suggests title/series metadata is not weighted enough during retrieval.

4. Generation has a real completeness problem.

Two answers showed output-budget symptoms:

- R2 produced a strong answer but stopped mid-sentence.
- R3 produced no final answer because reasoning used the full output budget.

This is the highest-priority quality issue because retrieval can be good while the user still receives no usable answer.

5. Citation discipline is mostly good, but "Beyond the sources" appears too readily.

The model correctly used citations in most answers. However, D2 and N1 added general knowledge after the grounded answer. That behavior is allowed by the current system prompt, but it can weaken QA confidence when the user explicitly asks for source-only answers.

## Recommended Fixes

| Priority | Fix | Why |
|---|---|---|
| P0 | Disable or reduce reasoning for production answers, or add an answer-only retry when final answer is empty. | Prevents the R3 failure mode where reasoning consumes the whole output budget. |
| P0 | Increase `RAG_LLM_NUM_PREDICT` if reasoning remains enabled. | Reduces truncation risk, especially for synthesis questions. |
| P1 | Add truncation detection before returning completion. | If the answer ends mid-sentence or is empty while output tokens hit the cap, retry with a shorter prompt or lower `top_k`. |
| P1 | Strengthen source-only behavior in the system prompt. | For QA and standards use, unsupported "Beyond the sources" content should be opt-in, not default. |
| P1 | Improve DOCX series retrieval with searchable aliases. | Add or preserve terms like "North America Electrical Design Series", "NA Electrical Design Training", volume number, and section type in searchable text or reranking features. |
| P2 | Add a small automated eval suite with fixed questions and expected source families. | Makes future ingestion and prompt changes easier to regression test. |
| P2 | Surface source metadata in the UI beyond raw context. | Helps reviewers quickly verify whether retrieved chunks came from PDFs, DOCX summaries, Q&A, terminology, or transcripts. |

## Suggested Acceptance Thresholds

For the next QA run, use these thresholds before calling the system production-ready:

| Metric | Target |
|---|---:|
| Retrieval strong pass | At least 80% |
| Generation strong pass | At least 75% |
| Empty final answers | 0 |
| Truncated final answers | 0 |
| Source-only prompts with unsourced extra content | 0 |
| DOCX-specific probes retrieving DOCX chunks | At least 80% |

## Follow-Up Test Set

Keep these fixed probes for regression testing:

1. What sources discuss fire alarm notification appliance placement or spacing requirements? Answer only from retrieved sources and cite them.
2. What does the knowledge base say about grounding and bonding requirements for electrical systems? Cite the retrieved sources.
3. According to the retrieved sources, what are the requirements for maintaining electrical equipment access or working clearance?
4. In the North America Electrical Design Series, what terminology or definitions are given for grounding, bonding, or overcurrent protection?
5. What Q&A examples discuss electrical room design or distribution equipment? Summarize the examples with citations.
6. Create a concise checklist for reviewing an electrical design package based only on the retrieved sources.
7. Compare the guidance for normal power, emergency power, and standby power using a table. Use only the retrieved sources.
8. What are the NFPA 25-2020 sprinkler inspection frequency requirements? If the retrieved sources do not contain them, say what is missing.

## Conclusion

The retrieval layer is mostly healthy on the new collection, and the DOCX ingestion is visible to the live application. The largest quality blocker is generation reliability, especially output-budget exhaustion caused by reasoning. Fix that first, then tune DOCX-specific retrieval aliases and source-only answer behavior.
