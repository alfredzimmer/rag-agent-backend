# RAG Agent — System Proposal

**A private, on-premise question-answering assistant over engineering standards and internal documents.**


---

## 1. Executive summary

The RAG Agent is a self-hosted assistant that answers natural-language questions about a
library of technical documents — electrical and telecommunications codes, IEEE standards,
and internal reference material — and cites the exact source passages it used. It runs
entirely on company hardware: no document, question, or answer ever leaves the premises,
and there is no per-query cost to an outside vendor.

An engineer can ask *"What are the sealing requirements for optical fiber cable in Class I
hazardous locations?"* and receive a synthesized answer drawn from the Canadian Electrical
Code and TIA-EIA-568, with the underlying passages shown alongside for verification. The
system holds a conversation — follow-up questions understand prior context — and streams
its answer word by word so the user sees progress immediately.

The system is built, deployed on the internal network, and serving a web interface today.
This document describes what it does, how it works, where it runs, its current limits, and
a proposed path forward.

---

## 2. The problem it solves

Engineering and compliance work depends on a large body of reference material: the NEC, the
Canadian Electrical Code, IEEE standards, TIA/EIA cabling specifications, and internal design
guides. In practice this material is:

- **Long and fragmented** — the answer to one question may span several clauses across
  multiple documents and editions.
- **Slow to search** — keyword search finds where a term appears, not what the rule *means*
  or where the relevant requirement actually lives.
- **Easy to misremember** — staff rely on recollection of code sections, which invites error
  and inconsistency between team members.

The RAG Agent turns that library into something you can ask questions of directly, in plain
English (or Chinese), and get answers that point back to the source.

---

## 3. What the system can do today

### 3.1 Answer questions grounded in the document library
The assistant retrieves the most relevant passages from the knowledge base and composes a
direct answer from them. Answers are grounded in retrieved text rather than invented: when
the library does not contain a relevant passage, the system says so instead of guessing.

### 3.2 Show its sources
Every answer is accompanied by the actual passages that informed it, including the source
document name and section headers. A user can expand the **Retrieved context** panel to
confirm the answer against the original standard — essential for compliance work where the
citation matters as much as the answer.

### 3.3 Hold a multi-turn conversation
The assistant remembers earlier turns in a conversation. A user can introduce context
("I'm working on a data center project in Vancouver"), ask a series of related questions,
and refer back to earlier answers ("repeat just the cable codes you listed"). Verified
behavior includes recalling facts stated many turns earlier and summarizing the whole
conversation on request.

### 3.4 Stream answers in real time
Answers appear progressively as they are generated, rather than after a delay. The user also
sees the model's intermediate reasoning in a separate, collapsible panel — useful for
understanding *why* an answer was reached.

### 3.5 Let the user stop a long answer
A response in progress can be interrupted immediately if it is going in the wrong direction,
without waiting for it to finish.

### 3.6 Manage multiple conversations
The web interface provides a sidebar of past conversations with auto-generated titles,
the ability to start a new chat, switch between chats, and delete a chat.

### 3.7 Work bilingually
The current knowledge base already contains both English standards and Chinese-language
electrical design guides, and the underlying model handles both languages, so questions and
source material are not restricted to English.

---

## 4. What is in the knowledge base

The current corpus is a library of electrical and telecommunications engineering references,
including:

| Category | Examples in the library |
|---|---|
| Electrical codes | NFPA 70 (National Electrical Code, 2017); Canadian Electrical Code, Part I, 26th ed. |
| IEEE standards | IEEE Std 141 (power distribution); IEEE Std 902 (maintenance & safety) |
| Telecom / cabling | TIA-EIA-568-B.3 (optical fiber cabling components) |
| Internal / reference | Speech Transcript; Chinese-language electrical design guides |

The library is not fixed. New documents (PDF, Word, and text) can be added, and each
document is tagged with a **scope** so that material can be shared organization-wide or
kept to a specific project or conversation.

---

## 5. How it works

The system follows the **Retrieval-Augmented Generation (RAG)** pattern: instead of relying
on a language model's general training, it first *retrieves* the relevant passages from your
own documents and then asks the model to answer *using those passages*.

```
User question
   │
   ▼
1. Retrieve   — find the most relevant passages in the document library
   │            (semantic vector search over Milvus, scoped by project)
   ▼
2. Augment    — attach those passages, plus prior conversation, to the prompt
   │
   ▼
3. Generate   — a local language model composes the answer from that context,
   │            streaming it back word by word
   ▼
Answer + the exact sources used
```

Three components do the work:

- **Vector database (Milvus)** stores every document as searchable numerical
  representations, so a question retrieves passages by *meaning*, not just keyword match.
- **Language model (via Ollama)** produces the embeddings for search and generates the
  final answers. It runs locally on the GPU server.
- **Application service (FastAPI)** ties the two together, manages conversations, and
  streams results to the web interface.

---

## 6. Where it runs

The system runs entirely on the internal AI server (a GPU workstation) and is reachable on
the company network:

- **Web interface** for end users — a browser-based chat, no installation required.
- **API service** for programmatic use or integration into other internal tools.
- **Vector database and model runtime** — both local; no external services are contacted to
  answer a question.

Because everything is on-premise:

- **Data stays private.** Documents and questions never leave the company network.
- **There is no usage bill.** Answering a question costs only local compute.
- **It works offline.** The system does not depend on an internet connection or a cloud
  vendor's availability.

---

## 7. Design principles

- **Grounded, not generative guesswork.** The system is deliberately built to answer from
  retrieved sources and to surface those sources, so answers are checkable.
- **Minimal and maintainable.** The service was deliberately reduced to its essential
  components (retrieval + generation + a clean API and UI), which keeps it reliable, fast to
  start, and easy to reason about.
- **Private by construction.** On-premise hosting is a design choice, not a configuration
  option — appropriate for proprietary standards, project documents, and client material.

---

## 8. Current limitations

- **Conversation memory is per-session and in-memory.** Chat history lives in the running
  service and is not permanently stored; restarting the service clears active conversations.
  Reloading an old conversation restores the questions and answers but not the retrieval or
  reasoning detail.
- **No user accounts yet.** The current deployment does not authenticate individual users or
  segregate conversations per person; it is intended for trusted internal-network use.
- **Retrieval is single-pass semantic search.** The system does one vector search per
  question. It does not yet re-rank results, expand queries, or search the public web —
  capabilities that existed in an earlier, heavier version and can be reintroduced if the
  value justifies the added complexity.
- **Answer quality depends on the library.** The assistant can only be as good as the
  documents it holds; questions outside the loaded corpus fall back to the model's general
  knowledge, which is not authoritative for code compliance.
- **Public/off-network access is not yet available.** The system serves the internal network
  today. Secure external access is a known, separable piece of work (see §10).
- **Does not support video ingestion.**: The system does not support ingestion of videos yet,
  only transcripts of the videos.

---

## 9. Representative use cases

- **Code lookup during design.** "What clause governs conduit fill for this configuration,
  and what's the limit?" — with the exact code passage shown.
- **Cross-standard questions.** Questions whose answer spans the NEC, the Canadian code, and
  a TIA cabling spec, synthesized into one response.
- **Onboarding and training.** New staff can ask the library questions conversationally
  instead of hunting through PDFs.
  divergent interpretations of a requirement.
  isolation from the general library.

---

## 10. Proposed next steps

Ordered by typical value-to-effort. Each is independent and optional.

1. **Persistent conversations and history.** Store chats durably so they survive restarts
   and are available across devices.
2. **User accounts and access control.** Authenticate users and keep each person's
   conversations and project scopes private — a prerequisite for wider or client-facing use.
3. **Document management interface.** A simple screen to upload, tag (by project/scope), and
   remove documents, so the library can be curated without developer involvement.
4. **Retrieval quality improvements.** Re-ranking and query expansion to raise answer
   precision on hard questions; optional web search for questions the library can't cover.
5. **Secure external access.** A vetted path for staff to use the system off the company
   network, with authentication in front.
6. **Integration hooks.** Expose the API to internal tools (e.g. project software, an
   intranet portal) so answers can be surfaced where staff already work.

---

## 11. Summary

The RAG Agent is a working, private, on-premise assistant that makes the company's library
of engineering standards directly answerable in natural language, with citations, in a
conversation, at no per-query cost and with no data leaving the building. It is deployed and
serving today. The proposed next steps would take it from a capable internal tool to a
durable, multi-user, curatable platform.
