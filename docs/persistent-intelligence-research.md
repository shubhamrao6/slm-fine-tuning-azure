# Persistent Intelligence Systems: Complete Research Document

**Author:** Shubham Rao
**Date:** June 2026
**Status:** Research direction — pre-prototype
**Last Updated:** 23 June 2026 (incorporates June 2026 papers)

---

## 1. Executive Summary

This document outlines a research program to build AI systems that develop genuine expertise over time through continued interaction — not by accumulating tokens in context windows, but by evolving their understanding at the structural level.

The core thesis: **A persistent intelligence system must have three memory layers (experience, knowledge graph, model weights) connected by an Evolution Engine that autonomously promotes knowledge from raw interaction to structured understanding to internalized expertise.**

This work builds directly on our BMVC 2026 paper (answer-conditioned CoT distillation for VLMs), extending the technique from one-shot domain adaptation to continuous personal adaptation.

---

## 2. The Problem

Current AI systems are stateless. Every session starts fresh. The industry's response has been:
- Larger context windows (4K → 128K → 1M+ tokens)
- RAG systems (retrieve relevant chunks from a vector store)
- Memory systems (store key-value pairs of user facts)

None of these solve the fundamental problem: **the model itself does not become better at helping a specific user over time.** It retrieves more, but it doesn't learn.

The difference matters:
- **Retrieval**: "I see from my notes that you work on cloud architecture"
- **Learning**: "Based on how you approach problems, you'll want a serverless design here — let me explain why in terms of the tradeoffs you typically prioritize"

The second response reflects internalized understanding. It doesn't come from looking up a fact. It comes from months of interaction shaping how the model reasons about your specific needs.

---

## 3. The Architecture

### 3.1 Three Memory Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERACTION                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: EXPERIENCE STORE                                   │
│  Raw interactions stored verbatim. Append-only.              │
│  Infinite capacity. Cheap. Not intelligence — just history.  │
└─────────────────┬───────────────────────────────────────────┘
                  │ (real-time extraction)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: KNOWLEDGE GRAPH (Temporal)                         │
│  Entities, relationships, facts with time validity.          │
│  Structured understanding. Queryable. Explainable.           │
└─────────────────┬───────────────────────────────────────────┘
                  │ (periodic consolidation via Evolution Engine)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: WEIGHT MEMORY (Temporal LoRA Stack)                │
│  Internalized expertise encoded in model parameters.         │
│  No retrieval needed — the model just "knows."               │
│  Stratified by age: staging → consolidated → expertise       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 The Evolution Engine (Dual-Process Architecture)

Inspired by DCPM (Jun 2026), the Evolution Engine is split into two processes following dual-process cognitive theory:

**System 1 — Synchronous "Daytime" Writer (runs during interaction):**
- Records belief revisions as doubly-linked supersedes chains in the graph
- Extracts entities and relationships in real-time (via Graphiti)
- Optionally applies fast-weight Δₜ updates for immediate in-session adaptation (TMEM-inspired)
- Lightweight, fast, doesn't block the conversation

**System 2 — Asynchronous "Nighttime" Engine (runs offline, periodically):**
1. **Schema Induction**: Identifies stable patterns across multiple sessions
2. **Cross-Domain Collision Detection**: Finds concepts that bridge different topic areas
3. **Concept Formation**: Detects emergent higher-order abstractions
4. **Belief Trajectory Analysis**: Tracks how the user's understanding evolves over time
5. **CoT Curriculum Generation**: Creates training data using answer-conditioned reasoning (our BMVC technique)
6. **Weight Update**: Applies LoRA updates to reasoning-skill adapters ONLY
7. **Validation**: Tests that new knowledge didn't break existing capabilities
8. **Pruning**: Removes redundant entries from graph once patterns are consolidated into skills

System 2 is triggered on a schedule (nightly) or when the graph reaches a complexity threshold.

### 3.3 The Temporal LoRA Stack

**Critical design principle (validated by "User as Engram", Jun 2026):** Weight memory must ONLY store reasoning skills and patterns — never individual facts. Facts stay in the graph or in surgical Engram-table edits. LoRA teaches the model *how to think about this user*, not *what happened on a specific date*.

Instead of a single LoRA adapter, the system maintains stratified adapters by knowledge age:

**Fast Δₜ (intra-session working memory):**
- Updated: within a single conversation, as useful patterns emerge
- Mechanism: lightweight online LoRA update (inspired by TMEM, Jun 2026)
- SVD-based initialization of the LoRA subspace for fast convergence
- Completely ephemeral — discarded at session end unless promoted
- Purpose: immediate adaptation within the current interaction
- Analogy: your train of thought right now

**LoRA_staging** (weekly working memory):
- Updated: daily or after every few interactions
- Learning rate: high
- Override resistance: low (easy to correct)
- Purpose: captures recent reasoning patterns that have proven useful across multiple sessions
- Analogy: sticky notes on your desk

**LoRA_consolidated** (proven knowledge):
- Updated: weekly or monthly
- Learning rate: medium
- Override resistance: medium (needs repeated contradictions to change)
- Purpose: stores validated reasoning skills (e.g., "this user prefers serverless architectures — always consider that angle")
- Analogy: notes you organized into a proper notebook

**LoRA_expertise** (deep understanding):
- Updated: monthly or quarterly
- Learning rate: very low
- Override resistance: high (nearly impossible to override with single interaction)
- Purpose: encodes deeply-learned reasoning habits and domain expertise
- Analogy: knowledge so internalized you don't consciously think about it

**Separate fact storage (Engram-style):**
- Personal facts stored as surgical hash-table edits (NOT in LoRA)
- Composable: multiple users' facts coexist without interference
- 33,000x smaller memory footprint than per-user LoRA for facts
- Facts are addressable, editable, deletable without touching reasoning skills

**Promotion criteria:**
```
Promotion Score = Importance × Frequency × Stability × Utility

Where:
- Importance: how central is this knowledge to the user's goals?
- Frequency: how often does it come up?
- Stability: how long has it remained uncontradicted?
- Utility: has it actually helped in past interactions?
```

**Demotion/invalidation:**
When knowledge is contradicted (user changes jobs, corrects a misunderstanding), the system doesn't immediately delete deep knowledge. Instead:
- New fact enters LoRA_staging
- If contradictions persist over time, the old deep knowledge gets weakened
- Eventually the new fact displaces the old one through repeated promotion

This prevents the system from being easily manipulated while remaining correctable over time.

### 3.4 Context Compilation (Inference Time)

At query time, the system doesn't just "retrieve" — it compiles context:

1. The personal LoRA stack is already active (deep knowledge requires no retrieval)
2. Query the knowledge graph for relevant entities/relationships to the current question
3. Pull any very recent interactions on this topic (last few turns)
4. Assemble a minimal, high-signal context package

Key insight: **as the system matures, less retrieval is needed** because more knowledge lives in weights. The context window gets shorter and more focused over time — the opposite of what happens with current systems.

---

## 4. Connection to Prior Work (Our BMVC 2026 Paper)

Our BMVC paper demonstrated:
1. A frontier model (GPT-4.1) can generate justified visual reasoning conditioned on the correct answer
2. A small model (Qwen2.5-VL-3B) can absorb this reasoning into its weights via LoRA
3. The small model becomes better than both the frontier model and direct fine-tuning
4. This works across multiple domains with minimal data (18-30 images)

**The direct extension to Persistent Intelligence:**

| BMVC Paper | Persistent Intelligence |
|------------|------------------------|
| Training images with labels | Knowledge graph patterns with validated facts |
| Human provides correct labels | The graph (validated over time) provides ground truth |
| GPT-4.1 generates justified reasoning | GPT-4.1 generates personalized reasoning curriculum |
| One-shot LoRA training | Continuous LoRA updates across temporal stack |
| Fixed domain (concrete, steel, etc.) | Evolving personal domain (user's expertise grows) |

The technique is the same. The application becomes autonomous and continuous rather than manual and one-time.

---

## 5. Related Work and Positioning

### 5.1 Memory Systems for LLM Agents

| System | Year | What it does | Relation to our work |
|--------|------|-------------|---------------------|
| **Zep/Graphiti** | 2025 | Temporal knowledge graph from conversations. Auto-extracts entities, facts with validity windows. | **We use this** as our Layer 2 implementation. |
| **Mem0** | 2024 | Simple key-value memory with LLM extraction. | Too simple for our needs — no graph structure, no temporal reasoning. |
| **AriGraph** | 2024 | Episodic + semantic graph for agent exploration. | Relevant architecture but focused on game environments, not personalization. |
| **SAGE** | May 2025 | Self-evolving graph with writer-reader architecture and Graph Foundation Model. | Closest to our Evolution Engine concept but lacks weight-level consolidation. |

**References:**
- Zep: "A Temporal Knowledge Graph Architecture for Agent Memory" (arxiv 2501.13956)
- AriGraph: "Learning Knowledge Graph World Models with Episodic Memory for LLM Agents" (arxiv 2407.04363)
- SAGE: "A Self-Evolving Agentic Graph-Memory Engine for Structure-Aware Associative Memory" (arxiv 2605.12061)

### 5.2 Continual Learning and Memory Consolidation

| System | Year | What it does | Relation to our work |
|--------|------|-------------|---------------------|
| **SCM (Sleep-Consolidated Memory)** | Apr 2025 | Neuroscience-inspired consolidation with sleep phases, forgetting, engram maturation. | Directly relevant — implements "artificial sleep." We adapt this concept for our Evolution Engine timing. |
| **Human-Inspired Memory Architecture** | May 2025 | Six cognitive mechanisms including sleep consolidation, entity KGs, hybrid retrieval. | Very close to our architecture. Key difference: they don't connect to weight-level adaptation. |
| **Continuum Memory Architecture (CMA)** | Jan 2025 | Defines the architecture class: persistent storage, selective retention, associative routing, temporal chaining, consolidation into abstractions. | The theoretical framework we're implementing. Our contribution is the concrete mechanism (CoT-based curriculum generation into temporal LoRA stack). |
| **The Continuity Layer** | Apr 2025 | Position paper arguing this is "the most consequential infrastructure not yet built." | Validates the importance of the problem. We're building what they describe. |

**References:**
- SCM: "Sleep-Consolidated Memory with Algorithmic Forgetting for Large Language Models" (arxiv 2604.20943)
- Human-Inspired: "Human-Inspired Memory Architecture for LLM Agents" (arxiv 2605.08538)
- CMA: "Continuum Memory Architectures for Long-Horizon LLM Agents" (arxiv 2601.09913)
- Continuity Layer: arxiv 2604.17273

### 5.3 Self-Adapting Models and Weight Updates

| System | Year | What it does | Relation to our work |
|--------|------|-------------|---------------------|
| **SEAL** | Jun 2025 | Model generates own fine-tuning data + optimization instructions via RL. | Core inspiration. We adapt SEAL for personalization: the graph provides the "task" and CoT generation provides the "self-edit." |
| **Weight-Based Consolidation vs. Cascading Compaction** | May 2025 | Compares per-user LoRA updates against context workarounds. | Directly validates our approach (per-user weights > context stuffing). |
| **Modular Memory for Continual Learning** | Mar 2025 | Separates ICL (fast adaptation) from IWL (stable weight updates). | Maps to our staging (ICL-like) vs. expertise (IWL-like) distinction. |
| **Titans/MIRAS** | Jan 2025, Google | Neural memory that updates its own weights at inference time using surprise. | Relevant for the "real-time" update mechanism. Could replace LoRA_staging for immediate adaptation. |

**References:**
- SEAL: "Self-Adapting Language Models" (arxiv 2506.10943)
- Weight Consolidation: arxiv 2605.24657
- Modular Memory: "Modular Memory is the Key to Continual Learning Agents" (arxiv 2603.01761)
- Titans: "Learning to Memorize at Test Time" (arxiv 2501.00663)

### 5.4 Temporal Awareness in Models

| System | Year | What it does | Relation to our work |
|--------|------|-------------|---------------------|
| **TiMoE** | 2025 | Time-Aware Mixture of Experts — masks experts by training time period. | Architecture inspiration for temporal LoRA: different adapters for different "eras" of knowledge. |
| **Temporal Domain Generalization** | Feb 2025 | Models how optimal weight configurations drift over time on a manifold. | Theoretical grounding for why temporal stratification makes sense. |
| **"Do LLMs Know Time Passes?"** | Jun 2025 | Empirically tests temporal awareness in LLMs. | Validates that models CAN represent temporal information — our stack makes it explicit. |

**References:**
- TiMoE: "Time-Aware Mixture of Language Experts" (arxiv 2508.08827)
- Temporal DG: "Manifold-Aware Temporal Domain Generalization for Large Language Models" (arxiv 2602.11965)
- Time Perception: "Do Language Models Know Time Passes?" (arxiv 2506.05790)

### 5.5 June 2026 Papers (Most Recent, Directly Relevant)

| System | Date | What it does | Lesson for us |
|--------|------|-------------|---------------|
| **"Language Models Need Sleep"** (Google) | Jun 2 2026 | "Sleep" paradigm: Memory Consolidation (Knowledge Seeding via distillation) + Dreaming (RL-generated curriculum for self-improvement). | Validates consolidation concept. Their RL curriculum ≈ our CoT curriculum. Key diff: they lack KG grounding, we have ground-truth from graph. |
| **TMEM** | Jun 3 2026 | Fast LoRA weights Δₜ updated WITHIN a single episode. SVD-based initialization. RL optimizes what gets written to parametric memory. | Add intra-session fast-weight layer. Use SVD init for LoRA subspace. |
| **DCPM** | Jun 8 2026 | Dual-process memory hierarchy: System1 (realtime belief chains) + System2 (async schema/intention induction). Cross-domain pattern detection. | Split Evolution Engine into System1/System2. Add belief trajectories and cross-domain collision detection. |
| **Engram: Bi-Temporal Memory Engine** (MIT) | Jun 14 2026 | Open-source dual-process memory with bi-temporal data model. "Lean retrieved context beats full history." | Potential Layer 2 alternative/complement to Graphiti. |
| **"User as Engram"** | Jun 17 2026 | Per-user facts as surgical hash-table edits (NOT LoRA). LoRA for shared reasoning skills only. 33,000x smaller footprint. 5.6x better indirect reasoning vs per-user LoRA. | **Critical change**: separate facts from skills in weights. Facts → Engram/graph. Skills → LoRA. |
| **Self-Evolving Memory Architecture via AutoResearch** | May 2026 | Memory systems that evolve their own retrieval/scoring policies — not just stored content. | Memory architecture itself should adapt, not just content. |

**References:**
- Sleep: arxiv 2606.03979 (Behrouz, Hashemi, Mirrokni — Google)
- TMEM: arxiv 2606.04536 (Ren et al.)
- DCPM: arxiv 2606.09483 (Fei et al.)
- Engram bi-temporal: arxiv 2606.09900
- User as Engram: arxiv 2606.19172 (Li)

### 5.6 Surveys

| Survey | Year | Scope |
|--------|------|-------|
| "A Survey on the Evolution of LLM Agent Memory Mechanisms" | May 2025 | Comprehensive: bridges OS engineering and cognitive science approaches |
| "LLM Agent Memory: Unified Representation-Management Perspective" | Mar 2025 | Taxonomy: tokens vs. intermediate representations vs. parameters |
| "Tiered Memory Architecture and the Retrieval Bottleneck" | May 2025 | Multi-tier memory with RL-based management |
| "Modular Architectures and Strategies" (benchmark) | Apr 2025 | Experimental comparison of memory architectures for agents |

**References:**
- arxiv 2605.06716
- preprints.org 202603.0359
- arxiv 2605.03675
- arxiv 2604.01707

---

## 6. What's Novel in Our Approach (Updated June 2026)

Given the rapid pace of research, our unique positioning has shifted. Here's what's now table-stakes vs. what remains novel:

### No longer novel (others have done it):
- Memory consolidation concept (Sleep paper, DCPM)
- Per-user weight updates (TMEM, User as Engram)
- Dual-process timing (DCPM)
- Schema/concept induction from memory (DCPM System 2)

### Still novel (our unique contribution):
| Component | Why nobody else has it |
|-----------|----------------------|
| **CoT curriculum generation from temporal KG** | Sleep uses RL-dreaming (no ground truth). TMEM uses extraction actions. We use frontier-model CoT conditioned on graph facts — higher quality, controllable, and we proved it works in our BMVC paper. |
| **Graph → Skills pipeline (not graph → facts)** | User as Engram stores facts surgically. We go further: detect PATTERNS in the graph, generate REASONING TRAINING from those patterns, and teach the model HOW TO THINK. Nobody else does this graph-to-reasoning-skill pipeline. |
| **Temporal stratification of SKILLS** (not just facts) | TMEM has one fast-weight layer. We have 4 layers (fast/staging/consolidated/expertise) specifically for reasoning skills with different consolidation rates. |
| **The complete loop running on real personal interactions over time** | Everyone evaluates on benchmarks (LoCoMo, LongMemEval, PersonaMem). Nobody has demonstrated the full system running on actual long-term personal use with measurable expertise development. |

**Our unique contribution in one sentence (updated):** We build the autonomous pipeline that detects reasoning patterns in a temporal knowledge graph and generates CoT-based curriculum to teach those patterns as skills into temporally-stratified LoRA adapters — the only system that uses structured knowledge to generate reasoning-skill training data rather than storing facts or using RL for self-improvement.

---

## 7. Future Directions: Embedding-Space Concept Detection

### 7.1 The Concept Formation Problem

The hardest unsolved piece: how does the system discover emergent higher-order concepts that were never explicitly named?

Current approach (Phase 3): ask GPT to look at graph clusters and identify patterns. Works but is expensive and may miss subtle geometric patterns.

Future approach (Phase 4): track how the model's internal embedding space evolves over time. Detect concept formation geometrically.

### 7.2 How It Would Work

**Traditional knowledge graph:** nodes are words, edges are labeled relationships.

**Embedding-space graph:** nodes are vector clusters (regions in the model's representation space), edges are geometric proximity/connectivity.

Over time, as the user repeatedly discusses related topics, the model's internal representations for those topics converge geometrically. A new cluster forms in embedding space BEFORE anyone names it. This cluster IS the concept — existing mathematically before it exists symbolically.

The system detects: "These 5 entities keep activating nearby regions → emergent concept forming." Then asks a frontier model to name it. Then promotes it to the symbolic graph.

### 7.3 Relevant Research

- "Knowledge Graphs as Structured Memory for Embedding Spaces" (arxiv 2511.14961) — builds graph OVER embedding space using prototype nodes and geometric edges
- "Probing Neural Topology of Large Language Models" (arxiv 2506.01042) — graphs internal neural connections, discovers universal topological properties that predict performance
- "A New Graph Perspective on Neural Networks" (SVR, arxiv 2302.08183) — represents networks as graphs via SVD, enables graph-based analysis

### 7.4 Why Defer This

- Requires solving noise vs. signal in embedding drift (hard research problem)
- No established evaluation methodology
- The simpler LLM-based approach may work 80% as well
- Can be added later as an interpretability/validation layer once the base system works

---

## 8. Technical Implementation Plan

### 8.1 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Experience Store | PostgreSQL | Reliable, queryable, familiar. Stores conversations + metadata. |
| Knowledge Graph | Graphiti (open-source from Zep) | Production-grade temporal KG. Auto entity/relationship extraction. Handles fact invalidation. |
| Base Model | Qwen2.5-3B-Instruct or Qwen3-4B | Small enough for personal adaptation, capable enough for useful responses. Already proven in BMVC work. |
| LoRA Training | PEFT + PyTorch | Standard library. Same pipeline as BMVC paper. |
| Frontier Model | GPT-4.1 (via Azure) | For curriculum generation. Same role as BMVC paper. |
| Evolution Engine | Python async service | Custom. Runs on schedule (nightly or triggered). |
| Serving | vLLM with LoRA adapter switching | Supports loading multiple LoRA adapters efficiently. |
| Evaluation | Custom benchmark + human judgment | Task-specific questions the user writes, measured over time. |

### 8.2 Phase 1: Base System (Months 1-3)

**Goal:** Prove knowledge flows from interactions → graph → weights and the model measurably improves.

**Build:**
1. Set up Graphiti with PostgreSQL backend
2. Create conversation ingestion pipeline (your actual interactions feed into Graphiti)
3. Build Evolution Engine v1 (simple Python script):
   - Query graph for entities with frequency > threshold
   - Query for relationships mentioned > N times
   - Generate CoT training pairs using GPT-4.1 (same technique as BMVC)
   - Train single LoRA adapter on accumulated curriculum
4. Evaluate: test the personalized model vs. base on domain-specific questions

**Deliverable:** Working end-to-end pipeline. Quantified improvement over time.

**Compute budget:** ~$50-100/month (LoRA training + GPT-4.1 API calls for curriculum)

### 8.3 Phase 2: Temporal LoRA Stack (Months 3-5)

**Goal:** Demonstrate that temporal stratification improves knowledge stability and resistance to forgetting.

**Build:**
1. Split the single LoRA into three temporal layers (staging, consolidated, expertise)
2. Implement promotion logic:
   - staging → consolidated: entity has been useful in 5+ interactions AND stable for 2+ weeks
   - consolidated → expertise: knowledge has been stable for 2+ months AND used in 10+ interactions
3. Implement demotion logic:
   - If a fact in consolidated is contradicted 3+ times → weaken and replace
4. Implement validation:
   - After each LoRA update, test on a held-out set of past interactions
   - If performance drops > threshold → rollback the update
5. Implement LoRA merging for inference:
   - Load base + all three temporal adapters
   - vLLM adapter composition or manual merge

**Experiment:**
- Run system for 4 weeks with temporal stack
- Run parallel baseline (single LoRA, same total training data)
- Measure: forgetting resistance (introduce contradictions, see which system handles better)
- Measure: stability (does deep knowledge persist when surface knowledge changes?)

**Deliverable:** Empirical comparison showing temporal stratification > single adapter for personal AI.

### 8.4 Phase 3: Concept Detection via LLM (Months 5-7)

**Goal:** Demonstrate emergent concept discovery from accumulated graph knowledge.

**Build:**
1. Weekly job: extract top-50 most connected entities from graph
2. Cluster them by relationship proximity (graph community detection — Louvain/Leiden algorithm)
3. For each detected cluster, ask GPT-4.1: "These entities frequently co-occur: [list]. What higher-order concept do they represent? Name it and define it in one sentence."
4. If GPT produces a coherent concept:
   - Add it as a new entity in the graph
   - Connect it to all member entities with "is_part_of" relationships
   - Generate curriculum training data for this concept
   - Feed into Evolution Engine for weight consolidation

**Experiment:**
- Seed with 2 months of real interactions
- Run concept detection
- Human evaluation: are the discovered concepts meaningful?
- Compare model with vs. without concept-enhanced training

**Deliverable:** Demonstrated concept formation from interaction patterns.

### 8.5 Phase 4: Embedding-Space Concept Detection (Months 7+)

**Goal:** Replace/augment LLM-based concept detection with geometric detection in embedding space.

**Build:**
1. During inference, extract hidden state embeddings for key entities (using model hooks)
2. Store embeddings weekly with timestamps
3. Track embedding distances between entities over time
4. Detect convergence (entities moving closer) → flag as proto-concept
5. Compare with LLM-detected concepts: do they match?

**This phase is exploratory.** It may produce a standalone paper on geometric concept formation, or it may prove that LLM-based detection is sufficient and the embedding approach adds complexity without proportional benefit.

---

## 9. Evaluation Framework

### 9.1 Metrics

| Metric | What it measures | How to compute |
|--------|-----------------|----------------|
| **Memory Retention** | Can the system recall information from 100+ interactions ago? | Periodically ask questions about old conversations. Measure accuracy over time. |
| **Knowledge Evolution** | Do higher-order concepts emerge? | Count new concepts detected by Evolution Engine. Human-judge their validity. |
| **Personalization Quality** | Does the model's advice improve for THIS specific user? | Blind comparison: personalized model vs. base model on user-written questions. |
| **Learning Efficiency** | Does the model need fewer examples to understand new topics over time? | Track how many interactions it takes to learn new concepts as expertise grows. |
| **Forgetting Resistance** | Can new learning occur without destroying old capabilities? | Introduce new domain knowledge. Test: does old domain knowledge degrade? |
| **Temporal Coherence** | Does the system correctly handle knowledge that changes over time? | Introduce contradictions at different time delays. Test: does the system appropriately update? |

### 9.2 Baselines

1. **Base model (no personalization)** — lower bound
2. **RAG-only (vector store + retrieval)** — current industry standard
3. **Single LoRA (one-shot fine-tuning)** — our BMVC approach applied once
4. **Mem0/simple memory** — key-value fact storage
5. **Our full system** — three layers + Evolution Engine + temporal LoRA

### 9.3 User Study Design

Since this is personal AI, the ultimate evaluation is: **does the user feel the system is getting better over time?**

Weekly survey (1 minute):
- "Did the system understand your needs better this week than last week?" (1-5)
- "Did the system make any responses that showed deep understanding?" (Y/N, examples)
- "Did the system make any errors that showed it forgot something important?" (Y/N, examples)

---

## 10. Publication Strategy

### 10.1 Paper 1: System Paper (Target: ICLR 2027 or AAAI 2027)

**Title:** "Persistent Intelligence: Autonomous Knowledge Consolidation from Interaction to Expertise via Temporal LoRA"

**Contribution:** The full system — three-layer architecture + Evolution Engine + temporal LoRA stack. Demonstrated on N weeks of real interaction with measurable improvement.

**Estimated timeline:** Submit Aug-Sep 2026 (needs 4-5 months of system running + evaluation)

### 10.2 Paper 2: Concept Formation (Target: ACL/EMNLP 2027)

**Title:** "Emergent Concept Formation in Personal Knowledge Graphs: From Interaction Patterns to Model Expertise"

**Contribution:** The concept detection mechanism — both LLM-based and geometric. Evaluation of discovered concepts. Connection between concept detection and effective curriculum generation.

**Estimated timeline:** Submit early 2027

### 10.3 Paper 3: Temporal Stratification Study (Target: NeurIPS 2027 or TMLR)

**Title:** "Temporal Weight Stratification for Continual Personal Adaptation: Staging, Consolidation, and Expertise Layers"

**Contribution:** Ablation study on the temporal LoRA stack. How many layers? What promotion criteria? Comparison with single-adapter baselines, EWC, and other continual learning methods.

**Estimated timeline:** Submit mid 2027

---

## 11. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Catastrophic forgetting in LoRA updates | Medium | High | Validation checks after every update. Rollback mechanism. Keep LoRA rank small. Separate adapters per domain. |
| Evolution Engine generates bad training data | Medium | Medium | Validate curriculum quality before training. Hold-out evaluation. Human review of generated data periodically. |
| Concept detection produces noise/artifacts | High | Low | Start with conservative thresholds. Human validation. The system still works without concept detection — it just has manual concepts. |
| Single-user evaluation isn't generalizable | Medium | Medium for publication | Frame as case study initially. Plan multi-user evaluation for Paper 2. |
| Someone publishes same system before us | Medium | High | Move fast on Phase 1-2. Focus on unique angle (CoT curriculum generation as the bridge mechanism). |
| Graphiti doesn't scale or has limitations | Low | Medium | It's open-source — can fork and extend. Or fall back to Neo4j with custom extraction. |

---

## 12. Immediate Next Steps (This Week)

1. **Install Graphiti locally** (`pip install graphiti-core`). Set up with Neo4j or in-memory mode.
2. **Feed it 10-20 real conversations** from your existing projects (this chat, previous sessions).
3. **Inspect the graph** it builds. Check: are entities sensible? Are relationships accurate? Are temporal facts tracked?
4. **Write the simplest possible Evolution Engine** — a Python script that:
   - Queries graph for top-10 most connected entities
   - Generates 20 training pairs using GPT-4.1 (same CoT technique as BMVC)
   - Trains a LoRA adapter on those 20 pairs
5. **Test:** Ask the personalized model domain questions. Does it outperform base?

If this works → you have proof of concept for the entire pipeline.
If it doesn't → identify which layer failed (bad extraction? bad curriculum? bad training?) and fix.

Total time for this validation: **~1 week with your existing skills.**

---

## 13. Summary

We are building a system where:
- Every interaction adds to the experience store (Layer 1)
- Graphiti extracts entities and relationships into a temporal knowledge graph (Layer 2)
- The Evolution Engine periodically detects patterns, generates CoT-based curriculum, and updates personal LoRA adapters (Layer 3)
- The temporal LoRA stack stratifies knowledge by age (staging → consolidated → expertise)
- Over time, the model needs less retrieval because more knowledge lives in weights
- The system detects emergent concepts and consolidates them autonomously

Nobody has built this complete pipeline. The individual components exist. The integration — and specifically the CoT-based curriculum generation as the bridge from graph to weights — is our novel contribution.

The connection to our BMVC work makes this a natural extension: we proved the mechanism works (CoT → LoRA = better model). Now we make it continuous and autonomous.
