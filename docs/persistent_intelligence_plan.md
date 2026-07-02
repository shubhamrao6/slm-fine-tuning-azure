# Persistent Intelligence Systems: Research Plan

## 1. What I Understand You Want to Build

You want to build a system where an AI doesn't just answer questions — it **develops expertise over time** through continued interaction with a user. The system should:

- Remember facts (who you are, what you're working on)
- Understand relationships (how your projects connect, what tools you prefer)
- Form concepts (recognizing patterns across your work that weren't explicitly stated)
- Develop intuition (becoming genuinely better at helping you specifically, not just remembering what you said)

The key insight from your article: **memory is not intelligence. The system must not just remember — it must learn.** And that learning must happen across three layers that mirror human cognition:

1. **Experience** (raw interactions, stored verbatim)
2. **Understanding** (structured knowledge graph of entities, relationships, concepts)
3. **Expertise** (internalized patterns encoded in model weights)

With an **Evolution Engine** that asynchronously promotes knowledge up the stack — analogous to how human sleep consolidates daily experiences into long-term understanding.

The connection to your BMVC work is direct: you've already demonstrated that a small model can absorb frontier-model reasoning into its weights via LoRA. The question is now: can you make this happen **continuously and autonomously** rather than as a one-shot fine-tuning exercise?

---

## 2. What Currently Exists in the Industry

### Layer 1: Experience Storage (solved)

This is effectively solved. Every chat platform stores conversation history. Vector databases (Pinecone, Weaviate, Qdrant) store embeddings. The infrastructure exists.

**What's available:**
- Vector stores for semantic retrieval
- Conversation logs in any database
- Document ingestion pipelines

**What's missing:** Nothing at this layer — it's commodity infrastructure.

### Layer 2: Knowledge Graph Memory (actively being built)

This is where most of the 2025 research energy is focused.

| System | What it does | Maturity |
|--------|-------------|----------|
| **Zep/Graphiti** | Temporal knowledge graph from conversations. Auto-extracts entities, relationships, facts with time validity. Invalidates stale facts. | Production-ready, commercial product |
| **AriGraph** | Episodic + semantic graph built during agent exploration. Associative retrieval. | Research prototype (2024) |
| **SAGE** | Self-evolving graph with writer-reader architecture. Graph Foundation Model for retrieval. | Research (May 2025) |
| **Mem0** | Simple key-value memory with LLM-based extraction. Stores user preferences and facts. | Production, simpler than Zep |

**What's available:** Zep/Graphiti gives you a production-grade temporal knowledge graph that auto-extracts entities and relationships from conversations. This is ready to use today.

**What's missing:** Concept formation. All existing systems extract explicit entities and relationships. None of them detect emergent higher-order concepts ("this user is building a persistent intelligence system" from scattered conversations about graphs, LoRA, memory, and SEAL).

### Layer 3: Weight Memory (early research, fragmented)

This is the least developed layer and where your unique expertise applies.

| System | What it does | Maturity |
|--------|-------------|----------|
| **SEAL** (MIT, 2025) | Model generates its own fine-tuning data + optimization instructions. RL loop for improvement. | Research, text-only, not personalization-focused |
| **Weight consolidation paper** (May 2025) | Compares per-user LoRA updates against context-window workarounds | Research, comparison study |
| **Per-user LoRA** (various) | Composable adapters: Base + Personal + Domain | Conceptually proposed, no end-to-end system |
| **Titans/MIRAS** (Google, 2025) | Neural memory that updates weights at inference time using surprise metric | Research, architectural innovation |

**What's available:** LoRA infrastructure is mature (PEFT library, vLLM serving with adapter switching). The mechanics of weight adaptation are solved.

**What's missing:** The **trigger mechanism** — deciding WHEN to update weights, WHAT to train on, and HOW to generate appropriate training data from accumulated graph knowledge. This is the Evolution Engine. Nobody has built it.

### The Evolution Engine (does not exist)

This is the gap. Everyone has:
- Storage ✓
- Graphs (getting there) ✓
- Weight adaptation mechanics ✓

Nobody has the autonomous process that:
1. Monitors the knowledge graph for mature patterns
2. Decides what's ready to become "expertise"
3. Generates appropriate training data (your SEAL/CoT technique)
4. Applies weight updates without catastrophic forgetting
5. Validates the update didn't break anything

**This is your research contribution.**

---

## 3. What to Use From the Industry

### Use directly (don't rebuild):

| Component | Use | Why |
|-----------|-----|-----|
| **Zep/Graphiti** | Knowledge graph layer | Production-grade, temporal, handles entity extraction + relationship tracking + fact invalidation. Open-source (Graphiti). Saves you 6+ months of graph engineering. |
| **PEFT/LoRA** | Weight adaptation mechanics | You already know this. Standard library for adapter training. |
| **Qwen2.5-VL or similar small model** | Base model | You've proven it works. Small enough for personal adaptation. |
| **Any LLM (GPT-4.1, Claude)** | Frontier model for CoT generation | Same role as in your BMVC paper — generates training data. |
| **PostgreSQL/SQLite** | Experience store | Simple conversation logging. Nothing fancy needed. |

### Build yourself (the novel parts):

| Component | Build | Why nobody has it |
|-----------|-------|-------------------|
| **Promotion function** | Decides what moves from graph → weights | Requires domain judgment: importance × frequency × longevity × utility. No standard algorithm exists. |
| **Curriculum generator** | Creates training data from graph knowledge | This is your SEAL-adapted CoT technique. Generate "what should the model understand?" from graph patterns. |
| **Consolidation scheduler** | When to trigger weight updates | Could be time-based (nightly), threshold-based (enough new concepts), or event-based (user explicitly teaches). |
| **Concept detector** | Identifies emergent abstractions in graph | The hardest unsolved problem. Cluster analysis on graph neighborhoods? LLM-based pattern detection? |
| **Forgetting guard** | Ensures new LoRA updates don't destroy old capabilities | EWC (Elastic Weight Consolidation), or simply maintaining separate LoRA adapters per domain and merging. |

---

## 4. How It Comes Together: The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERACTION                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              EXPERIENCE STORE (PostgreSQL)                    │
│  Every message, document, action — stored verbatim           │
│  Immutable log. Append-only. Cheap.                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ (Real-time: entity extraction via Graphiti)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│           KNOWLEDGE GRAPH (Zep/Graphiti)                      │
│                                                              │
│  Entities: [User, Projects, Tools, People, Concepts]         │
│  Relations: [works_on, uses, prefers, related_to]            │
│  Temporal: facts have valid_from, valid_to, confidence       │
│  Auto-updated on every interaction                           │
│                                                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ (Async: Evolution Engine — runs periodically)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              EVOLUTION ENGINE (your novel contribution)       │
│                                                              │
│  1. DETECT: Find mature patterns in graph                    │
│     - High-frequency entities/relationships                  │
│     - Stable facts (not recently contradicted)               │
│     - Clusters that suggest emerging concepts                │
│                                                              │
│  2. DECIDE: Promotion function                               │
│     - Score: importance × frequency × stability × utility    │
│     - Only promote knowledge above threshold                 │
│                                                              │
│  3. GENERATE: Create training data (your CoT technique)      │
│     - Take graph patterns                                    │
│     - Use frontier model to generate reasoning-rich          │
│       training pairs that teach these patterns               │
│     - Answer-condition on the graph's ground truth           │
│                                                              │
│  4. UPDATE: Apply LoRA update to personal adapter            │
│     - Micro-fine-tune on generated curriculum                │
│     - Validate: test on held-out interactions                │
│     - Rollback if quality drops                              │
│                                                              │
│  5. PRUNE: Remove consolidated knowledge from active graph   │
│     - Facts now "in weights" don't need graph retrieval      │
│     - Reduces retrieval noise over time                      │
│                                                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│           ADAPTIVE MODEL (Base + Personal LoRA)              │
│                                                              │
│  Base: Qwen2.5-3B or similar (frozen, general capability)    │
│  Personal LoRA: Updated by Evolution Engine                  │
│  Serves inference with personalized expertise                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Context Compilation (at inference time):

When the user sends a message, the system doesn't just retrieve — it **compiles context**:

```
1. Query knowledge graph for relevant entities/relationships
2. Pull recent relevant experiences (last few interactions on this topic)
3. The personal LoRA already "knows" consolidated patterns (no retrieval needed)
4. Assemble minimal, high-signal context package
5. Inference with personalized model + compiled context
```

This means: over time, less needs to be retrieved because more lives in weights. Context windows get shorter and more focused as the system matures.

---

## 5. The Research Roadmap

### Phase 1: Prove the concept on a single user (you) — 2-3 months

**Goal:** Build the simplest possible end-to-end pipeline. One user. One domain. Demonstrate measurable improvement over time.

**Setup:**
- Experience store: SQLite (your interactions with the system)
- Graph: Graphiti (open-source from Zep team)
- Model: Qwen2.5-3B-Instruct with LoRA
- Frontier: GPT-4.1 for curriculum generation
- Evolution Engine: Python script, runs nightly

**Experiment:**
- Use the system for 2-4 weeks (ask it questions about your research domain)
- Measure: Does it get better at answering domain-specific questions over time?
- Compare: Personalized model vs. base model vs. RAG-only baseline
- Metric: Accuracy on held-out questions you write yourself

**What you demonstrate:** The pipeline works. Knowledge flows from interactions → graph → weights. The model improves.

### Phase 2: The concept formation problem — 2-3 months

**Goal:** Solve the hardest piece — detecting emergent concepts.

**Approach options:**
- Graph community detection (Louvain/Leiden) on entity clusters
- LLM-based: periodically ask GPT-4.1 to "look at these 50 entities and relationships — what higher-order concepts do they represent?"
- Frequency-based: entities that always co-occur but lack an explicit parent concept

**Experiment:**
- Seed with a month of real interactions
- Run concept detection
- Evaluate: Do the discovered concepts make sense? Would a human name them the same way?
- Compare: LLM-detected concepts vs. graph-algorithm-detected concepts

### Phase 3: Catastrophic forgetting and multi-domain — 2-3 months

**Goal:** Show the system doesn't break when learning new domains.

**Approach:**
- Separate LoRA adapters per domain (cloud architecture, ML research, personal preferences)
- LoRA merging at inference time
- Or: EWC-regularized single adapter

**Experiment:**
- Train on domain A (cloud architecture) for 2 weeks
- Train on domain B (ML research) for 2 weeks
- Measure: Does domain A performance degrade?
- Compare: Merged LoRA vs. switched LoRA vs. single LoRA with EWC

### Phase 4: Paper and system release — 1-2 months

**Target venues:**
- NeurIPS 2026 (deadline ~May 2026 — already past for this year)
- ICLR 2027 (deadline likely Sep 2026)
- AAAI 2027 (deadline likely Aug 2026)
- ACL 2027 or EMNLP 2026 (if framed as NLP contribution)
- Or: A journal (TMLR, JMLR) with no deadline pressure

**Paper framing:**
"We present [SystemName], a three-tier persistent intelligence architecture that continuously consolidates interaction knowledge into personal model expertise via answer-conditioned curriculum generation. We demonstrate that over N weeks of interaction, the personalized model improves X% on domain-specific tasks compared to RAG-only and base model baselines."

---

## 6. What Makes This Publishable (vs. what's already published)

| Existing work | What it does | What it doesn't do |
|---------------|-------------|-------------------|
| Zep/Graphiti | Graph from conversations | No weight updates |
| SEAL | Self-generated training data | No graph, no personalization, no external knowledge |
| Titans | Neural memory at inference | No persistent cross-session learning |
| SCM | Sleep consolidation | No weight-level expertise |
| CMA | Defines the architecture class | No implementation |
| Weight consolidation paper | Compares approaches | No curriculum generation |

**Your unique combination:**
- Graph memory (Graphiti) **+** answer-conditioned CoT curriculum generation (your BMVC technique) **+** personal LoRA adaptation (SEAL-inspired)
- Running as an autonomous loop (Evolution Engine)
- Demonstrated on real interactions over time

Nobody has connected these three pieces into a working system. The individual components exist. The integration does not.

---

## 7. Concrete First Step

Start with this minimal prototype:

1. Set up Graphiti locally (it's open-source Python: `pip install graphiti-core`)
2. Feed it 20-30 of your actual conversations (from this project or others)
3. Inspect the graph it builds — entities, relationships, temporal facts
4. Write a "promotion scorer" that identifies the top 10 most stable, frequent patterns
5. Use GPT-4.1 to generate 50 training pairs that teach those patterns (same technique as your BMVC paper, but from graph knowledge instead of image labels)
6. LoRA fine-tune Qwen on those 50 pairs
7. Test: Does the personalized model answer domain questions better than base?

If yes → you have a proof of concept for the full pipeline.
If no → debug which layer is failing (bad graph extraction? bad curriculum? bad training?).

This is ~1 week of work with your existing skills.
