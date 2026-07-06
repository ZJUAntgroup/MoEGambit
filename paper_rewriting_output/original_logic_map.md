# Original Logic Map

## Unit-by-Unit Analysis

### U1: Title
- **Current text**: "MoEGambit: Trading Bounded Expert Staleness for Fast Failure Recovery in Sparse MoE Training"
- **Role**: Conveys the core trade-off (staleness ↔ speed)
- **Problem**: Reads as a systems paper title. Missing SE framing (contract, specification, auditability). "Fast Failure Recovery" leads with performance.
- **Decision**: REWRITE — Lead with the SE artifact (recovery contract), keep the trade-off but frame it as a specification problem.

### U2: Abstract (lines 33-38)
- **Current role**: States the problem (checkpoint restart is monolithic), introduces MoEGambit with R1/R2/R3, gives mechanism, reports numbers.
- **Motivation link**: Partially aligned — mentions "runtime reliability problem" and "auditable runtime recovery contract" but then pivots to mechanism details and speedup numbers.
- **Problems**: (1) The 20.6% and 36.9× appear as the climax — they should be evidence, not the contribution. (2) "MoE-aware hybrid restore" gets more words than the contract. (3) Missing: the SE insight that recovery must be *specified* against the post-recovery trajectory. (4) Too long — could be tighter.
- **Decision**: REWRITE — Restructure: SE problem → gap (no specification) → contract (R1/R2/R3) → mechanism (brief) → evidence → SE claim.

### U3: Keywords (lines 40-42)
- **Current**: "Mixture of Experts, fault tolerance, distributed training, checkpoint recovery, failure recovery"
- **Problem**: Missing SE-specific keywords that ICSE reviewers search for.
- **Decision**: REWRITE — Add "self-adaptive systems", "runtime specification", "recovery contract". Remove redundant "checkpoint recovery" / "failure recovery".

### U4: Introduction ¶1 (lines 44-47, the MoE landscape + failure frequency)
- **Current role**: Establishes MoE as widely adopted, then pivots to failure frequency (Llama-3, MegaScale, OPT).
- **Motivation link**: Good — establishes the field problem (failures are routine).
- **Problems**: (1) The MoE model listing is too long (7 models) — ICSE reviewers don't need this. (2) The paragraph is very dense. (3) Could start with the SE framing earlier.
- **Decision**: REWRITE — Shorten model listing to 3-4 exemplars. Lead with "large-scale training is a long-running software process" framing. Keep failure statistics.

### U5: Introduction ¶2 (lines 49, prior fault-tolerance systems)
- **Current role**: Surveys dense-model fault-tolerance systems, identifies 3 structural limitations for MoE, introduces MoC-System.
- **Motivation link**: Good — establishes the gap (all prior work targets dense or save-side).
- **Problems**: (1) Very long paragraph. (2) The three limitations (i-iii) are well-structured but could be more concise. (3) MoC-System positioning is good.
- **Decision**: KEEP with light edits — Tighten prose, maintain the three-limitation structure.

### U6: Introduction ¶3 (line 51, the gap + insight + MoEGambit intro)
- **Current role**: States the gap, frames as SE problem, introduces the insight and MoEGambit.
- **Motivation link**: Strong — "We frame this as a software engineering problem" is the key sentence.
- **Problems**: (1) Very long paragraph — mixes gap statement, SE framing, insight, mechanism details, self-adaptive systems reference, and the footnote. (2) The mechanism details (path P, path C, two-phase) are too detailed for the intro. (3) The SE framing is buried mid-paragraph.
- **Decision**: REWRITE — Split into 2-3 shorter paragraphs. Lead with the SE gap. Move mechanism details to a brief summary.

### U7: Introduction ¶4 (line 53, evaluation summary)
- **Current role**: Summarizes evaluation results.
- **Motivation link**: Good — mentions paired runs and Wohlin/Arcuri.
- **Problems**: (1) Leads with speedup numbers. (2) The paired-run methodology mention is good but could be more prominent.
- **Decision**: REWRITE — Lead with "we evaluate under SE experimentation standards" then give numbers as evidence.

### U8: Introduction contributions list (lines 55-60)
- **Current role**: Three contributions: (1) Runtime recovery contract, (2) MoE-aware hybrid repair, (3) Paired benchmark.
- **Motivation link**: Strong — contract is first, mechanism is second.
- **Problems**: (1) Good ordering. (2) Could be slightly more concise. (3) The third contribution could emphasize the SE methodology more.
- **Decision**: KEEP with light edits.

### U9: Introduction positioning paragraph (lines 62)
- **Current role**: Two-axis positioning (save-side vs recovery-side, dense vs heterogeneous MoE).
- **Motivation link**: Good — positions MoEGambit in the previously empty quadrant.
- **Problems**: (1) Good content. (2) Could be slightly tighter.
- **Decision**: KEEP.

### U10: Background §2.1 (lines 69-78, MoE as heterogeneous-state program)
- **Current role**: Defines MoE training state classes (replicated, sharded, metadata).
- **Motivation link**: Good — establishes the structural heterogeneity that enables hybrid recovery.
- **Problems**: (1) Well-structured. (2) The "software-engineering consequence" sentence at the end is excellent.
- **Decision**: KEEP.

### U11: Background §2.2 (lines 100-114, checkpoint restart + hidden risk)
- **Current role**: Motivating example + formal analysis of staleness risk.
- **Motivation link**: Strong — shows both the opportunity (structural waste) and the risk (staleness).
- **Problems**: (1) The motivating example is concrete and effective. (2) The two regimes of risk are well-articulated. (3) The final sentence connecting to the policy is good.
- **Decision**: KEEP.

### U12: Related Work (lines 116-129)
- **Current role**: 6 subsections covering self-adaptive systems, save-side, topology adaptation, peer-based dense, MoE systems, MoC-System.
- **Motivation link**: Good — self-adaptive systems is first (ICSE audience), MoC-System positioning is clear.
- **Problems**: (1) Self-adaptive systems paragraph could be stronger — currently just maps R1/R2/R3 to MAPE-K. Could cite more ICSE-relevant self-adaptive work. (2) The "freshness blind spot" paragraph is excellent. (3) Overall well-structured.
- **Decision**: REWRITE self-adaptive paragraph to be stronger. KEEP rest with light edits.

### U13: Problem Statement §4 (lines 131-148)
- **Current role**: Input/Output/Design Goals/Assumptions/Requirements.
- **Motivation link**: Strong — R1/R2/R3 are clearly derived from G1/G2/G3.
- **Problems**: (1) Well-structured. (2) Could be slightly more concise. (3) The MAPE-K mapping in the last paragraph is good.
- **Decision**: KEEP with light edits.

### U14: MoEGambit §5 (lines 149-282)
- **Current role**: Overview → R1 → R2 → Mechanism → Two-phase → R3 → Implementation.
- **Motivation link**: Good — contract (R1/R2/R3) comes before mechanism.
- **Problems**: (1) The overview is clear. (2) R1 is concise. (3) R2 is well-formalized. (4) The mechanism section is appropriately detailed. (5) Implementation section is good.
- **Decision**: KEEP with light edits — ensure contract language dominates topic sentences.

### U15: Evaluation §6 (lines 286-533)
- **Current role**: Setup → Compared Systems → Policy Parameters → RQ1-RQ6 → Threats.
- **Motivation link**: Partially aligned — RQs are framed as performance questions, not contract-validation questions.
- **Problems**: (1) RQ framing could be more contract-linked. (2) The evaluation is thorough and well-structured. (3) Threats section is comprehensive.
- **Decision**: REWRITE RQ framing sentences to link each RQ to the contract. KEEP data and analysis.

### U16: Discussion §7 (lines 535-546)
- **Current role**: Lessons → Implications → Limitations → Calibration cost → Future work.
- **Motivation link**: Good — "recovery must be specified, not just fast" is the key lesson.
- **Problems**: (1) Well-structured. (2) Limitations are honest. (3) Future work is concrete.
- **Decision**: KEEP with light edits.

### U17: Conclusion §8 (lines 548-551)
- **Current role**: Returns to the SE claim, summarizes results.
- **Motivation link**: Good — "recovery should be engineered as a software reliability artifact."
- **Problems**: (1) Good structure. (2) Could end more strongly with the methodological claim.
- **Decision**: KEEP with light edits.

## Summary of Decisions

| Unit | Section | Decision | Priority |
|------|---------|----------|----------|
| U1 | Title | REWRITE | High |
| U2 | Abstract | REWRITE | High |
| U3 | Keywords | REWRITE | Medium |
| U4 | Intro ¶1 | REWRITE | High |
| U5 | Intro ¶2 | KEEP+edit | Medium |
| U6 | Intro ¶3 | REWRITE | High |
| U7 | Intro ¶4 | REWRITE | Medium |
| U8 | Intro contributions | KEEP+edit | Low |
| U9 | Intro positioning | KEEP | Low |
| U10 | Background §2.1 | KEEP | Low |
| U11 | Background §2.2 | KEEP | Low |
| U12 | Related Work | REWRITE (self-adaptive) + KEEP | Medium |
| U13 | Problem Statement | KEEP+edit | Low |
| U14 | MoEGambit §5 | KEEP+edit | Low |
| U15 | Evaluation §6 | REWRITE (RQ framing) | Medium |
| U16 | Discussion | KEEP+edit | Low |
| U17 | Conclusion | KEEP+edit | Low |
