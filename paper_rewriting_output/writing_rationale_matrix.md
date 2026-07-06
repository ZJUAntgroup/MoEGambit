# Writing Rationale Matrix

Based on: confirmed_motivation.md, original_logic_map.md, 9 ICSE exemplar papers

## Deep Framework Row (Row 0)

| Field | Content |
|---|---|
| Row ID | R0 |
| Manuscript Unit | Whole paper |
| Current Problem | Paper reads as a systems paper with SE framing bolted on. The SE contribution (recovery contract R1/R2/R3) is present but competes with mechanism details for attention. Title leads with "Fast Failure Recovery" (performance). Abstract climaxes on speedup numbers. |
| Motivation Link | The confirmed motivation says: "recovery is a SE problem requiring an auditable contract." The red thread is: "MoEGambit treats MoE failure recovery as a contract-governed runtime subsystem." |
| Reference/SOTA Pattern | ICSE exemplars (QPG, Rango, PBT-in-Practice) lead with the SE problem, not the performance result. Abstract structure: problem → gap → method → evidence → claim. |
| Target Venue Norm | ICSE values: Novelty (SE artifact, not just faster system), Rigor (paired evaluation, statistical tests), Relevance (to SE community), Verifiability, Presentation. |
| Evidence/Citation Anchor | R1/R2/R3 contract, MAPE-K mapping, Wohlin/Arcuri methodology, $\Phi'(t)$ specification |
| Planned Change | Restructure strategic touchpoints (title, abstract first/last sentence, intro ¶1 opener, conclusion closer) to lead with the SE contribution. Performance numbers become evidence, not the headline. |
| Final Text Check | Does the title say "contract/specification"? Does the abstract open with the SE problem? Does the conclusion return to the SE claim? |

## Per-Unit Rows

| Row | Unit | Problem | Motivation Link | ICSE Pattern | Planned Change |
|-----|------|---------|-----------------|--------------|----------------|
| R1 | Title | "Fast Failure Recovery" leads with performance | Red thread: contract-governed recovery | ICSE titles name the SE artifact or methodology | Change to emphasize recovery contract/specification |
| R2 | Abstract S1-S2 | Opens with "Failure recovery... is a runtime reliability problem" then immediately dives into mechanism | Should open with the SE problem at field level | ICSE: field importance → specific gap | Rewrite: "Large-scale MoE training is a weeks-long software process where failures are routine... yet recovery has no specification" |
| R3 | Abstract S3-S4 | R1/R2/R3 introduced but then mechanism gets equal space | Contract is the contribution, mechanism serves it | ICSE: method description is concise | Shorten mechanism to one clause; expand contract description |
| R4 | Abstract S5 | Climaxes on 20.6% and 36.9× | Numbers are evidence, not contribution | ICSE: results stated briefly, claim is the closer | Move numbers to evidence position; close with SE claim |
| R5 | Keywords | Missing SE terms | Need ICSE-searchable terms | ICSE papers include domain-specific SE keywords | Add "self-adaptive systems", "runtime specification"; remove redundant terms |
| R6 | Intro ¶1 | 7 MoE model names, very dense | Field problem: failures are routine in long-running training | ICSE ¶1: importance + problem (2-3 sentences on context, then pain point) | Cut model list to 3-4; lead with "training as a software process"; keep failure stats |
| R7 | Intro ¶2 | Prior work survey, 3 structural limitations | Gap: all prior work targets dense or save-side | ICSE ¶2: existing solutions + their limitations | Tighten prose; keep 3-limitation structure; add one sentence connecting to SE gap |
| R8 | Intro ¶3 | Very long paragraph mixing gap, SE framing, insight, mechanism details, footnote | Core gap + insight + MoEGambit intro | ICSE: separate gap paragraph from solution paragraph | Split into 2 paragraphs: (a) gap + SE framing, (b) insight + MoEGambit brief |
| R9 | Intro ¶4 (eval summary) | Leads with speedup numbers | Evidence supports the contract | ICSE: "We evaluate... on X benchmarks" | Lead with "We evaluate under SE experimentation standards (paired runs)" |
| R10 | Intro contributions | Good ordering (contract first) | Aligned | ICSE: 3-4 bullet contributions | Light edit: make third bullet emphasize SE methodology more |
| R11 | Intro positioning ¶ | Good two-axis positioning | Aligned | ICSE: positioning paragraph common | Keep as-is |
| R12 | §2.1 Concepts | Well-structured, "SE consequence" sentence is excellent | Aligned | Keep | No change |
| R13 | §2.2 Motivating example | Concrete, effective | Aligned | Keep | No change |
| R14 | §2.2 Staleness risk | Two regimes well-articulated | Aligned | Keep | No change |
| R15 | §3 Related Work: Self-adaptive | Only maps R1/R2/R3 to MAPE-K briefly | This is the ICSE audience's home turf | ICSE: self-adaptive/self-healing is a core SE topic | Expand: cite more ICSE-relevant self-adaptive work (Weyns, Cheng roadmap); explain how MoEGambit extends the vocabulary |
| R16 | §3 Related Work: Save-side | Good structure | Aligned | Keep | Light edit |
| R17 | §3 Related Work: Topology | Good structure | Aligned | Keep | Light edit |
| R18 | §3 Related Work: Peer-based | Good FlashRecovery comparison | Aligned | Keep | No change |
| R19 | §3 Related Work: MoE systems | "Freshness blind spot" is excellent | Aligned | Keep | No change |
| R20 | §3 Related Work: MoC-System | Good positioning | Aligned | Keep | No change |
| R21 | §4 Problem Statement | Well-structured Input/Output/Goals/Assumptions/Requirements | Aligned | Keep | Light edit for conciseness |
| R22 | §5.1 Overview | Clear 5-component description | Contract before mechanism | Keep | No change |
| R23 | §5.2 R1 Safe-point | Concise | Aligned | Keep | No change |
| R24 | §5.3 R2 Policy | Well-formalized | Aligned | Keep | No change |
| R25 | §5.4 Mechanism | Appropriately detailed | Mechanism serves contract | Keep | No change |
| R26 | §5.5 Two-phase | Clear | Aligned | Keep | No change |
| R27 | §5.6 R3 State machine | Good | Aligned | Keep | No change |
| R28 | §5.7 Implementation | Good detail level | Aligned | Keep | No change |
| R29 | §6 Eval intro + RQ list | RQs framed as performance questions | RQs should validate the contract | ICSE: RQs tied to contributions | Rewrite RQ descriptions to link each to the contract |
| R30 | §6 Setup | Thorough | Aligned | Keep | No change |
| R31 | §6 RQ1 text | Good factorial analysis | Mechanism decomposition | Keep | Add one sentence linking to contract |
| R32 | §6 RQ2 text | Good paired-run analysis | Quality contract validation | Keep | Add opening sentence: "RQ2 validates the quality guarantee of R2" |
| R33 | §6 RQ3 text | Good cliff validation | $\Phi_{\max}$ calibration | Keep | Add opening sentence linking to R2 specification |
| R34 | §6 RQ4 text | Good overhead analysis | Practical deployability | Keep | No change |
| R35 | §6 RQ5 text | Good scalability | Generalization | Keep | No change |
| R36 | §6 RQ6 text | Good cross-architecture | Generalization | Keep | No change |
| R37 | §6 Threats | Comprehensive, well-organized | Aligned | Keep | No change |
| R38 | §7 Discussion: Lessons | Good SE lessons | "Recovery must be specified, not just fast" | Keep | Light edit: strengthen SE framing |
| R39 | §7 Discussion: Implications | Good | Aligned | Keep | No change |
| R40 | §7 Discussion: Limitations | Honest, well-structured | Aligned | Keep | No change |
| R41 | §7 Discussion: Future work | Concrete | Aligned | Keep | No change |
| R42 | §8 Conclusion | Returns to SE claim | "Recovery should be engineered as a software reliability artifact" | ICSE: conclusion returns to the field-level claim | Light edit: strengthen the methodological claim in the last sentence |

## Summary: Units to REWRITE vs KEEP

**REWRITE (high priority):**
- R1: Title
- R2-R4: Abstract (restructure)
- R5: Keywords
- R6: Intro ¶1 (shorten, SE-lead)
- R8: Intro ¶3 (split, SE-lead)
- R9: Intro ¶4 (reframe)
- R15: Related Work self-adaptive paragraph
- R29: Evaluation RQ list

**KEEP + light edit:**
- R7, R10, R16-R17, R21, R31-R33, R38, R42

**KEEP as-is:**
- R11-R14, R18-R20, R22-R28, R30, R34-R37, R39-R41
