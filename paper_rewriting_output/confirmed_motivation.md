# Confirmed Motivation

| Field | Content |
|---|---|
| Source | Derived from existing draft + ICSE venue alignment |
| Confirmed motivation statement | Runtime failure recovery in distributed MoE training is a software engineering problem that requires an explicit, auditable recovery contract—not just a faster mechanism—because the heterogeneous state structure of MoE creates a quality-risk dimension (expert staleness) that no existing system specifies, monitors, or bounds at runtime. |
| One-sentence red thread | MoEGambit treats MoE failure recovery as a contract-governed runtime subsystem whose correctness is specified by a staleness density bound, realized by state-aware hybrid repair, and verified through structured audit logs—shifting the operational model from "restart and hope" to "recover and verify." |
| Field problem | Large-scale MoE training suffers routine failures (every ~45 min on 16K GPUs), and checkpoint restart—the dominant recovery mechanism—treats the entire distributed state as monolithic, wasting recovery time on state that is already live in memory on healthy peers. |
| Specific gap | No existing fault-tolerance system for MoE training (1) exploits the heterogeneous state structure on the recovery side, (2) provides a runtime-checkable quality contract for partial recovery, or (3) makes recovery decisions auditable. The only MoE-specific system (MoC-System) optimizes the save side but leaves the recovery path unchanged. |
| Design response | MoEGambit: a recovery-side framework with three SE artifacts (R1 safe-point, R2 staleness density Φ'(t), R3 reintegration state machine), realized by MoE-aware hybrid restore + two-phase protocol. |
| Main evidence available | 2×2 factorial (RQ1), paired NoFault runs (RQ2), 3×4 burst sweep with cliff validation (RQ3), overhead measurement (RQ4), 64→128 GPU scalability (RQ5), DeepSeek-V2-Lite cross-architecture (RQ6), 8 downstream tasks, Wilcoxon + Cliff's delta. |
| Target-venue fit | ICSE values SE methodology: contracts, specifications, runtime monitoring, auditability, self-adaptive systems framing. The paper's R1/R2/R3 structure maps directly to MAPE-K. The paired-run evaluation follows Wohlin/Arcuri SE experimentation standards. |
| Prioritized claims | (1) Recovery contract (R1/R2/R3) as the primary SE contribution; (2) Staleness density Φ'(t) as a novel runtime-checkable specification with empirical cliff; (3) State-aware hybrid repair as the mechanism that realizes the contract; (4) Paired evaluation demonstrating no quality degradation. |
| Claims to avoid | (1) Do not overclaim systems-level novelty (the hybrid restore mechanism is conceptually straightforward); (2) Do not position as a general fault-tolerance framework (it is MoE-specific); (3) Do not claim the 36.9× speedup without immediately clarifying it includes replay elimination. |
| Secondary motivations or boundaries | The two-phase protocol is a secondary mechanism contribution. Scalability and cross-architecture generalization are validation, not primary contributions. |
| Surface language priorities | Lead with "recovery contract," "runtime specification," "auditable recovery," "self-adaptive," "software engineering artifact" in strategic positions. Avoid leading with "speedup" or "performance" in topic sentences—these are evidence, not the contribution. |

## Section Consequences

| Section | What This Motivation Requires | What It Should Avoid |
|---|---|---|
| Abstract | Lead with the SE problem (recovery as a runtime reliability problem), then the contract (R1/R2/R3), then the mechanism, then the evidence. The 20.6% and 36.9× are evidence, not the contribution. | Leading with speedup numbers. Making it sound like a systems paper. |
| Introduction | Build necessity through the SE lens: (1) failures are routine, (2) recovery is a runtime subsystem, (3) current recovery has no specification, (4) MoE state heterogeneity creates a new quality-risk dimension, (5) MoEGambit provides the contract + mechanism. | Spending too much space on MoE architecture details. Listing systems without connecting to the SE gap. |
| Background | Frame MoE training as a stateful distributed program with heterogeneous state classes. The motivating example should show why the current recovery flow is structurally wasteful AND why partial recovery introduces a quality risk that needs a specification. | Pure systems-level description without SE framing. |
| Related Work | Position against self-adaptive systems first (ICSE audience), then against systems work. Show that MoEGambit occupies a unique quadrant: recovery-side + contract. | Treating Related Work as a literature survey rather than a positioning argument. |
| MoEGambit (§5) | Present R1/R2/R3 as the primary structure. The mechanism (§5.4-5.5) serves the contract, not the other way around. | Letting the mechanism dominate the section. The contract should be the spine. |
| Evaluation | Frame RQs around the contract's claims: Does the mechanism reduce cost (RQ1)? Does the contract preserve quality (RQ2)? Does the specification correctly identify the cliff (RQ3)? Is the overhead negligible (RQ4)? Does it scale (RQ5)? Does it generalize (RQ6)? | Presenting RQs as pure performance benchmarks without connecting to the contract. |
| Discussion | Lessons should be SE lessons, not systems lessons. The key insight is that recovery must be specified, not just fast. | Generic "our system is faster" discussion. |
| Conclusion | Return to the SE claim: recovery should be engineered as a software reliability artifact. | Ending with speedup numbers. |
