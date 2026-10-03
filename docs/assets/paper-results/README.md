# Figures displayed in the READMEs

The main READMEs display the current paper PDF's seven original numbered figures.
Their source content streams were verified against the PDF's embedded figures.
[figure_inventory.json](figure_inventory.json) records the correspondence.

| Paper figure | Asset |
| --- | --- |
| 1 · Recovery architecture | [PDF](moegambit_runtime_architecture.pdf); [PNG at README top](../moegambit-runtime-architecture.png) |
| 2 · Ten-fault training loss | [PDF](train_loss.pdf) / [PNG](train_loss.png) |
| 3 · Training stages and rank counts | [PDF](quality_checkpoint_study.pdf) / [PNG](quality_checkpoint_study.png) |
| 4 · 64/128-expert full-state quality | [PDF](quality_full_state_500.pdf) / [PNG](quality_full_state_500.png) |
| 5 · Completion and repeated faults | [PDF](quality_full_state_terminal.pdf) / [PNG](quality_full_state_terminal.png) |
| 6 · GPU scaling and parallel layouts | [PDF](recovery_scaling.pdf) / [PNG](recovery_scaling.png) |
| 7 · GQA / MLA architecture comparison | [PDF](quality_architecture_transfer.pdf) / [PNG](quality_architecture_transfer.png) |

Two separately requested additions are the original [EDP distribution schematic](edp_distribution.pdf)
and the [eight-task plot](downstream_accuracy.pdf) drawn from paper Table 6.
Other table-derived conclusion charts are withdrawn. Data and redraw instructions
are in [examples/paper_results](../../../examples/paper_results/README.md).

The architecture retains its original five-panel organization. Checkpoint, guard,
replay and resume icons use Lucide assets under their
[license notice](../licenses/Lucide-LICENSE).
