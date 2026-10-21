# Benchmark visualizations

These assets illustrate system design and measured benchmark results. Inputs,
measurement definitions, and redraw commands are in
[examples/benchmarks](../../../examples/benchmarks/README.md).
[figure_inventory.json](figure_inventory.json) records asset checksums.

| Visualization | Files |
| --- | --- |
| Recovery architecture | [PDF](moegambit_runtime_architecture.pdf); [PNG](../moegambit-runtime-architecture.png) |
| EDP state placement | [PDF](edp_distribution.pdf) / [PNG](edp_distribution.png) |
| Ten-fault training loss | [PDF](train_loss.pdf) / [PNG](train_loss.png) |
| Training stages and affected ranks | [PDF](quality_checkpoint_study.pdf) / [PNG](quality_checkpoint_study.png) |
| Expert counts and routing load | [PDF](quality_full_state_500.pdf) / [PNG](quality_full_state_500.png) |
| Completion and repeated faults | [PDF](quality_full_state_terminal.pdf) / [PNG](quality_full_state_terminal.png) |
| GPU scaling and parallel layouts | [PDF](recovery_scaling.pdf) / [PNG](recovery_scaling.png) |
| GQA and MLA configurations | [PDF](quality_architecture_transfer.pdf) / [PNG](quality_architecture_transfer.png) |
| Eight-task accuracy | [PDF](downstream_accuracy.pdf) / [PNG](downstream_accuracy.png) |
| Web update pipeline | [PDF](web_update_pipeline.pdf) / [PNG](web_update_pipeline.png) / [SVG](web_update_pipeline.svg) |
| FiQA reranking quality | [PDF](fiqa_reranking_quality.pdf) / [PNG](fiqa_reranking_quality.png) / [SVG](fiqa_reranking_quality.svg) |

The Web update and FiQA experiments use cold starts, checkpoint-simulated full
expert-state loss, and R2 admission before Hybrid restoration. Update time includes the first real HTTP
rerank response; query intervals describe retrieval evaluation uncertainty.
The active exports include the author-confirmed protocol correction, with
original export bytes retained in the benchmark data archive.

Architecture icons use Lucide assets under the
[license notice](../licenses/Lucide-LICENSE).
