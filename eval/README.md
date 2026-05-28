# MoEGuard Downstream Evaluation Pipeline

8-task downstream eval (BoolQ, WinoGrande, RACE-Middle, MathQA, SWAG,
PIQA, ARC-Easy, OpenBookQA) for an air-gapped GPU cluster.

## Directory layout

```
eval/
├── README.md                       (this file)
├── scripts/
│   ├── download_datasets.py        run on a networked workstation (e.g. Mac)
│   ├── convert_megatron_to_hf.sh   run on the cluster, once per checkpoint
│   └── run_eval.sh                 run on the cluster, the evaluation driver
├── data/                           target of download_datasets.py
│   ├── repos/                      raw HF repo snapshots
│   ├── hf_cache/                   arrow cache (HF_DATASETS_CACHE)
│   └── manifest.json
└── results/                        per-checkpoint lm-eval output
```

## End-to-end workflow

### 1. On your Mac (or any networked host) — one-time

```bash
cd eval/scripts
pip install -U "datasets>=2.18" huggingface_hub
python3 download_datasets.py --out ../data
# ~3-5 GB total. Inspect manifest.json for per-dataset status.
```

Tarball + ship to the cluster:

```bash
cd ..
tar czf evaldata.tar.gz data/
scp evaldata.tar.gz USER@gpu:/mnt/ais-c1/dataset/zds/
ssh USER@gpu '
  cd /mnt/ais-c1/dataset/zds && \
  tar xzf evaldata.tar.gz && \
  mv data evaldata
'
# final layout on the cluster:
#   /mnt/ais-c1/dataset/zds/evaldata/
#       repos/ ...
#       hf_cache/ ...
#       manifest.json
```

### 2. On the GPU cluster — one-time setup

Install lm-evaluation-harness into the offline venv (whatever wheels you
already mirror internally):

```bash
pip install --no-index --find-links /mnt/ais-c1/pip_mirror lm-eval
```

### 3. On the GPU cluster — per checkpoint

```bash
# 3a. Convert Megatron checkpoint -> HF
CKPT_IN=/mnt/ais-c1/dataset/zds/main_exp/5.27/moeguard/ckpt \
CKPT_OUT=/mnt/ais-c1/dataset/zds/eval/models/moeguard-iter10000-hf \
bash eval/scripts/convert_megatron_to_hf.sh

# 3b. Run all 8 tasks (0-shot by default)
MODEL_PATH=/mnt/ais-c1/dataset/zds/eval/models/moeguard-iter10000-hf \
bash eval/scripts/run_eval.sh
```

Results land in `/mnt/ais-c1/dataset/zds/eval/results/<ckpt>_<ts>/` as
`results.json` + `samples_*.jsonl` + `eval.log`.

## Environment knobs (run_eval.sh)

| Variable        | Default                                  | Purpose                          |
|-----------------|------------------------------------------|----------------------------------|
| `MODEL_PATH`    | (required)                               | HF-format checkpoint directory   |
| `EVAL_DATA_ROOT`| `/mnt/ais-c1/dataset/zds/evaldata`       | Where data/ was extracted        |
| `RESULTS_DIR`   | `…/eval/results/<ckpt>_<ts>`             | Output directory                 |
| `TASKS`         | `boolq,winogrande,race,mathqa,swag,piqa,arc_easy,openbookqa` | Comma-separated lm-eval task ids |
| `NUM_FEWSHOT`   | `0`                                      | n-shot setting                   |
| `BATCH_SIZE`    | `auto`                                   | lm-eval auto-batches             |
| `MODEL_BACKEND` | `hf`                                     | Set to `vllm` for vLLM backend   |
| `DTYPE`         | `bfloat16`                               | HF model dtype                   |

## Task → dataset mapping

| lm-eval task id | HF repo                | Config         |
|-----------------|------------------------|----------------|
| `boolq`         | `aps/super_glue`       | `boolq`        |
| `winogrande`    | `allenai/winogrande`   | `winogrande_xl`|
| `race`          | `EleutherAI/race`      | `middle`       |
| `mathqa`        | `allenai/math_qa`      | -              |
| `swag`          | `allenai/swag`         | `regular`      |
| `piqa`          | `ybisk/piqa`           | -              |
| `arc_easy`      | `allenai/ai2_arc`      | `ARC-Easy`     |
| `openbookqa`    | `allenai/openbookqa`   | `main`         |

`download_datasets.py` mirrors every (repo, config) above and also runs
`datasets.load_dataset()` once to materialise the arrow cache, so the
GPU host can load each split with `HF_DATASETS_OFFLINE=1` and no
network access.

## Troubleshooting

- **`OfflineModeIsEnabled` error during lm-eval** — the dataset was not
  warmed in step 1. Re-run `download_datasets.py` without `--skip-warm`
  and re-ship `data/`.
- **`race` falls back to the wrong subset** — pass
  `TASKS="boolq,winogrande,race,..."` and check that lm-eval's `race`
  task defaults to `middle` in your version; otherwise pin
  `TASKS="...,race_middle,..."` if your harness fork exposes it.
- **MathQA loader script needs internet** — `trust_remote_code=True`
  is set both during warmup and at eval time, but ensure
  `~/.cache/huggingface/modules/datasets_modules` is **inside**
  `EVAL_DATA_ROOT` (download_datasets.py sets `HF_HOME` for that).
