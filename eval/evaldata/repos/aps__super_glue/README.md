---
annotations_creators:
- expert-generated
language_creators:
- other
language:
- en
license:
- other
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- extended|other
task_categories:
- text-classification
- token-classification
- question-answering
task_ids:
- natural-language-inference
- word-sense-disambiguation
- coreference-resolution
- extractive-qa
paperswithcode_id: superglue
pretty_name: SuperGLUE
tags:
- superglue
- NLU
- natural language understanding
dataset_info:
- config_name: axb
  features:
  - name: sentence1
    dtype: string
  - name: sentence2
    dtype: string
  - name: idx
    dtype: int32
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': not_entailment
  splits:
  - name: test
    num_bytes: 237694
    num_examples: 1104
  download_size: 80924
  dataset_size: 237694
- config_name: axg
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: idx
    dtype: int32
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': not_entailment
  splits:
  - name: test
    num_bytes: 53348
    num_examples: 356
  download_size: 14284
  dataset_size: 53348
- config_name: boolq
  features:
  - name: question
    dtype: string
  - name: passage
    dtype: string
  - name: idx
    dtype: int32
  - name: label
    dtype:
      class_label:
        names:
        - 'False'
        - 'True'
  splits:
  - name: train
    num_bytes: 6173303
    num_examples: 9427
  - name: validation
    num_bytes: 2116452
    num_examples: 3270
  - name: test
    num_bytes: 2105959
    num_examples: 3245
  download_size: 6470231
  dataset_size: 10395714
- config_name: cb
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: idx
    dtype: int32
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': contradiction
          '2': neutral
  splits:
  - name: train
    num_bytes: 87050
    num_examples: 250
  - name: validation
    num_bytes: 21851
    num_examples: 56
  - name: test
    num_bytes: 93492
    num_examples: 250
  download_size: 139513
  dataset_size: 202393
- config_name: copa
  features:
  - name: premise
    dtype: string
  - name: choice1
    dtype: string
  - name: choice2
    dtype: string
  - name: question
    dtype: string
  - name: idx
    dtype: int32
  - name: label
    dtype:
      class_label:
        names:
          '0': choice1
          '1': choice2
  splits:
  - name: train
    num_bytes: 49233
    num_examples: 400
  - name: validation
    num_bytes: 12479
    num_examples: 100
  - name: test
    num_bytes: 59846
    num_examples: 500
  download_size: 86141
  dataset_size: 121558
- config_name: multirc
  features:
  - name: paragraph
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: idx
    struct:
    - name: paragraph
      dtype: int32
    - name: question
      dtype: int32
    - name: answer
      dtype: int32
  - name: label
    dtype:
      class_label:
        names:
        - 'False'
        - 'True'
  splits:
  - name: train
    num_bytes: 46182913
    num_examples: 27243
  - name: validation
    num_bytes: 7753452
    num_examples: 4848
  - name: test
    num_bytes: 14985531
    num_examples: 9693
  download_size: 2595545
  dataset_size: 68921896
- config_name: record
  features:
  - name: passage
    dtype: string
  - name: query
    dtype: string
  - name: entities
    sequence: string
  - name: entity_spans
    sequence:
    - name: text
      dtype: string
    - name: start
      dtype: int32
    - name: end
      dtype: int32
  - name: answers
    sequence: string
  - name: idx
    struct:
    - name: passage
      dtype: int32
    - name: query
      dtype: int32
  splits:
  - name: train
    num_bytes: 178757646
    num_examples: 100730
  - name: validation
    num_bytes: 17432944
    num_examples: 10000
  - name: test
    num_bytes: 17154943
    num_examples: 10000
  download_size: 91157911
  dataset_size: 213345533
- config_name: rte
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: idx
    dtype: int32
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': not_entailment
  splits:
  - name: train
    num_bytes: 847177
    num_examples: 2490
  - name: validation
    num_bytes: 90716
    num_examples: 277
  - name: test
    num_bytes: 973916
    num_examples: 3000
  download_size: 1277939
  dataset_size: 1911809
- config_name: wic
  features:
  - name: word
    dtype: string
  - name: sentence1
    dtype: string
  - name: sentence2
    dtype: string
  - name: start1
    dtype: int32
  - name: start2
    dtype: int32
  - name: end1
    dtype: int32
  - name: end2
    dtype: int32
  - name: idx
    dtype: int32
  - name: label
    dtype:
      class_label:
        names:
        - 'False'
        - 'True'
  splits:
  - name: train
    num_bytes: 658381
    num_examples: 5428
  - name: validation
    num_bytes: 81811
    num_examples: 638
  - name: test
    num_bytes: 178831
    num_examples: 1400
  download_size: 597226
  dataset_size: 919023
- config_name: wsc
  features:
  - name: text
    dtype: string
  - name: span1_index
    dtype: int32
  - name: span2_index
    dtype: int32
  - name: span1_text
    dtype: string
  - name: span2_text
    dtype: string
  - name: idx
    dtype: int32
  - name: label
    dtype:
      class_label:
        names:
        - 'False'
        - 'True'
  splits:
  - name: train
    num_bytes: 89311
    num_examples: 554
  - name: validation
    num_bytes: 21521
    num_examples: 104
  - name: test
    num_bytes: 31408
    num_examples: 146
  download_size: 49592
  dataset_size: 142240
- config_name: wsc.fixed
  features:
  - name: text
    dtype: string
  - name: span1_index
    dtype: int32
  - name: span2_index
    dtype: int32
  - name: span1_text
    dtype: string
  - name: span2_text
    dtype: string
  - name: idx
    dtype: int32
  - name: label
    dtype:
      class_label:
        names:
        - 'False'
        - 'True'
  splits:
  - name: train
    num_bytes: 89311
    num_examples: 554
  - name: validation
    num_bytes: 21521
    num_examples: 104
  - name: test
    num_bytes: 31404
    num_examples: 146
  download_size: 49706
  dataset_size: 142236
configs:
- config_name: axb
  data_files:
  - split: test
    path: axb/test-*
- config_name: axg
  data_files:
  - split: test
    path: axg/test-*
- config_name: boolq
  data_files:
  - split: train
    path: boolq/train-*
  - split: validation
    path: boolq/validation-*
  - split: test
    path: boolq/test-*
- config_name: cb
  data_files:
  - split: train
    path: cb/train-*
  - split: validation
    path: cb/validation-*
  - split: test
    path: cb/test-*
- config_name: copa
  data_files:
  - split: train
    path: copa/train-*
  - split: validation
    path: copa/validation-*
  - split: test
    path: copa/test-*
- config_name: multirc
  data_files:
  - split: train
    path: multirc/train-*
  - split: validation
    path: multirc/validation-*
  - split: test
    path: multirc/test-*
- config_name: record
  data_files:
  - split: train
    path: record/train-*
  - split: validation
    path: record/validation-*
  - split: test
    path: record/test-*
- config_name: rte
  data_files:
  - split: train
    path: rte/train-*
  - split: validation
    path: rte/validation-*
  - split: test
    path: rte/test-*
- config_name: wic
  data_files:
  - split: train
    path: wic/train-*
  - split: validation
    path: wic/validation-*
  - split: test
    path: wic/test-*
- config_name: wsc
  data_files:
  - split: train
    path: wsc/train-*
  - split: validation
    path: wsc/validation-*
  - split: test
    path: wsc/test-*
- config_name: wsc.fixed
  data_files:
  - split: train
    path: wsc.fixed/train-*
  - split: validation
    path: wsc.fixed/validation-*
  - split: test
    path: wsc.fixed/test-*
---

# Dataset Card for "super_glue"

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://super.gluebenchmark.com/
- **Repository:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Paper:** https://arxiv.org/abs/1905.00537
- **Point of Contact:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Size of downloaded dataset files:** 58.36 MB
- **Size of the generated dataset:** 249.57 MB
- **Total amount of disk used:** 307.94 MB

### Dataset Summary

SuperGLUE (https://super.gluebenchmark.com/) is a new benchmark styled after
GLUE with a new set of more difficult language understanding tasks, improved
resources, and a new public leaderboard.

### Supported Tasks and Leaderboards

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Languages

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Dataset Structure

### Data Instances

#### axb

- **Size of downloaded dataset files:** 0.03 MB
- **Size of the generated dataset:** 0.24 MB
- **Total amount of disk used:** 0.27 MB

An example of 'test' looks as follows.
```

```

#### axg

- **Size of downloaded dataset files:** 0.01 MB
- **Size of the generated dataset:** 0.05 MB
- **Total amount of disk used:** 0.06 MB

An example of 'test' looks as follows.
```

```

#### boolq

- **Size of downloaded dataset files:** 4.12 MB
- **Size of the generated dataset:** 10.40 MB
- **Total amount of disk used:** 14.52 MB

An example of 'train' looks as follows.
```

```

#### cb

- **Size of downloaded dataset files:** 0.07 MB
- **Size of the generated dataset:** 0.20 MB
- **Total amount of disk used:** 0.28 MB

An example of 'train' looks as follows.
```

```

#### copa

- **Size of downloaded dataset files:** 0.04 MB
- **Size of the generated dataset:** 0.13 MB
- **Total amount of disk used:** 0.17 MB

An example of 'train' looks as follows.
```

```

### Data Fields

The data fields are the same among all splits.

#### axb
- `sentence1`: a `string` feature.
- `sentence2`: a `string` feature.
- `idx`: a `int32` feature.
- `label`: a classification label, with possible values including `entailment` (0), `not_entailment` (1).

#### axg
- `premise`: a `string` feature.
- `hypothesis`: a `string` feature.
- `idx`: a `int32` feature.
- `label`: a classification label, with possible values including `entailment` (0), `not_entailment` (1).

#### boolq
- `question`: a `string` feature.
- `passage`: a `string` feature.
- `idx`: a `int32` feature.
- `label`: a classification label, with possible values including `False` (0), `True` (1).

#### cb
- `premise`: a `string` feature.
- `hypothesis`: a `string` feature.
- `idx`: a `int32` feature.
- `label`: a classification label, with possible values including `entailment` (0), `contradiction` (1), `neutral` (2).

#### copa
- `premise`: a `string` feature.
- `choice1`: a `string` feature.
- `choice2`: a `string` feature.
- `question`: a `string` feature.
- `idx`: a `int32` feature.
- `label`: a classification label, with possible values including `choice1` (0), `choice2` (1).

### Data Splits

#### axb

|   |test|
|---|---:|
|axb|1104|

#### axg

|   |test|
|---|---:|
|axg| 356|

#### boolq

|     |train|validation|test|
|-----|----:|---------:|---:|
|boolq| 9427|      3270|3245|

#### cb

|   |train|validation|test|
|---|----:|---------:|---:|
|cb |  250|        56| 250|

#### copa

|    |train|validation|test|
|----|----:|---------:|---:|
|copa|  400|       100| 500|

## Dataset Creation

### Curation Rationale

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the source language producers?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Annotations

#### Annotation process

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the annotators?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Personal and Sensitive Information

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Discussion of Biases

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Other Known Limitations

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Additional Information

### Dataset Curators

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Licensing Information

The primary SuperGLUE tasks are built on and derived from existing datasets. We refer users to the original licenses accompanying each dataset, but it is our understanding that these licenses allow for their use and redistribution in a research context.

### Citation Information

If you use SuperGLUE, please cite all the datasets you use in any papers that come out of your work. In addition, we encourage you to use the following BibTeX citation for SuperGLUE itself:
```
@article{wang2019superglue,
  title={Super{GLUE}: A Stickier Benchmark for General-Purpose Language Understanding Systems},
  author={Alex Wang and Yada Pruksachatkun and Nikita Nangia and Amanpreet Singh and Julian Michael and Felix Hill and Omer Levy and Samuel R. Bowman},
  journal={arXiv preprint 1905.00537},
  year={2019}
}
@inproceedings{clark2019boolq,
  title={{B}ool{Q}: Exploring the Surprising Difficulty of Natural Yes/No Questions},
  author={Clark, Christopher and Lee, Kenton and Chang, Ming-Wei and Kwiatkowski, Tom and Collins, Michael and Toutanova, Kristina},
  booktitle={Proceedings of NAACL-HLT 2019},
  year={2019}
}
@inproceedings{demarneffe:cb,
  title={{The CommitmentBank}: Investigating projection in naturally occurring discourse},
  author={De Marneffe, Marie-Catherine and Simons, Mandy and Tonhauser, Judith},
  note={To appear in proceedings of Sinn und Bedeutung 23. Data can be found at https://github.com/mcdm/CommitmentBank/},
  year={2019}
}
@inproceedings{roemmele2011choice,
  title={Choice of plausible alternatives: An evaluation of commonsense causal reasoning},
  author={Roemmele, Melissa and Bejan, Cosmin Adrian and Gordon, Andrew S.},
  booktitle={2011 AAAI Spring Symposium Series},
  year={2011}
}
@inproceedings{khashabi2018looking,
  title={Looking beyond the surface: A challenge set for reading comprehension over multiple sentences},
  author={Khashabi, Daniel and Chaturvedi, Snigdha and Roth, Michael and Upadhyay, Shyam and Roth, Dan},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
  pages={252--262},
  year={2018}
}
@article{zhang2018record,
  title={{ReCoRD}: Bridging the Gap between Human and Machine Commonsense Reading Comprehension},
  author={Sheng Zhang and Xiaodong Liu and Jingjing Liu and Jianfeng Gao and Kevin Duh and Benjamin Van Durme},
  journal={arXiv preprint 1810.12885},
  year={2018}
}
@incollection{dagan2006pascal,
  title={The {PASCAL} recognising textual entailment challenge},
  author={Dagan, Ido and Glickman, Oren and Magnini, Bernardo},
  booktitle={Machine learning challenges. evaluating predictive uncertainty, visual object classification, and recognising tectual entailment},
  pages={177--190},
  year={2006},
  publisher={Springer}
}
@article{bar2006second,
  title={The second {PASCAL} recognising textual entailment challenge},
  author={Bar Haim, Roy and Dagan, Ido and Dolan, Bill and Ferro, Lisa and Giampiccolo, Danilo and Magnini, Bernardo and Szpektor, Idan},
  year={2006}
}
@inproceedings{giampiccolo2007third,
  title={The third {PASCAL} recognizing textual entailment challenge},
  author={Giampiccolo, Danilo and Magnini, Bernardo and Dagan, Ido and Dolan, Bill},
  booktitle={Proceedings of the ACL-PASCAL workshop on textual entailment and paraphrasing},
  pages={1--9},
  year={2007},
  organization={Association for Computational Linguistics},
}
@article{bentivogli2009fifth,
  title={The Fifth {PASCAL} Recognizing Textual Entailment Challenge},
  author={Bentivogli, Luisa and Dagan, Ido and Dang, Hoa Trang and Giampiccolo, Danilo and Magnini, Bernardo},
  booktitle={TAC},
  year={2009}
}
@inproceedings{pilehvar2018wic,
  title={{WiC}: The Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations},
  author={Pilehvar, Mohammad Taher and Camacho-Collados, Jose},
  booktitle={Proceedings of NAACL-HLT},
  year={2019}
}
@inproceedings{rudinger2018winogender,
  title={Gender Bias in Coreference Resolution},
  author={Rudinger, Rachel  and  Naradowsky, Jason  and  Leonard, Brian  and  {Van Durme}, Benjamin},
  booktitle={Proceedings of NAACL-HLT},
  year={2018}
}
@inproceedings{poliak2018dnc,
  title={Collecting Diverse Natural Language Inference Problems for Sentence Representation Evaluation},
  author={Poliak, Adam and Haldar, Aparajita and Rudinger, Rachel and Hu, J. Edward and Pavlick, Ellie and White, Aaron Steven and {Van Durme}, Benjamin},
  booktitle={Proceedings of EMNLP},
  year={2018}
}
@inproceedings{levesque2011winograd,
  title={The {W}inograd schema challenge},
  author={Levesque, Hector J and Davis, Ernest and Morgenstern, Leora},
  booktitle={{AAAI} Spring Symposium: Logical Formalizations of Commonsense Reasoning},
  volume={46},
  pages={47},
  year={2011}
}
```

### Contributions

Thanks to [@thomwolf](https://github.com/thomwolf), [@lewtun](https://github.com/lewtun), [@patrickvonplaten](https://github.com/patrickvonplaten) for adding this dataset.