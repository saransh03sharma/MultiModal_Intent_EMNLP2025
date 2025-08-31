
# MultiModal\_Intent\_EMNLP2025

## Overview

**MultiModal\_Intent\_EMNLP2025** is the official code and dataset repository for the EMNLP 2025 main conference paper:

**“Text Takes Over: A Study of Modality Bias in Multimodal Intent Detection.”**

This project investigates **intent recognition using multimodal data** (text, audio, and video) and presents a systematic study of **modality bias** in existing datasets and models. The repository contains datasets, experimental pipelines, and scripts required to **reproduce, analyze, and extend** the findings of the paper.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Dataset](#dataset)
* [Directory Structure](#directory-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Citation](#citation)

---

## Features

* 💡 **Research Contribution**: Codebase for the EMNLP 2025 paper *“Text Takes Over: A Study of Modality Bias in Multimodal Intent Detection.”*
* 📂 **Datasets Included**:

  * Original datasets from **MIntRec-1** and **MIntRec 2.0**.
  * **Debiased variants** created after Phase-3 of our study.
  * **Automated annotation variant** generated after Phase-2.
* ⚙️ **Experimental Pipelines**:

  * `Phase-1/` contains scripts for reproducing baseline experiments.
  * Instructions provided for evaluating **different categories of models** (LLMs, multimodal encoders, text-only baselines, etc.).
* 🔬 **Extensible Framework**: The structure allows researchers to test new models and compare them against our benchmarks.

---

## Dataset

The dataset folder provides:

* **Raw Data**: Multimodal samples with text, audio, and video.
* **Preprocessed Variants**: Ready-to-use splits aligned with our study phases.
* **Debiased & Annotated Versions**: To study and mitigate modality bias.

> ⚠️ Note: Due to licensing restrictions, certain subsets may need to be downloaded separately. Instructions are included inside the `Dataset/` folder.

---

## Directory Structure

```
MultiModal_Intent_EMNLP2025/
│
├── Dataset/             # All dataset variants (original, debiased, annotated)
├── Phase-1/             # Scripts and configs for Phase-1 experiments
├── requirements.txt      # Python dependencies
├── README.md             # Documentation (this file)
└── ...
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/saransh03sharma/MultiModal_Intent_EMNLP2025.git
   cd MultiModal_Intent_EMNLP2025
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running Phase-1 Experiments

```bash
cd Phase-1
python run_experiment.py --config configs/text_only.json
```

### Evaluating a Model

```bash
python evaluate.py --model your_model --dataset Dataset/debiased_variant
```

> More detailed instructions can be found inside each phase directory.

---

## Citation

If you use this repository, please cite our EMNLP 2025 paper:

```
@misc{mullick2025texttakesoverstudy,
      title={Text Takes Over: A Study of Modality Bias in Multimodal Intent Detection}, 
      author={Ankan Mullick and Saransh Sharma and Abhik Jana and Pawan Goyal},
      year={2025},
      eprint={2508.16122},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.16122}, 
}
```
