# 📄 DeepMetaIoT: A Multimodal Deep Learning Framework Harnessing Metadata for IoT Sensor Data Classification

[![Paper](https://img.shields.io/badge/DeepMetaIoT-PDF-green)](https://ieeexplore.ieee.org/document/11112691)

## 📌 Abstract

*Internet of Things (IoT) sensor data, which capture time series physical measurements such as temperature and humidity, often lack proper classification. This limits their effective understanding, integration, and reuse. While sensor metadata—textual descriptions of the measurements—is sometimes available, it is frequently incomplete or ambiguous. As a result, classification often depends solely on the time series data. Leveraging both time series sensor readings and textual metadata for automated and accurate classification remains a challenge due to the heterogeneity and inconsistency of these data sources. In this paper, we propose DeepMetaIoT, a multimodal deep learning framework that integrates time series and textual data for classification. DeepMetaIoT employs a cross-residual architecture comprising a time series encoder and a text encoder based on a pre-trained large language model, enabling effective fusion of both modalities. Experimental results on real-world IoT sensor datasets show that DeepMetaIoT consistently outperforms state-of-the-art machine learning and deep learning baselines.*

---

## 📂 Repository Overview

```
.
├── datasets/             # Contains datasets
├── scripts/              # Source code
├── paper/                # paper related other resources
├── requirements.txt      # Python dependencies
└── README.md

```
---

## ⚙️ Installation (Ubuntu Linux / MAC)

Clone this repository and install dependencies:

```bash
git clone https://github.com/skinan/DeepMetaIoT.git
cd DeepMetaIoT
```

**Conda environment setup**

```bash
conda create -n deepmetaiot python=3.9.0
conda activate deepmetaiot
pip install -r requirements.txt
```

---


## 📝 Citation

If you use this code or any resources of the paper (including datasets), please kindly cite our paper:

```bibtex
@article{inan2025deepmetaiot,
  title={DeepMetaIoT: A Multimodal Deep Learning Framework Harnessing Metadata for IoT Sensor Data Classification},
  author={Inan, Muhammad Sakib Khan and Liao, Kewen and Shen, Haifeng and Jayaraman, Prem Prakash and Montori, Federico and Georgakopoulos, Dimitrios},
  journal={IEEE Internet of Things Journal},
  year={2025},
  publisher={IEEE}
}
```
