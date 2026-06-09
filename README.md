# Learning What It Is Not: Boundary-Grounded Contrastive Prompting for Camouflaged Object Detection
## 1. Introduction

This repository provides the official PyTorch implementation of:

**Learning What It Is Not: Boundary-Grounded Contrastive Prompting for Camouflaged Object Detection**

**Status:** Under Review at Pacific Graphics (PG) 2026

Camouflaged Object Detection (COD) aims to segment objects that are visually concealed within their surroundings. Existing language-guided COD methods mainly focus on enhancing target semantics while overlooking explicit target-background discrimination. To address this issue, we propose a novel **Boundary-Grounded Contrastive Prompting Network (BCPNet)**, which introduces contrastive semantic prompting and boundary-aware alignment to facilitate accurate camouflage perception.

##2. Proposed Baseline
### 2.1 Create Environment

Creating a virtual environment in terminal：conda create -n BCPNet python=3.8
Installing necessary packages: pip install -r requirements.txt.
### 2.2 Downloading necessary data
downloading testing dataset and move it into ./data/TestDataset/

downloading training dataset and move it into ./data/TrainDataset/
