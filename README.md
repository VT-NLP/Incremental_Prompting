# Incremental Prompting

### Introduction
This is the official repository for the paper "Incremental Prompting: Episodic Memory Prompt for Lifelong Event Detection" (COLING'22). More details on how to use the code are releasing soon. 

### Basic Requirements
Please ensure you have installed the following packages in your environment:
```
transformers==4.18.0
torch==1.7.1
torchmeta==1.8.0
numpy==1.19.5
tqdm==4.62.3
```

### Data Preparation
- We use the ACE and MAVEN datasets for evaluation. Please note that ACE is not publicly released and requires a license to access.
- First download the dataset files under the following directory with specified file names:
```
./data/{DATASET_NAME}/{DATASET_SPLIT}.json
```
where `DATASET_NAME = \[MAVEN, ACE\], DATASET_SPLIT = \[train, dev, test\]`

### Training & Evaluation
Run the following script to start training. The script will also periodically evaluate the model on dev and test set.
```
python run.py
```

### Reference
