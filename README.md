# Incremental Prompting

## Introduction
This is the official repository for the paper "Incremental Prompting: Episodic Memory Prompt for Lifelong Event Detection" (COLING'22). More details on how to use the code are releasing soon. 

## Data Preparation
- We use the ACE and MAVEN datasets for evaluation. Please note that ACE is not publicly released and requires a license to access.
- For MAVEN dataset, download the json files under
```
./data/MAVEN/
```


## Training & Evaluation
Run the following script to start training. The script will also periodically evaluate the model on dev and test set.
```
python run.py
```

## Reference
