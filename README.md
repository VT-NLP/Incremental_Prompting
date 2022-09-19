# Incremental Prompting

### Introduction
This is the official repository for the paper "[Incremental Prompting: Episodic Memory Prompt for Lifelong Event Detection](https://arxiv.org/abs/2204.07275)" (COLING'22). More details on code will be released soon. 

### Basic Requirements
- Please make sure you have installed the following packages in your environment:
```
transformers==4.18.0
torch==1.7.1
torchmeta==1.8.0
numpy==1.19.5
tqdm==4.62.3
```
- You can install the requirements via running:
```
pip install -r requirements.txt
```

### Data Preparation
- We use the ACE and MAVEN datasets for evaluation. Please note that ACE is not publicly released and requires a license to access.
- First download the dataset files under the following directory with specified file names:
```
./data/{DATASET_NAME}/{DATASET_SPLIT}.jsonl
```
- Here `DATASET_NAME = {MAVEN, ACE}, DATASET_SPLIT = {train, dev, test}`. Please make sure you have downloaded the files on all three splits. Also note that you need to preprocess the ACE dataset into the same format as MAVEN.
- Then run the follow script to preprocess the datasets:
```
python prepare_inputs.py
```
The script will generate preprocessed files under the corresponding dataset directory.

### Training & Evaluation
- First create a directory`./logs/` which will stored the model checkpoints, and `./log/` which will stored log files. 
- Run the following script to start training. The script will also periodically evaluate the model on dev and test set.
```
python run.py
```

### Reference
**Please consider citing our paper if find it useful or interesting.**
```
@inproceedings{liu2022incremental,
    title={Incremental Prompting: Episodic Memory Prompt for Lifelong Event Detection},
    author={Liu, Minqian and Chang, Shiyu and Huang, Lifu},
    booktitle={Proceedings of the 29th International Conference On Computational Linguistics},  
    year={2022}
}
```

### Acknowledgement
Parts of the code in this repository are adopted from the work [Lifelong Event Detection with Knowledge Transfer](https://github.com/Perfec-Yu/Lifelong-ED). We thank Zhiyang Xu for constructive comments to this work.
