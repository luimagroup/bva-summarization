# Automatic Summarization of Legal Decisions using Iterative Masking of Predictice Sentences

## Introduction

The repo is the code for our submission of `Automatic Summarization of Legal Decisions using Iterative Masking of Predictice Sentences' at ICAIL 2019.

The system is not end-to-end.


## Prerequisites

```
pip install -r requirements.txt
```


## Train-Attribute-Mask pipeline

```
cd attribution
python script.py [init_round] [mask_val_flag] [max_iter]
```

mask_val_flag == "true" means most predictive sentence will also be masked out in the validation data


## Create Summarization

```
cd summarization
python summarize.py
```

This gives summarization for every case. If want summary for a single case. The following function will help.

```
summarize.summarize(case_id, summary_len, verbose=1)
```
