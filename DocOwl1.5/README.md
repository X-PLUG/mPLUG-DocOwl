# mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding

<div align="center">
Anwen Hu, Haiyang Xu†, Jiabo Ye, Ming Yan†, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, Jingren Zhou

† Corresponding Author

</div>


<div align="center">
<a href="http://arxiv.org/abs/2403.12895"><img src="assets/Paper-Arxiv-orange.svg" ></a>
</div>
<div align="center">
<a href="https://huggingface.co/datasets/mPLUG/DocStruct4M">DocStruct4M 🤗</a>
<a href="https://huggingface.co/datasets/mPLUG/DocReason25K">DocReason25K 🤗</a>
<a href="https://huggingface.co/datasets/mPLUG/DocDownstream-1.0">DocDownstream 🤗</a>
<a href="https://huggingface.co/datasets/mPLUG/DocLocal4K">DocLocal4K 🤗</a>
</div>


<hr>
<div align="center">
<img src="assets/radar.png" alt="image" width="50%" height="auto">
<img src="assets/doc_instruct.png" alt="image" width="50%" height="auto">
</div>
</p>


## Training and Evaluation Datasets

### DocStruct4M
DocStruct4M is a training set for Unified Structure Learning, covering images of documents, webpages, tables, charts and natural images. It consists of ~3M samples for Struct-aware Parsing tasks and ~1M samples for Multi-grained Text Localization tasks. 

Download DocStruct4M dataset from huggingface [mPLUG/DocStruct4M](https://huggingface.co/datasets/mPLUG/DocStruct4M). Training images (~311G) are split into 8 files, run following cmds to prepare training and validation images.
```
cat partial-imgs* > imgs.tar.gz
tar -zxvf imgs.tar.gz
tar -zxvf val_imgs.tar.gz
```

The dataset is orgnized in such format:
```
DocStruct4M
├── imgs
├── val_imgs
├── multi_grained_text_localization.jsonl
├── struct_aware_parse.jsonl
├── val.jsonl
```
The ```./imgs``` and ```./val_imgs``` directory contains images for the training and validation samples, respectively. 

### DocDownstream-1.0
DocDownstream-1.0 is the combination of 10 text-rich image understanding benchmarks, including DocVQA, InfographicsVQA, DeepForm, KleisterCharity, WikiTableQuestions, TabFact, ChartQA, TextCaps, TextVQA and VisualMRC, covering tasks of Information Extraction, Visual Question Answering, Natural Language Inference and Image Captioning. All tasks are unified in the form of Visual Question Answering.

Download DocDownstream-1.0 dataset from huggingface [mPLUG/DocDownstream-1.0](https://huggingface.co/datasets/mPLUG/DocDownstream-1.0). Images (~70G) are split into 2 files, run following cmds to prepare images.
```
cat partial-imgs* > imgs.tar.gz
tar -zxvf imgs.tar.gz
```

The dataset is orgnized in such format:
```
DocDownstream-1.0
├── meta
├── test
├── imgs
├── train.jsonl
├── val.jsonl
```
The ```./imgs``` directory contains images for the training/validation/test samples. The ```train.jsonl``` and ```val.jsonl``` are ensembled samples of 10 datasets for training and validation. There are ~57w samples in ```train.jsonl```. The ```./test``` directory contain test files for each dataset. The ```./meta``` directory contain meta files used for evaluation. 

### DocReason25K
DocReason25K is instruction tuning set with detailed explanation for Visual Document Understanding. It's built based on training samples from DocVQA, InfographicsVQA, WikiTableQuestions, VisualMRC, ChartQA and TextVQA. Detailed explanations are given by GPT3.5/GPT4V and further filtred according to manually annoatetd simple answer.

Download DocReason25K dataset from huggingface [mPLUG/DocReason25K](https://huggingface.co/datasets/mPLUG/DocReason25K). 
The dataset is orgnized in such format:
```
DocReason25K
├── imgs
├── detailed_explanation.jsonl
```

### DocLocal4K
DocLocal4K is a evaluation set for Multi-grained Text Localization, covering both text recognition and text grounding tasks.

Download DocLocal4K dataset from huggingface [mPLUG/DocLocal4K](https://huggingface.co/datasets/mPLUG/DocLocal4K). 
The dataset is orgnized in such format:
```
DocLocal4K
├── imgs
├── text_grounding.jsonl
├── text_recognition.jsonl
```

## Model





## Spotlights

* Support struct-aware document parsing, table to markdown, chart to markdown.
* Support multi-grained text recognition and text grounding
* Support question answering with simple phrases or detailed explanations.

* Coming soon
    - [x] Training Data: DocStruct4M, DocReason25K, DocDownsteam-1.0
    - [x] Mutli-grained Text Localization Evaluation set: DocLocal4K
    - [ ] Model: DocOwl 1.5, DocOwl 1.5-Chat
    - [ ] Source code.
    - [ ] Online Demo on ModelScope.
    - [ ] Online Demo on HuggingFace.
          
