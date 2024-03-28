# mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding

<div align="center">
Anwen Hu, Haiyang Xu†, Jiabo Ye, Ming Yan†, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, Jingren Zhou

† Corresponding Author

</div>


<div align="center">
<a href="http://arxiv.org/abs/2403.12895"><img src="assets/Paper-Arxiv-orange.svg" ></a>
</div>
<div align="center">
Data: 
<a href="https://huggingface.co/datasets/mPLUG/DocStruct4M">DocStruct4M 🤗</a>
<a href="https://huggingface.co/datasets/mPLUG/DocReason25K">DocReason25K 🤗</a>
<a href="https://huggingface.co/datasets/mPLUG/DocDownstream-1.0">DocDownstream 🤗</a>
<a href="https://huggingface.co/datasets/mPLUG/DocLocal4K">DocLocal4K 🤗</a>
</div>
<div align="center">
Models:
<a href="https://huggingface.co/mPLUG/DocOwl1.5-stage1">DocOwl1.5-stage1 🤗</a>
<a href="https://huggingface.co/mPLUG/DocOwl1.5">DocOwl1.5 🤗</a>
<a href="https://huggingface.co/mPLUG/DocOwl1.5-Chat">DocOwl1.5-Chat 🤗</a>
</div>


<hr>
<div align="center">
<img src="assets/radar.png" alt="image" width="50%" height="auto">
<img src="assets/doc_instruct.png" alt="image" width="50%" height="auto">
</div>
</p>

## Spotlights

* Support struct-aware document parsing, table to markdown, chart to markdown.
* Support multi-grained text recognition and text grounding
* Support question answering with simple phrases or detailed explanations.

* Coming soon
    - [x] Training Data: DocStruct4M, DocReason25K, DocDownsteam-1.0
    - [x] Mutli-grained Text Localization Evaluation set: DocLocal4K
    - [x] Model: DocOwl 1.5-stage1, DocOwl 1.5, DocOwl 1.5-Chat
    - [x] Source code.
    - [ ] Online Demo on ModelScope.
    - [ ] Online Demo on HuggingFace.

## Training and Evaluation Datasets

### DocStruct4M
DocStruct4M is a training set for Unified Structure Learning, covering images of documents, webpages, tables, charts and natural images. It consists of ~3M samples for Struct-aware Parsing tasks and ~1M samples for Multi-grained Text Localization tasks. 

Download DocStruct4M dataset from huggingface [mPLUG/DocStruct4M](https://huggingface.co/datasets/mPLUG/DocStruct4M). Training images (~311G) are split into 8 files, run following cmds to prepare training and validation images.
```
cat partial-imgs* > imgs.tar.gz
tar -zxvf imgs.tar.gz
tar -zxvf val_imgs.tar.gz
```

The dataset is organized in such format:
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

The dataset is organized in such format:
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
The dataset is organized in such format:
```
DocReason25K
├── imgs
├── detailed_explanation.jsonl
```

### DocLocal4K
DocLocal4K is a evaluation set for Multi-grained Text Localization, covering both text recognition and text grounding tasks.

Download DocLocal4K dataset from huggingface [mPLUG/DocLocal4K](https://huggingface.co/datasets/mPLUG/DocLocal4K). 
The dataset is organized in such format:
```
DocLocal4K
├── imgs
├── text_grounding.jsonl
├── text_recognition.jsonl
```

## Models
### Model Card
|  Model   | Download Link  | Abilities |
|  ----  | ----  | ----  |
| DocOwl1.5-stage1  |  [🤗 mPLUG/DocOwl1.5-stage1](https://huggingface.co/mPLUG/DocOwl1.5-stage1) | <li> document/webpage parsing <li> table to markdown <li> chart to markdown <li> natural image parsing <li> multi-grained text recognition <li> multi-grained text  grounding |
| DocOwl1.5  |  [🤗 mPLUG/DocOwl1.5](https://huggingface.co/mPLUG/DocOwl1.5) | <li> VQA with concise answers <li> infomation extraction <li> image captioning <li> natural language inference |
| DocOwl1.5-Chat  |  [🤗 mPLUG/DocOwl1.5-Chat](https://huggingface.co/mPLUG/DocOwl1.5-Chat) | <li> VQA with detailed explanations <li> VQA with concise answers <li> infomation extraction <li> image captioning <li> natural language inference |
| DocOwl1.5-Chat+  |  coming soon | <li> document/webpage parsing <li> table to markdown <li> chart to markdown <li> natural image parsing <li> multi-grained text recognition <li> multi-grained text grounding <li> VQA with detailed explanations <li> VQA with concise answers <li> infomation extraction <li> image captioning <li> natural language inference |

### Model Inference
prepare python environments as [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2).

* DocOwl1.5-stage1 inference examples
```
from docowl_infer import DocOwlInfer
model_path = './mPLUG/DocOwl1.5-stage1'
docowl = DocOwlInfer(ckpt_path=model_path, anchors='grid_9', add_global_img=False)
print('load model from ', model_path)

# document/webpage parsing
image='./DocStruct4M/val_imgs/CCpdf/pages/1e531ef22cff3f01dab8720e99427c4f_page19.png'
query='Recognize text in the image.'
answer = docowl.inference(image, query)
print(answer)

# table/chart to markdown
image='./DocStruct4M/val_imgs/TURL/col_type_197091.jpg'
query='Convert the picture to Markdown syntax.'
answer = docowl.inference(image, query)
print(answer)

# natural image parsing
image='./DocStruct4M/val_imgs/OCRCC/02749938.jpg'
query=Provide a description of the image content and text.
answer = docowl.inference(image, query)
print(answer)
```

* DocOwl1.5-Chat inference examples
```
from docowl_infer import DocOwlInfer
model_path = './mPLUG/DocOwl1.5-chat'
docowl = DocOwlInfer(ckpt_path=model_path, anchors='grid_9', add_global_img=True)
print('load model from ', model_path)

# VQA with concise phrases
image='./DocDownstream-1.0/imgs/DUE_Benchmark/DocVQA/pngs/rnbx0223_193.png'
query='What is the Compound Annual Growth Rate (CAGR) for total assets?'
answer = docowl.inference(image, query)
print(answer)

# VQA with detailed explanation
image='./DocDownstream-1.0/imgs/DUE_Benchmark/DocVQA/pngs/rnbx0223_193.png'
query='What is the Compound Annual Growth Rate (CAGR) for total assets? Answer the question with detailed explanation.'
answer = docowl.inference(image, query)
print(answer)
```

### Model Evaluation
prepare environments for evaluation as follows:
```
pip install textdistance
pip install editdistance
pip install pycocoevalcap
```

Evaluate DocOwl1.5/DocOwl1.5-Chat on 10 downstream tasks:
```
python docowl_benchmark_evaluate.py --model_path $MODEL_PATH --dataset $DATASET --downstream_dir $DOWNSTREAM_DIR_PATH --save_dir $SAVE_DIR
```
Note: ```$DATASET``` should be chosen from ```[DocVQA, InfographicsVQA, WikiTableQuestions, DeepForm,KleisterCharity, TabFact, ChartQA, TextVQA, TextCaps, VisualMRC]```. ```$DOWNSTREAM_DIR_PATH``` is the local path of [mPLUG/DocDownstream-1.0](https://huggingface.co/datasets/mPLUG/DocDownstream-1.0).

Evaluate DocOwl1.5-stage1 on DocLocal4K:
```
python docowl_doclocal4k_evaluate.py --model_path $MODEL_PATH --task $TASK --doclocal4k_dir $DOCLOCAL4K_DIR_PATH --save_dir $SAVE_DIR
```
Note: ```$TASK``` should be chosen from ```[text_grounding, text_recognition]```. ```$DOCLOCAL4K_DIR_PATH``` is the local path of [mPLUG/DocLocal4K](https://huggingface.co/datasets/mPLUG/DocLocal4K).

### Model Training
coming soon


## Citation
If you found this work useful, consider giving this repository a star and citing our paper as followed:
```
@article{hu2024docowl,
  title={mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding},
  author={Hu, Anwen and Xu, Haiyang and Ye, Jiabo and Yan, Ming and Zhang, Liang and Zhang, Bo and Li, Chen and Zhang, Ji and Jin, Qin and Huang, Fei and others},
  journal={arXiv preprint arXiv:2403.12895},
  year={2024}
}
```

          
