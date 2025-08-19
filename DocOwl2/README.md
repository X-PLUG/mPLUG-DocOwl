# mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding

<div align="center">
Anwen Hu, Haiyang Xu†, Liang Zhang, Jiabo Ye, Ming Yan†, Ji Zhang, Qin Jin, Fei Huang, Jingren Zhou

† Corresponding Author

</div>


<div align="center">
<a href="https://arxiv.org/abs/2409.03420"><img src="assets/Paper-Arxiv-orange.svg" ></a>
</div>
<div align="center">
Data: 
MP-DocStruct1M
<a href="https://huggingface.co/datasets/mPLUG/MP-DocStruct1M">🤗</a><a href="https://modelscope.cn/datasets/iic/MP-DocStruct1M"> <img src="assets/modelscope.png" width='20'></a>
MP-DocReason51K <a href="https://huggingface.co/datasets/mPLUG/MP-DocReason51K">🤗</a><a href="https://modelscope.cn/datasets/iic/MP-DocReason51K"> <img src="assets/modelscope.png" width='20'></a>
DocDownstream 2.0 <a href="https://huggingface.co/datasets/mPLUG/DocDownstream-2.0">🤗</a><a href="https://modelscope.cn/datasets/iic/DocDownstream-2.0"> <img src="assets/modelscope.png" width='20'></a>
DocGenome12K <a href="https://huggingface.co/datasets/mPLUG/DocGenome12K">🤗</a><a href="https://modelscope.cn/datasets/iic/DocGenome12K"> <img src="assets/modelscope.png" width='20'></a>
</div>
<div align="center">
Models:
DocOwl2-stage1 <a href="">🤗</a><a href=""> <img src="assets/modelscope.png" width='20'></a>
DocOwl2-stage2 <a href="">🤗</a><a href=""> <img src="assets/modelscope.png" width='20'></a>
DocOwl2 <a href="https://huggingface.co/mPLUG/DocOwl2">🤗</a><a href="https://modelscope.cn/models/iic/DocOwl2"> <img src="assets/modelscope.png" width='20'></a>
</div>


<hr>
<div align="center">
<img src="assets/docowl2_effiency_and_case.jpg" alt="image" width="90%" height="auto">
</div>
</p>

## Spotlights

* Support Multi-page Text Lookup and Multi-page Text Parsing.
* Support Multi-page Question Answering using simple phrases or detailed explanations with evidence pages.
* Support Text-rich Video Understanding.

* Open Source
    - ✅ Training Data: MP-DocStruct1M, MP-DocReason51K, DocDownsteam-2.0, DocGenome12K
    - ✅ Model: DocOwl2
    - ✅ Source code of model inference and evaluation.
    - [ ] Model: DocOwl2-stage1, DocOwl2-stage2, 
    - [ ]  Online Demo on ModelScope and HuggingFace.
    - [ ]  Source code of launching a local demo.
    - [ ] Training code.


## Training and Evaluation Datasets
|  Dataset   | Download Link |
|  ----  | ----  | 
|  MP-DocStruct1M   | <li>[HuggingFace: mPLUG/MP-DocStruct1M](https://huggingface.co/datasets/mPLUG/MP-DocStruct1M) <li> [ModelScope: iic/MP-DocStruct1M](https://modelscope.cn/datasets/iic/MP-DocStruct1M)|
|  DocDownstream-2.0   | <li>[HuggingFace: mPLUG/DocDownstream-2.0](https://huggingface.co/datasets/mPLUG/DocDownstream-2.0) <li> [ModelScope: iic/DocDownstream-2.0](https://modelscope.cn/datasets/iic/DocDownstream-2.0) |
|  MP-DocReason51K   | <li>[HuggingFace: mPLUG/MP-DocReason51K](https://huggingface.co/datasets/mPLUG/MP-DocReason51K) <li> [ModelScope: iic/MP-DocReason51K](https://modelscope.cn/datasets/iic/MP-DocReason51K)|
|  DocGenome12K   | <li>[HuggingFace: mPLUG/DocGenome12K](https://huggingface.co/datasets/mPLUG/DocGenome12K) <li> [ModelScope: iic/DocGenome12K](https://modelscope.cn/datasets/iic/DocGenome12K)|


## Models
### Model Card
|  Model   | Download Link  | Abilities |
|  ----  | ----  | ----  |
| DocOwl2  | <li> 🤗 [mPLUG/DocOwl2](https://huggingface.co/mPLUG/DocOwl2) <li> <img src="assets/modelscope.png" width='20'> [iic/DocOwl2](https://modelscope.cn/models/iic/DocOwl2) | <li> Multi-page VQA with detailed explanations <li> Multi-page VQA with concise answers |

### Model Inference
Prepare python environment.
For example, you can use uv as a virtual environment manager as below. In the top directory;
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
cd DocOwl2
```

Then just run below.
```python
python sample.py
```

### Model Evaluation
Prepare environments for evaluation.
For example, uv case,
```bash
uv sync
```
Or, you can manually install dependencies as follows:
```
pip install textdistance
pip install editdistance
pip install pycocoevalcap
```

Evaluate DocOwl2 on 10 single-image tasks, 2 multi-page tasks and 1 video task:
```
python docowl_benchmark_evaluate.py --model_path $MODEL_PATH --dataset $DATASET --downstream_dir $DOWNSTREAM_DIR_PATH --save_dir $SAVE_DIR --split $split
```
Note: 
For sinlge-image evaluation, ```$DATASET``` should be chosen from ```[DocVQA, InfographicsVQA, WikiTableQuestions, DeepForm,KleisterCharity, TabFact, ChartQA, TextVQA, TextCaps, VisualMRC]```. ```$DOWNSTREAM_DIR_PATH``` is the local path of [mPLUG/DocDownstream-1.0](https://huggingface.co/datasets/mPLUG/DocDownstream-1.0), ```$split==test```.

For multi-page evaluation and video evaluation, ```$DATASET``` should be chosen from ```[MP-DocVQA, DUDE, NewsVideoQA]```. ```$DOWNSTREAM_DIR_PATH``` is the local path of [mPLUG/DocDownstream-2.0](https://huggingface.co/datasets/mPLUG/DocDownstream-2.0), ```$split==val```. You can also set ```$split==test``` and submit the file named with suffix ```_submission.json``` to the [official evaluation website](https://rrc.cvc.uab.es/).
