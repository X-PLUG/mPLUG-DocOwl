# mPLUG-PaperOwl: Scientific Diagram Analysis with the Multimodal Large Language Model

<div align="center">

Anwen Hu*, Yaya Shi*, Haiyang Xu†, Jiabo Ye, Qinghao Ye, Ming Yan†, Chenliang Li, Qi Qian, Ji Zhang, Fei Huang

Alibaba Group

*Equal Contribution; † Corresponding Author

</div>

<hr>
<div align="center">
<a href="http://anwenhu.oss-cn-zhangjiakou.aliyuncs.com/PaperOwl_arxiv.pdf"><img src="assets/Paper-PDF-orange.svg"></a>
<a href="https://arxiv.org/abs/2311.18248"><img src="assets/Paper-Arxiv-orange.svg" ></a>
<p>
<img src="assets/intro_case.jpeg" alt="image" width="50%" height="auto">
</div>
</p>

## M-Paper
Download M-Paper dataset from []().

The dataset is orgnized in such format:
```
M-Paper
├── imgs
├── sft
│   ├── 3tasks_{split}.jsonl
│   ├── cap_{split}.jsonl
│   ├── analysis_{split}.jsonl
│   └── outline_{split}.jsonl
├── meta
│   ├── cap_{split}.jsonl
│   ├── analysis_{split}.jsonl
│   └── outline_{split}.jsonl
```
The ```./imgs``` directory contains figure or table images. Files in the ```./sft``` directory are the instruction-tuning data. Files in the ```./meta``` directory store different components, e.g. [Context], [Outline], [Table_Latex], [Question], [Answer]，etc, of each sample in a dictionary format. 

The "task_type" item in each sample is organzied as $Object_$Task, where $Object indicates the understanding objects, including "figure", "table", "tablelatex", "figure_table" and "figure_tablelatex". $Task indicates the task, including "cap", "analysis", "outline_to_analysis", "simple_outline" and "detailed_outline".


## Training, Inference and Evaluation
### Environment
Follow [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) to prepare your environment.

We validate the codes with: 
* PyTorch 1.13.1
* CUDA 11.7
* transformers 4.29.1.

### Training
Prepare the checkpoint of mPLUG-Owl from [https://huggingface.co/MAGAer13/mplug-owl-llama-7b](https://huggingface.co/MAGAer13/mplug-owl-llama-7b). Put the download checkpoint in ```checkpoints/mplug-owl-llama-7b```.

For A100 80G
```
bash scripts/train_it.sh
```
For V100 32G
```
bash scripts/train_it_v100.sh
```
