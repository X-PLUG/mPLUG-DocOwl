<div align="center">
<img src="assets/mPLUG_new1.png" width="80%">
</div>

# mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding
<div align="center">
Jiabo Ye*, Anwen Hu*, Haiyang Xu†, Qinghao Ye, Ming Yan†, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, Qian Qi, Ji Zhang, Fei Huang
</div>
<div align="center">
<strong>DAMO Academy, Alibaba Group</strong>
</div>
<div align="center">
*Equal Contribution; † Corresponding Author
</div>

<div align="center">
    <a href="https://github.com/X-PLUG/mPLUG-DocOwl/blob/main/LICENSE"><img src="assets/LICENSE-Apache%20License-blue.svg" alt="License"></a>
    <a href="https://modelscope.cn/studios/damo/mPLUG-DocOwl/summary"><img src="assets/Demo-ModelScope-brightgreen.svg" alt="Demo ModelScope"></a>
    <a href="http://mm-chatgpt.oss-cn-zhangjiakou.aliyuncs.com/mplug_owl_demo/released_checkpoint/mPLUG_DocOwl_paper.pdf"><img src="assets/Paper-PDF-orange.svg"></a>
    <a href="https://arxiv.org/abs/2307.02499"><img src="assets/Paper-Arxiv-orange.svg" ></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FX-PLUG%2FmPLUG-DocOwl&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
    <a href="https://twitter.com/xuhaiya2483846/status/1677117982840090625"><img src='assets/-twitter-blue.svg'></a>
</div>

<div align="center">
<hr>
</div>

## News
* 🔥 [01.13] Our Scientific Diagram Analysis dataset [M-Paper](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/PaperOwl) has been available in HuggingFace, containing 447k high-resolution diagram images and corresponding paragraph analysis.
* [10.10] Our paper [UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model](https://arxiv.org/abs/2310.05126) is accepted by EMNLP 2023.
<!-- * 🔥 [10.10] The source code and instruction data will be released in [UReader](https://github.com/LukeForeverYoung/UReader). -->
* [07.10] The demo on [ModelScope](https://modelscope.cn/studios/damo/mPLUG-DocOwl/summary) is avaliable.
* [07.07] We release the technical report and evaluation set. The demo is coming soon.

## Spotlights

* An OCR-free end-to-end multimodal large language model.
* Applicable to various document-related scenarios.
* Capable of free-form question-answering and multi-round interaction.

* Comming soon
    - [x] Online Demo on ModelScope.
    - [ ] Online Demo on HuggingFace.
    - [ ] Source code.
    - [x] Instruction Training Data.
          
## Online Demo

### ModelScope
<a href="https://modelscope.cn/studios/damo/mPLUG-DocOwl/summary"><img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="250"/></a>

## Overview

![images](assets/overview.jpg)

## Cases

![images](assets/cases_git.jpg)





## DocLLM
The evaluation dataset DocLLM can be found in ```./DocLLM```.


## Related Projects

* [LoRA](https://github.com/microsoft/LoRA).
* [mPLUG](https://github.com/alibaba/AliceMind/tree/main/mPLUG).
* [mPLUG-2](https://github.com/alibaba/AliceMind).
* [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl)

## Citation
If you found this work useful, consider giving this repository a star and citing our paper as followed:
```
@misc{ye2023ureader,
      title={UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model}, 
      author={Jiabo Ye and Anwen Hu and Haiyang Xu and Qinghao Ye and Ming Yan and Guohai Xu and Chenliang Li and Junfeng Tian and Qi Qian and Ji Zhang and Qin Jin and Liang He and Xin Alex Lin and Fei Huang},
      year={2023},
      eprint={2310.05126},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@misc{ye2023mplugdocowl,
      title={mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding}, 
      author={Jiabo Ye and Anwen Hu and Haiyang Xu and Qinghao Ye and Ming Yan and Yuhao Dan and Chenlin Zhao and Guohai Xu and Chenliang Li and Junfeng Tian and Qian Qi and Ji Zhang and Fei Huang},
      year={2023},
      eprint={2307.02499},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
