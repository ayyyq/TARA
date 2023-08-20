# TARA

Source code for ACL 2023 paper: [An AMR-based Link Prediction Approach for Document-level Event Argument Extraction](https://arxiv.org/abs/2305.19162).

## 🔧 How to use our code?

### 1. Dependencies
```bash
pip install git+https://github.com/fastnlp/fastNLP@dev0.8.0
pip install git+https://github.com/fastnlp/fitlog
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install transformers==4.22.2
pip install dgl-cu102 dglgo -f https://data.dgl.ai/wheels/repo.html
```

### 2. Data Preprocessing
Kindly follow the instructions outlined in [TSAR](https://github.com/RunxinXu/TSAR#2-data-preprocessing) to obtain the *RAMS* and *WikiEvents* data, each accompnaied by its parsed AMR graphs. **Please replace `tranfer.py` in [data/wikievents](https://github.com/ayyyq/TARA/tree/main/data/wikievents) directory.**

It it important to note that we have not yet supplied compressed AMR graphs or AMR graphs generated using AMRBART. However, we plan to release them in the near future.

### 3. Training
Please first navigate to the apppropriate directory for *RAMS* or *Wikievents*: `src_rams` or `src_wikievents`. Then execute the commond `python train.py`. To make adjustments to hyperparameters, kindly refer to `src_x/parse.py` and implement any necessary modifications.

### 4. Evaluation
You can evaluate the trained model by running the following commands:
```shell
bash evaluate_rams.sh
bash evaluate_wikievents.sh
```

## 🥳 Citation

If you find our work useful, please cite our paper:
```bibtex
@inproceedings{DBLP:conf/acl/0004GHZQZ23,
  author       = {Yuqing Yang and
                  Qipeng Guo and
                  Xiangkun Hu and
                  Yue Zhang and
                  Xipeng Qiu and
                  Zheng Zhang},
  editor       = {Anna Rogers and
                  Jordan L. Boyd{-}Graber and
                  Naoaki Okazaki},
  title        = {An AMR-based Link Prediction Approach for Document-level Event Argument
                  Extraction},
  booktitle    = {Proceedings of the 61st Annual Meeting of the Association for Computational
                  Linguistics (Volume 1: Long Papers), {ACL} 2023, Toronto, Canada,
                  July 9-14, 2023},
  pages        = {12876--12889},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://doi.org/10.18653/v1/2023.acl-long.720},
  doi          = {10.18653/v1/2023.acl-long.720},
  timestamp    = {Thu, 10 Aug 2023 12:35:57 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/0004GHZQZ23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```