# Data Processing for TARA

We provide the preprocessed data [here](https://drive.google.com/drive/folders/1R6mIFFtGcF_d-Pjkeu-xPnOWXcp0zKHW?usp=sharing), which can be downloaded and used directly.

If you need to preprocess data from text, particularly from `{split}.jsonlines` files from RAMS and `{split}.json` files from WikiEvents, please follow the instructions below.

## 1. Transfer WikiEvents
Begin by using `transfer.py` in the [data_processing/wikievents](https://github.com/ayyyq/TARA/tree/main/data_processing/wikievents) directory to convert `{split}.jsonl` files from WikiEvents to `transfer-{split}.jsonl`. We will only require these files later.

## 2. Parse AMR

### Transition AMR
Please install the `transition-amr-parser` according to the instructions provided by [TSAR](https://github.com/RunxinXu/TSAR#2-data-preprocessing). Then, navigate to the [transition-amr-parser](https://github.com/RunxinXu/TSAR/blob/main/transition-amr-parser) folder and follow these steps:
1. Run `amrparse.py` to parse text into AMR and obtain `amr-{dataset_name}-{split}.pkl`.
2. Execute `amr2dglgraph.py` to transform the AMR into DGL graphs, resulting in `transition-dglgraph-{split}.pkl`.

### Compressed Transition AMR
We provide code `data_processing/compressed_transition/compress.py` for compressing parsed AMR. Please follow the instructions below to obtain compressed AMR graphs:
1. Parse text into AMR.
2. Compress the parsed AMR.
3. Transfer the compressed AMR to DGL graphs.

### AMRBART
Thanks to the authors of [AMRBART](https://github.com/goodbai-nlp/AMRBART), we offer [data_processing/amrbart](https://github.com/ayyyq/TARA/tree/main/data_processing/amrbart). Here are the steps:
1. Run `amr_parallel.py` to parse text into AMR using AMRBART.
2. Run `amr2dglgraph.py` to transform the AMR into DGL graphs, resulting in `amrbart-dglgraph-{split}.pkl`.
