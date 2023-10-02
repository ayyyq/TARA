CUDA_VISIBLE_DEVICES=1 python src_wikievents/train.py \
--model_name bert-base-uncased \
--train_amr_path amrbart/amrbart-dglgraph-train.pkl \
--dev_amr_path amrbart/amrbart-dglgraph-dev.pkl \
--test_amr_path amrbart/amrbart-dglgraph-test.pkl \
--amr_type amrbart \
--refresh