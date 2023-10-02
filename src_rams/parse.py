import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default=None, type=str)
    parser.add_argument('--suffix', default=None, type=str)
    parser.add_argument('--dataset_dir', default='data', type=str)
    parser.add_argument('--dataset_name', default='rams', type=str)

    parser.add_argument('--train_text_path', default='train.jsonlines', type=str)
    parser.add_argument('--dev_text_path', default='dev.jsonlines', type=str)
    parser.add_argument('--test_text_path', default='test.jsonlines', type=str)
    parser.add_argument('--train_amr_path', default='transition/transition-dglgraph-train.pkl', type=str)
    parser.add_argument('--dev_amr_path', default='transition/transition-dglgraph-dev.pkl', type=str)
    parser.add_argument('--test_amr_path', default='transition/transition-dglgraph-test.pkl', type=str)
    parser.add_argument('--amr_type', default='transition', choices=['amrbart', 'transition'])

    parser.add_argument('--model_name', default='roberta-large', type=str)
    parser.add_argument('--load_model_dir', default=None, type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--refresh', action='store_true')

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--accumulation_steps', default=4, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('-n', '--n_epochs', default=100, type=int)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--non_pretrain_learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--warmup', default=0.1, type=float)

    parser.add_argument('--activation', default='relu', type=str)

    parser.add_argument('--max_len', default=512, type=int)
    parser.add_argument('--event_embedding_size', default=200, type=int)
    parser.add_argument('--max_span_width', default=12, type=int, help='max subwords span width')
    parser.add_argument('--span_width_embedding_size', default=20, type=int, help='dimension of span_width_embs')
    parser.add_argument('--prune_ratio', default=0.1, type=float)
    parser.add_argument('--max_num_extracted_spans', default=50, type=int)
    parser.add_argument('--prune_object', default='part', choices=['all', 'part'])
    parser.add_argument('--part_ratio', default=0.4, type=float)
    parser.add_argument('--not_use_topk', action='store_true')
    parser.add_argument('--span_threshold', default=0.0, type=float)
    parser.add_argument('--gcn_layer_num', default=3, type=int)
    parser.add_argument('--ffnn_depth', default=1, type=int)

    parser.add_argument('--loss_type', default='cross_enrtopy', choices=['cross_entropy', 'label_smooth', 'focal'])
    parser.add_argument('--not_use_label_mask', action='store_true')
    parser.add_argument('--pos_loss_weight', default=10, type=float)
    parser.add_argument('--label_smoothing', default=0.0, type=float)
    parser.add_argument('--lambda_weight', default=1.0, type=float)

    parser.add_argument('--seed', default=None, type=int)

    args = parser.parse_args()
    return args
