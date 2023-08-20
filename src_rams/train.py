import os
import json
import collections
from functools import partial

from fastNLP import cache_results, prepare_torch_dataloader
from fastNLP import Trainer, Evaluator
from fastNLP import FitlogCallback, TorchWarmupCallback, TorchGradClipCallback
from fastNLP import CheckpointCallback, LoadBestModelCallback
from fastNLP import SortedSampler, BucketedBatchSampler
import torch
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig, DebertaV2Tokenizer, DebertaV2Config

from fastNLP import print, logger
import fitlog

from parse import parse_args
from pipe import AMRSpanPipe
from model import MyModel
from metrics import SpanMetric, OutputSaver
from utils import collate_fn, RAMSCheckpointCallback


def main():
    args = parse_args()
    # 根据CUDA_VISIBLE_DEVICES自动设置使用多少张卡
    device = list(range(os.environ.get('CUDA_VISIBLE_DEVICES', '').count(',') + 1))
    args.device = 0 if len(device) == 1 else device

    if args.load_model_dir is None:
        model_name = args.model_name.split('/')[1] if '/' in args.model_name else args.model_name
        tag_dir = '_'.join([args.dataset_name, model_name, args.amr_type, args.tag] if args.tag else [args.dataset_name, model_name, args.amr_type])
        tag_dir = os.path.join('outputs', tag_dir)

        save_dir = 'output'
        if args.debug:
            save_dir += '_debug'
        else:
            if args.learning_rate != 3e-5:
                save_dir += '_lr' + str(args.learning_rate)
            if args.non_pretrain_learning_rate != 1e-4:
                save_dir += '_nplr' + str(args.non_pretrain_learning_rate)
            if args.max_num_extracted_spans != 50:
                save_dir += '_nextract' + str(args.max_num_extracted_spans)
            if args.weight_decay != 0.1:
                save_dir += '_wd' + str(args.weight_decay)
            if args.prune_object != 'part':
                save_dir += '_' + args.prune_object
            if args.activation != 'relu':
                save_dir += '_' + args.activation
            if args.lambda_weight != 1.0:
                save_dir += '_lw' + str(args.lambda_weight)

            if args.loss_type == 'cross_entropy':
                # 默认use_label_mask
                if args.not_use_label_mask:
                    save_dir += '_wolm'
                if args.pos_loss_weight != 10:
                    save_dir += '_plw' + str(args.pos_loss_weight)
            elif args.loss_type == 'label_smooth':
                save_dir += '_ls' + str(args.label_smoothing)
                args.not_use_label_mask = True  # 默认not_use_label_mask
                if args.pos_loss_weight != 10:
                    save_dir += '_plw' + str(args.pos_loss_weight)
            elif args.loss_type == 'focal':
                save_dir += '_focal'
                args.not_use_label_mask = True
                if args.pos_loss_weight != -1:
                    save_dir += '_plw' + str(args.pos_loss_weight)

            if args.model_name == 'bert-base-cased' or args.model_name == 'bert-base-uncased':
                if args.gcn_layer_num != 4:
                    save_dir += '_gl' + str(args.gcn_layer_num)
                if args.warmup != 0.05:
                    save_dir += '_warm' + str(args.warmup)

            if args.dataset_name == 'rams':
                if args.gcn_layer_num != 4:
                    save_dir += '_gl' + str(args.gcn_layer_num)
                if args.weight_decay != 0.01:
                    save_dir += '_wd' + str(args.weight_decay)
                if args.warmup != 0.2:
                    save_dir += '_warm' + str(args.warmup)
            if args.suffix:
                save_dir += '_' + args.suffix
        save_dir = os.path.join(tag_dir, save_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        cur_save_dir = os.path.join(save_dir, os.environ['FASTNLP_LAUNCH_TIME'])  # outputs/dataset_model_tag/hparam/time
    else:
        cur_save_dir = args.load_model_dir  # outputs/dataset_model_tag/hparam/time/checkpoint
        tag_dir = os.path.split(os.path.split(os.path.split(cur_save_dir)[0])[0])[0]
    #####hyper
    schedule = 'linear'
    clip_value = 1
    #####hyper

    fitlog.set_log_dir('logs/')
    if args.debug or args.load_model_dir is not None:
        fitlog.debug()
    fitlog.commit(__file__)
    args.seed = fitlog.set_rng_seed(rng_seed=args.seed)
    os.environ['FASTNLP_GLOBAL_SEED'] = str(args.seed)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)
    logger.add_file(cur_save_dir, level='INFO')
    print(args)

    # ======== construct meta schema ===========
    if args.dataset_name == 'wikievents' or args.dataset_name == 'rams':
        meta_file = os.path.join(args.dataset_dir, args.dataset_name, 'meta.json')
    else:
        raise NotImplementedError
    with open(meta_file) as f:
        meta = json.load(f)
    event2id = {}
    id2event = {}
    eventid2role2id = {}
    eventid2id2role = {}
    role_id = 1
    num_labels = 1  # None + 所有的argument roles
    for i, d in enumerate(meta):
        event2id[d[0]] = i
        id2event[i] = d[0]
        roles = d[1]
        eventid2role2id[i] = {}
        eventid2id2role[i] = collections.OrderedDict()
        for role in roles:
            eventid2role2id[i][role] = role_id
            eventid2id2role[i][role_id] = role
            role_id += 1
            num_labels += 1
    event_num = len(event2id)
    meta_info = {'event2id': event2id,
                 'eventid2role2id': eventid2role2id,
                 'eventid2id2role': eventid2id2role,
                 'num_labels': num_labels}
    print('event_type_num', event_num)
    print('argument_type_num', num_labels)
    # ==================================================

    if args.dataset_name == 'wikievents':
        cache_file = 'caches/wikievents_caches.pkl'
    elif args.dataset_name == 'rams':
        cache_file = 'caches/rams_caches.pkl'
    else:
        cache_file = 'caches.pkl'
    @cache_results(os.path.join(tag_dir, cache_file), _refresh=args.refresh)  # 实际保存的文件名会受到传递给get_data函数参数的影响
    def get_data(paths, tokenizer, meta_info, max_len, max_span_width, model_name, dataset_name, amr_type):
        pipe = AMRSpanPipe(tokenizer, meta_info, max_len, max_span_width, model_name, dataset_name, amr_type)
        dl = pipe.process_from_file(paths)
        return dl

    print('Preprocessing data...')
    if args.dataset_name == 'wikievents':
        if args.amr_type == 'transition':
            if args.tag == 'compressed':
                raise NotImplementedError
            else:
                paths = {'train': [os.path.join(args.dataset_dir, args.dataset_name, 'transfer-train.jsonl'),
                                   os.path.join(args.dataset_dir, args.dataset_name, 'dglgraph-wikievents/dglgraph-wikievents-train.pkl')],
                         'dev': [os.path.join(args.dataset_dir, args.dataset_name, 'transfer-dev.jsonl'),
                                 os.path.join(args.dataset_dir, args.dataset_name, 'dglgraph-wikievents/dglgraph-wikievents-dev.pkl')],
                         'test': [os.path.join(args.dataset_dir, args.dataset_name, 'transfer-test.jsonl'),
                                  os.path.join(args.dataset_dir, args.dataset_name, 'dglgraph-wikievents/dglgraph-wikievents-test.pkl')]}
        else:
            raise NotImplementedError
    elif args.dataset_name == 'rams':
        if args.amr_type == 'transition':
            if args.tag == 'compressed':
                raise NotImplementedError
            else:
                paths = {'train': [os.path.join(args.dataset_dir, args.dataset_name, 'train.jsonlines'),
                                   os.path.join(args.dataset_dir, args.dataset_name,
                                                'dglgraph-rams/dglgraph-rams-train.pkl')],
                         'dev': [os.path.join(args.dataset_dir, args.dataset_name, 'dev.jsonlines'),
                                 os.path.join(args.dataset_dir, args.dataset_name,
                                              'dglgraph-rams/dglgraph-rams-dev.pkl')],
                         'test': [os.path.join(args.dataset_dir, args.dataset_name, 'test.jsonlines'),
                                  os.path.join(args.dataset_dir, args.dataset_name,
                                               'dglgraph-rams/dglgraph-rams-test.pkl')]}
        else:
            return NotImplementedError
    else:
        raise NotImplementedError
    if args.model_name.startswith('bert'):
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        config = BertConfig.from_pretrained(args.model_name, num_labels=num_labels)
    elif 'roberta' in args.model_name:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
        config = RobertaConfig.from_pretrained(args.model_name, num_labels=num_labels)
    elif 'deberta' in args.model_name:
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_name)
        config = DebertaV2Config.from_pretrained(args.model_name, num_labels=num_labels)
    else:
        raise NotImplementedError
    dl = get_data(paths, tokenizer, meta_info, max_len=args.max_len, max_span_width=args.max_span_width,
                  model_name=args.model_name, dataset_name=args.dataset_name, amr_type=args.amr_type)
    print(dl)

    dls = {}
    for name, ds in dl.iter_datasets():
        if name == 'train':
            _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=0,
                                           collate_fn=partial(collate_fn, vocab_pad=tokenizer.pad_token_id),
                                           batch_sampler=BucketedBatchSampler(ds, 'input_ids',
                                                                              batch_size=args.batch_size,
                                                                              num_batch_per_bucket=30),  # 桶内数据的长度都接近，使得每个batch中的padding数量会比较少
                                           pin_memory=True, shuffle=True)
        else:
            _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=0,
                                           collate_fn=partial(collate_fn, vocab_pad=tokenizer.pad_token_id),
                                           # sampler=SortedSampler(ds, 'input_ids'),  # 根据length从长到短迭代
                                           pin_memory=True, shuffle=False)
        dls[name] = _dl
    evaluate_dls = {'dev': dls['dev'], 'test': dls['test']}

    # ======== make some additional setting ==============
    setattr(config, 'model_name', args.model_name)
    setattr(config, 'event_num', event_num)
    setattr(config, 'dataset_name', args.dataset_name)
    # ======== make some additional setting ==============
    model = MyModel(config=config,
                    activation=args.activation,
                    gcn_layer_num=args.gcn_layer_num,
                    ffnn_depth=args.ffnn_depth,
                    event_embedding_size=args.event_embedding_size,
                    max_span_width=args.max_span_width,
                    span_width_embedding_size=args.span_width_embedding_size,
                    prune_ratio=args.prune_ratio,
                    part_ratio=args.part_ratio,
                    max_num_extracted_spans=args.max_num_extracted_spans,
                    prune_object=args.prune_object,
                    use_topk=not args.not_use_topk,
                    span_threshold=args.span_threshold,
                    loss_type=args.loss_type,
                    use_label_mask=not args.not_use_label_mask,
                    pos_loss_weight=args.pos_loss_weight,
                    label_smoothing=args.label_smoothing,
                    # pivot_epoch=args.pivot_epoch,
                    lambda_weight=args.lambda_weight,)

    counter = collections.Counter()
    for name, param in model.named_parameters():
        counter[name.split('.')[0]] += torch.numel(param)
    print(counter)
    print('Total param ', sum(counter.values()))  # 计算模型参数量
    fitlog.add_to_line(json.dumps(counter, indent=2))
    fitlog.add_other(value=sum(counter.values()), name='total_param')

    if args.load_model_dir is None:
        # optimizer
        non_decay_params = []
        decay_params = []
        non_pretrain_non_decay_params = []
        non_pretrain_decay_params = []
        for name, param in model.named_parameters():
            name = name.lower()
            if param.requires_grad is False:
                continue
            if 'pretrain_model' in name:
                if 'norm' in name or 'bias' in name:
                    non_decay_params.append(param)
                else:
                    decay_params.append(param)
            else:
                if 'norm' in name or 'bias' in name:
                    non_pretrain_non_decay_params.append(param)
                else:
                    non_pretrain_decay_params.append(param)
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
            {'params': non_decay_params, 'lr': args.learning_rate, 'weight_decay': 0.0},
            {'params': non_pretrain_decay_params, 'lr': args.non_pretrain_learning_rate, 'weight_decay': args.weight_decay},
            {'params': non_pretrain_non_decay_params, 'lr': args.non_pretrain_learning_rate, 'weight_decay': 0.0},
        ])

        monitor = 'f#f#dev'

        # callbacks
        callbacks = []
        callbacks.append(FitlogCallback(log_loss_every=500))
        callbacks.append(TorchGradClipCallback(clip_value=clip_value))
        callbacks.append(TorchWarmupCallback(warmup=args.warmup, schedule=schedule))
        callbacks.append(CheckpointCallback(folder=save_dir, topk=1, monitor=monitor))  # save_dir/时间/具体的checkpoint
        if args.dataset_name == 'rams':
            callbacks.append(RAMSCheckpointCallback(folder=save_dir, topk=2, monitor=monitor))
        callbacks.append(LoadBestModelCallback())  # 用于在训练结束之后加载性能最好的model的权重

        metrics = {'f': SpanMetric()}

        print('Training...')
        trainer = Trainer(model=model,
                          driver='torch',
                          train_dataloader=dls['train'],
                          evaluate_dataloaders=evaluate_dls,
                          optimizers=optimizer,
                          callbacks=callbacks,
                          device=args.device,
                          n_epochs=args.n_epochs,
                          # n_batches=10,
                          metrics=metrics,
                          monitor=monitor,
                          evaluate_every=-1,
                          evaluate_use_dist_sampler=True,
                          accumulation_steps=args.accumulation_steps,
                          fp16=False,
                          progress_bar='rich',
                          overfit_batches=0,  # overfit_batches > 0时不能用分布式
                          torch_kwargs={'ddp_kwargs': {'find_unused_parameters': True}})
        trainer.run(num_eval_sanity_batch=2)

    if args.load_model_dir is not None:
        dev_save_fp = os.path.join(args.load_model_dir, 'validation_predictions_span.jsonlines')
        test_save_fp = os.path.join(args.load_model_dir, 'test_predictions_span.jsonlines')
    else:
        dev_save_fp = os.path.join(cur_save_dir, 'validation_predictions_span.jsonlines')
        test_save_fp = os.path.join(cur_save_dir, 'test_predictions_span.jsonlines')
    if args.dataset_name == 'wikievents':
        dev_ref_fp = os.path.join(args.dataset_dir, args.dataset_name, 'transfer-dev.jsonl')
        test_ref_fp = os.path.join(args.dataset_dir, args.dataset_name, 'transfer-test.jsonl')
    elif args.dataset_name == 'rams':
        dev_ref_fp = os.path.join(args.dataset_dir, args.dataset_name, 'dev.jsonlines')
        test_ref_fp = os.path.join(args.dataset_dir, args.dataset_name, 'test.jsonlines')
    else:
        raise NotImplementedError

    print('Deving...')
    evaluator = Evaluator(model=model,
                          dataloaders=evaluate_dls['dev'],
                          metrics={'output': OutputSaver(dev_save_fp, dev_ref_fp, tokenizer, meta_info, args.dataset_name),
                                   'f': SpanMetric()},
                          driver='torch',
                          device=args.device,
                          use_dist_sampler=False)
    if args.load_model_dir is not None:
        print('Load checkpoint from ' + args.load_model_dir)
        evaluator.load_model(args.load_model_dir)
    evaluator.run()

    print('Testing...')
    evaluator = Evaluator(model=model,
                          dataloaders=evaluate_dls['test'],
                          metrics={'output': OutputSaver(test_save_fp, test_ref_fp, tokenizer, meta_info, args.dataset_name),
                                   'f': SpanMetric()},
                          driver='torch',
                          device=args.device,
                          use_dist_sampler=False)
    if args.load_model_dir is not None:
        print('Load checkpoint from ' + args.load_model_dir)
        evaluator.load_model(args.load_model_dir)
    evaluator.run()

    fitlog.finish()


if __name__ == '__main__':
    main()
