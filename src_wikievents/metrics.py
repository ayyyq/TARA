from fastNLP import Metric, Callback, print
import torch
import numpy as np
import json


class SpanMetric(Metric):
    def __init__(self):
        super(SpanMetric, self).__init__()
        # 定义评测时需要用到的变量
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('fp', 0, aggregate_method='sum')
        self.register_element('fn', 0, aggregate_method='sum')
        # self.span_len = span_len

    def update(self, preds, labels, spans):
        # 定义评测变量的更新方式；参数名和evluate_step中的输出名称及数据集中对应字段名称一致
        # logits: bsz * span_num * num_labels
        preds = preds.tolist()
        labels = labels.tolist()
        spans = spans.tolist()

        NA_LABEL = 0
        PAD_LABEL = -100

        for example_idx, (pred, label, span) in enumerate(zip(preds, labels, spans)):
            for _pred, _label, _span in zip(pred, label, span):
                if _label == PAD_LABEL:
                    continue

                if _pred == _label:
                    if _pred != NA_LABEL:
                        self.tp += 1
                else:
                    if _pred != NA_LABEL:
                        self.fp += 1
                    else:
                        self.fn += 1

    def get_metric(self) -> dict:
        # 将根据update累计的评价指标统计量来计算最终的评价结果
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return {'f': f, 'p': p, 'r': r}


class SpanMetric_Eval(Metric):
    def __init__(self):
        super(SpanMetric_Eval, self).__init__()
        # 定义评测时需要用到的变量
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('fp', 0, aggregate_method='sum')
        self.register_element('fn', 0, aggregate_method='sum')
        self.register_element('gold', 0, aggregate_method='sum')
        self.register_element('exact', 0, aggregate_method='sum')
        self.register_element('overlap', 0, aggregate_method='sum')
        # self.span_len = span_len

    def update(self, preds, labels, spans):
        # 定义评测变量的更新方式；参数名和evluate_step中的输出名称及数据集中对应字段名称一致
        # logits: bsz * span_num * num_labels
        preds = preds.tolist()
        labels = labels.tolist()
        spans = spans.tolist()

        NA_LABEL = 0
        PAD_LABEL = -100

        for example_idx, (pred, label, span) in enumerate(zip(preds, labels, spans)):
            gold_infos = []
            for _label, _span in zip(label, span):
                if _label != NA_LABEL and _label != PAD_LABEL:
                    gold_infos.append([_span, _label])
            self.gold += len(gold_infos)

            for _pred, _label, _span in zip(pred, label, span):
                if _label == PAD_LABEL:
                    continue

                if _pred == _label:
                    if _pred != NA_LABEL:
                        self.tp += 1
                        self.exact += 1
                else:
                    if _pred != NA_LABEL:
                        self.fp += 1

                        for gold_span, gold_label in gold_infos:
                            if set(range(_span[0], _span[1] + 1)) & set(range(gold_span[0], gold_span[1] + 1)):
                                if _pred == gold_label:
                                    self.overlap += 1
                    else:
                        self.fn += 1

    def get_metric(self) -> dict:
        # 将根据update累计的评价指标统计量来计算最终的评价结果
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return {'f': f, 'p': p, 'r': r, 'gold': self.gold, 'exact': self.exact, 'overlap': self.overlap}


class PruneSpanMetric(Metric):
    def __init__(self):
        super(PruneSpanMetric, self).__init__()
        # 定义评测时需要用到的变量
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('fp', 0, aggregate_method='sum')
        self.register_element('fn', 0, aggregate_method='sum')
        # self.span_len = span_len

    def update(self, prune_preds, prune_labels):
        # 定义评测变量的更新方式；参数名和evluate_step中的输出名称及数据集中对应字段名称一致
        # logits: bsz * span_num * num_labels
        prune_preds = prune_preds.tolist()
        prune_labels = prune_labels.tolist()

        NA_LABEL = 0
        PAD_LABEL = -100

        for example_idx, (pred, label) in enumerate(zip(prune_preds, prune_labels)):
            for _pred, _label in zip(pred, label):
                if _label == PAD_LABEL:
                    continue

                if _pred == _label:
                    if _pred != NA_LABEL:
                        self.tp += 1
                else:
                    if _pred != NA_LABEL:
                        self.fp += 1
                    else:
                        self.fn += 1

    def get_metric(self) -> dict:
        # 将根据update累计的评价指标统计量来计算最终的评价结果
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return {'f': f, 'p': p, 'r': r}


class OutputSaver(Metric):
    def __init__(self, save_fp, ref_fp, tokenizer, meta_info, dataset_name):
        super().__init__()
        self.save_fp = save_fp
        self.ref_fp = ref_fp
        self.tokenizer = tokenizer
        self.event2id = meta_info['event2id']
        self.eventid2id2role = meta_info['eventid2id2role']
        self.dataset_name = dataset_name
        # self.span_len = span_len

        self.pred_list = []
        self.label_list = []
        self.span_list = []
        # 验证eval的顺序
        self.event_id_list = []
        self.doc_key_list = []
        self.start_subwordidx2wordidx_list = []
        self.end_subwordidx2wordidx_list = []

    def reset(self):
        # 在对每个evaluate_dataloaders遍历进行验证之前，reset会被调用来重置每个非element对象
        self.pred_list = []
        self.label_list = []
        self.span_list = []
        self.event_id_list = []
        self.doc_key_list = []
        self.start_subwordidx2wordidx_list = []
        self.end_subwordidx2wordidx_list = []

    def update(self, preds, labels, spans, event_ids, doc_keys, start_subwordidx2wordidx, end_subwordidx2wordidx):
        self.pred_list += preds.tolist()
        self.label_list += labels.tolist()
        self.span_list += spans.tolist()

        self.event_id_list += event_ids.tolist()
        self.doc_key_list += doc_keys
        self.start_subwordidx2wordidx_list += start_subwordidx2wordidx
        self.end_subwordidx2wordidx_list += end_subwordidx2wordidx

    def get_metric(self):
        from fastNLP import get_global_rank
        if get_global_rank() == 0:
            with open(self.save_fp, 'w') as writer, open(self.ref_fp, 'r') as reader:
                for pred, label, span, event_id, doc_key, start_subwordidx2wordidx, end_subwordidx2wordidx, line in zip(
                        self.pred_list, self.label_list, self.span_list,
                        self.event_id_list, self.doc_key_list, self.start_subwordidx2wordidx_list,
                        self.end_subwordidx2wordidx_list, reader):
                    example = json.loads(line)
                    assert doc_key == example['doc_key']
                    trigger = example['evt_triggers'][0]
                    trigger_b, trigger_e, event = trigger[0], trigger[1], trigger[2][0][0]
                    assert event_id == self.event2id[event]
                    result = {'doc_key': doc_key,
                              'predictions': [[]]}

                    result['predictions'][0].append([trigger_b, trigger_e])
                    all_ready_in_result = set()
                    for _pred, _label, _span in zip(pred, label, span):
                        if _label == -100 or _pred == 0:
                            continue
                        role_name = self.eventid2id2role[event_id][_pred]
                        if self.dataset_name == 'rams':
                            role_name = role_name[role_name.find('arg0') + 5:]
                        if start_subwordidx2wordidx[_span[0]] == -1 or end_subwordidx2wordidx[_span[1]] == -1:
                            continue
                        final_r = tuple([start_subwordidx2wordidx[_span[0]],
                                         end_subwordidx2wordidx[_span[1]],
                                         role_name,
                                         1.0])
                        if final_r not in all_ready_in_result:
                            all_ready_in_result.add(final_r)
                            result['predictions'][0].append(list(final_r))
                    writer.write(json.dumps(result) + '\n')
        print('Save the output file in ' + self.save_fp)
        return {}
