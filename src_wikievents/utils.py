import torch
from fastNLP import Callback, CheckpointCallback


def which_snt(snt2span, span):
    for snt in range(len(snt2span)):
        snt_spans = snt2span[snt]
        if span[0] >= snt_spans[0] and span[1] <= snt_spans[1]:
            return snt  # 返回该span所在的snt idx
    assert False


def collate_fn(batch, vocab_pad):
    VOCAB_PAD = vocab_pad
    LABEL_PAD = -100

    max_seq_len = max([len(ex['input_ids']) for ex in batch])
    input_ids = [ex['input_ids'] + [VOCAB_PAD] * (max_seq_len - len(ex['input_ids'])) for ex in batch]
    input_ids = torch.LongTensor(input_ids)
    attention_mask = [ex['attention_mask'] + [0] * (max_seq_len - len(ex['attention_mask'])) for ex in batch]
    attention_mask = torch.LongTensor(attention_mask)
    # belongingsnts = [ex['belongingsnt'] + [0] * (max_seq_len - len(ex['belongingsnt'])) for ex in batch]
    # belongingsnts = torch.LongTensor(belongingsnts)

    max_span_num = max([len(ex['spans']) for ex in batch])
    spans = [ex['spans'] + [[0, 0]] * (max_span_num - len(ex['spans'])) for ex in batch]  # TODO: spans的pad会有多大的影响？
    spans = torch.LongTensor(spans)
    # span_lens = [ex['span_lens'] + [1] * (max_span_num - len(ex['span_lens'])) for ex in batch]
    # span_lens = torch.LongTensor(span_lens)
    prune_labels = [ex['prune_labels'] + [LABEL_PAD] * (max_span_num - len(ex['prune_labels'])) for ex in batch]
    prune_labels = torch.LongTensor(prune_labels)
    labels = [ex['labels'] + [LABEL_PAD] * (max_span_num - len(ex['labels'])) for ex in batch]
    labels = torch.LongTensor(labels)

    span_nums = torch.LongTensor([ex['span_num'] for ex in batch])
    label_masks = torch.stack([ex['label_mask'] for ex in batch], dim=0)  # bsz * num_labels
    event_ids = torch.LongTensor([ex['event_id'] for ex in batch])
    trigger_spans = [ex['trigger_span'] for ex in batch]
    other_trigger_spans = [ex['other_trigger_spans'] for ex in batch]
    # trigger_index = torch.LongTensor([ex['trigger_index'] for ex in batch])
    # trigger_snt_ids = torch.LongTensor([ex['trigger_snt_id'] for ex in batch])
    #
    # max_sent_num = max([len(ex['subwords_snt2span']) for ex in batch])
    # subwords_snt2spans = [ex['subwords_snt2span'] + [[0, 0]] * (max_sent_num - len(ex['subwords_snt2span'])) for ex in batch]
    # subwords_snt2spans = torch.LongTensor(subwords_snt2spans)

    doc_keys = [ex['doc_key'] for ex in batch]
    start_subwordidx2wordidx = [ex['start_subwordidx2wordidx'] for ex in batch]
    end_subwordidx2wordidx = [ex['end_subwordidx2wordidx'] for ex in batch]

    graphs = [ex['amr_graph'] for ex in batch]  # list
    # trigger_node_ids = [ex['trigger_node_id'] for ex in batch]  # list
    # node2span_masks = [ex['node2span_mask'] for ex in batch]  # list
    # max_graph_span_num = max([len(ex['node_label']) for ex in batch])
    # node_labels = [ex['node_label'] + [LABEL_PAD] * (max_graph_span_num - len(ex['node_label'])) for ex in batch]
    # node_labels = torch.LongTensor(node_labels)
    # pair_labels = [ex['pair_label'] + [LABEL_PAD] * (max_graph_span_num - len(ex['pair_label'])) for ex in batch]
    # pair_labels = torch.LongTensor(pair_labels)

    result = {'input_ids': input_ids,
              'attention_mask': attention_mask,
              'spans': spans,
              'span_nums': span_nums,
              # 'span_lens': span_lens,
              'prune_labels': prune_labels,
              'labels': labels,
              'label_masks': label_masks,
              'event_ids': event_ids,
              'trigger_spans': trigger_spans,
              'other_trigger_spans': other_trigger_spans,
              # 'trigger_index': trigger_index,
              # 'trigger_snt_ids': trigger_snt_ids,
              # 'subwords_snt2spans': subwords_snt2spans,
              # 'belongingsnts': belongingsnts,
              'doc_keys': doc_keys,
              'start_subwordidx2wordidx': start_subwordidx2wordidx,
              'end_subwordidx2wordidx': end_subwordidx2wordidx,
              'graphs': graphs,}
              # 'trigger_node_ids': trigger_node_ids,
              # 'node2span_masks': node2span_masks,
              # 'node_labels': node_labels,
              # 'pair_labels': pair_labels}
    return result


# def map_arg_spans(graph_span, context_span_info):
#     if graph_span[1] - graph_span[0] + 1 > 5:
#         return 0, 0, (-1, -1)
#     for context_span, (arg_role, flag) in context_span_info.items():
#         if set(range(graph_span[0], graph_span[1] + 1)) & set(range(context_span[0], context_span[1] + 1)):
#             assert flag is False  # Q2: 一个gold argument span对应唯一一个graph span
#             return 1, arg_role, context_span
#     return 0, 0, (-1, -1)


def map_context2graph(graph_span_info, context_span, head):
    ret_span = (-1, -1)
    for graph_span in graph_span_info.keys():
        if context_span[0] == graph_span[0] and context_span[1] == graph_span[1]:
            # 完全重合
            return graph_span
        elif graph_span[0] <= head <= graph_span[1]:
            # 部分重合时只考虑head
            if ret_span[0] < 0:
                ret_span = graph_span
            else:
                # 取更短的graph_span
                if graph_span[1] - graph_span[0] + 1 < ret_span[1] - ret_span[0] + 1:
                    ret_span = graph_span
    return ret_span


def get_head(doc, span_b, span_e):
    cur_i = span_b
    while span_b <= doc[cur_i].head.i <= span_e:
        if doc[cur_i].head.i == cur_i:
            # self is the head
            break
        else:
            cur_i = doc[cur_i].head.i
    return cur_i


def get_head_span(doc, head):
    pos_idx = [child.i for child in doc[head].children]
    span = [head]
    for i in range(1, 5):
        if head + i not in pos_idx and head - i not in pos_idx:
            break
        if head + i in pos_idx:
            span.append(head + i)
        if head - i in pos_idx:
            span.append(head - i)
    span = [min(span), max(span)]
    return span

class MyCheckpointCallback(CheckpointCallback):
    def __init__(self, folder=None, topk=0, monitor=None, larger_better=True, dataset_name='wikievents'):
        super().__init__(folder=folder, topk=topk, monitor=monitor, larger_better=larger_better)
        self.monitor = monitor
        self.larger_better = larger_better
        self.pivot_epoch_idx = 60 if dataset_name == 'wikievents' else 40

    def on_evaluate_end(self, trainer, results):
        if trainer.cur_epoch_idx < self.pivot_epoch_idx:
            def monitor(results):
                return 0.0
            self.topk_saver.set_monitor(monitor=monitor, larger_better=self.larger_better)
        else:
            self.topk_saver.set_monitor(monitor=self.monitor, larger_better=self.larger_better)

        # 如果发生了保存，则返回的 folder 不为 None
        folder = self.topk_saver.save_topk(trainer, results)
