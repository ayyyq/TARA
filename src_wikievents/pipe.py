from fastNLP.io import Pipe, Loader, DataBundle
from fastNLP import DataSet, Instance
from fastNLP import print
import torch
import json
from tqdm import tqdm
import collections
import numpy as np
import dgl
import spacy
from spacy.tokens import Doc
from scipy.sparse.csgraph import dijkstra

from utils import which_snt, map_context2graph, get_head, get_head_span


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)


class AMRSpanPipe(Pipe):
    def __init__(self, tokenizer, meta_info, max_len, max_span_width, model_name, dataset_name, amr_type):
        self.dataset_name = dataset_name
        self.amr_type = amr_type
        self.tokenizer = tokenizer
        self.TRIGGER_LEFT = 5 if model_name.startswith('bert') else self.tokenizer('[', add_special_tokens=False, return_attention_mask=False)['input_ids'][0]
        self.TRIGGER_RIGHT = 6 if model_name.startswith('bert') else self.tokenizer(']', add_special_tokens=False, return_attention_mask=False)['input_ids'][0]
        # print(self.TRIGGER_LEFT, self.TRIGGER_RIGHT)
        self.SNT_EDGE_TYPE = '6'
        self.meta_info = meta_info
        self.max_len = max_len  # input_ids的最大长度
        self.max_span_width = max_span_width  # span的subwords的最大长度

        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.tokenizer = WhitespaceTokenizer(self.nlp.vocab)

    def process(self, data_bundle: DataBundle) -> DataBundle:
        event2id = self.meta_info['event2id']
        eventid2role2id = self.meta_info['eventid2role2id']
        eventid2id2role = self.meta_info['eventid2id2role']
        num_labels = self.meta_info['num_labels']

        def process(ins):
            sentences = ins['sentences']

            snt2span = []  # snt2span[i]表示第i个句子头尾word的position
            start, end = 0, 0
            for idx, snt in enumerate(sentences):
                end = start + len(snt) - 1
                snt2span.append([start, end])
                start += len(snt)

            trigger = ins['evt_triggers'][0]
            trigger_b, trigger_e, event = trigger[0], trigger[1], trigger[2][0][0]
            # trigger_snt_id = which_snt(snt2span, [trigger_b, trigger_e])
            eventid = event2id[event]

            now_snt_idx = 0
            input_ids = [self.tokenizer.cls_token_id]
            belongingsnt = [0]
            subwords_snt2span = []
            wordidx2subwordidx = []
            start_subwordidx2wordidx = [-1]  # -1 denotes special tokens
            end_subwordidx2wordidx = [-1]
            for i, sentence in enumerate(sentences):
                subwords_snt2span_st = len(input_ids)
                for j, word in enumerate(sentence):
                    if now_snt_idx == trigger_b:
                        input_ids.append(self.TRIGGER_LEFT)
                        belongingsnt.append(i)
                        start_subwordidx2wordidx.append(-1)
                        end_subwordidx2wordidx.append(-1)
                    if now_snt_idx == trigger_e + 1:
                        input_ids.append(self.TRIGGER_RIGHT)
                        belongingsnt.append(i)
                        start_subwordidx2wordidx.append(-1)
                        end_subwordidx2wordidx.append(-1)
                    subwords_ids = self.tokenizer(word, add_special_tokens=False, return_attention_mask=False)['input_ids']
                    wordidx2subwordidx.append((len(input_ids), len(input_ids) + len(subwords_ids) - 1))  # wordidx2subwordidx[i]表示word[i]对应的subword的start和end
                    input_ids.extend(subwords_ids)
                    belongingsnt.extend([i] * len(subwords_ids))
                    start_subwordidx2wordidx.append(now_snt_idx)
                    start_subwordidx2wordidx += [-1] * (len(subwords_ids) - 1)
                    end_subwordidx2wordidx += [-1] * (len(subwords_ids) - 1)
                    end_subwordidx2wordidx.append(now_snt_idx)
                    now_snt_idx += 1
                subwords_snt2span.append([subwords_snt2span_st, len(input_ids) - 1])

            if len(input_ids) > self.max_len - 1:
                input_ids = input_ids[:self.max_len - 1]
                belongingsnt = belongingsnt[:self.max_len - 1]
                start_subwordidx2wordidx = start_subwordidx2wordidx[:self.max_len - 1]
                end_subwordidx2wordidx = end_subwordidx2wordidx[:self.max_len - 1]
            input_ids.append(self.tokenizer.sep_token_id)
            attention_mask = [1] * len(input_ids)
            belongingsnt.append(0)
            start_subwordidx2wordidx.append(-1)
            end_subwordidx2wordidx.append(-1)
            max_seq_len = len(input_ids)

            trigger_span = [min(wordidx2subwordidx[trigger_b][0], max_seq_len - 1),
                            min(wordidx2subwordidx[trigger_e][-1], max_seq_len - 1)]
            other_trigger_spans = []
            for trg in ins['other_evt_triggers']:
                trg_b = wordidx2subwordidx[trg[0]][0]
                trg_e = wordidx2subwordidx[trg[1]][-1]
                if trg_e < max_seq_len:
                    other_trigger_spans.append([trg_b, trg_e])
            if self.amr_type == 'transition':
                amr_graph = self.process_graph(ins['amr_graph'], wordidx2subwordidx, snt2span, max_seq_len)
            else:
                amr_graph = self.process_amrbart_graph(ins['amr_graph'], wordidx2subwordidx, max_seq_len)

            spans = []
            span_labels = []
            label_mask = [0] * num_labels
            label_mask[0] = 1
            label_mask = torch.tensor(label_mask, dtype=torch.long)
            for link in ins['gold_evt_links']:
                role_b, role_e = link[1]
                if role_b == -1 or role_e == -1:
                    continue
                role = link[-1]
                if role not in eventid2role2id[eventid]:
                    continue
                roleid = eventid2role2id[eventid][role]
                base_roleid = list(eventid2id2role[eventid].keys())[0]
                uppper_roleid = list(eventid2id2role[eventid].keys())[-1]
                label_mask[base_roleid:uppper_roleid + 1] = 1  # 给定event type，那么可选的label只有None + 该onthology下的argument roles
                role_subword_start_idx = wordidx2subwordidx[role_b][0]
                role_subword_end_idx = wordidx2subwordidx[role_e][-1]
                if role_subword_end_idx < max_seq_len:
                    spans.append([role_subword_start_idx, role_subword_end_idx])
                    span_labels.append(roleid)

            # construct negative examples
            # filter out spans that cross sentence boundaries and not starting or ending in a middle of a word
            for i in range(len(sentences)):
                start_idx, end_idx = snt2span[i]
                for s in range(start_idx, end_idx + 1):
                    sub_s = wordidx2subwordidx[s][0]
                    if sub_s >= max_seq_len:
                        break
                    for e in range(s, end_idx + 1):
                        sub_e = wordidx2subwordidx[e][-1]
                        if sub_e >= max_seq_len:
                            break
                        if 0 <= sub_e - sub_s < self.max_span_width:
                            if [sub_s, sub_e] not in spans:
                                spans.append([sub_s, sub_e])
            span_labels.extend([0] * (len(spans) - len(span_labels)))

            assert len(input_ids) == len(attention_mask)
            assert len(spans) == len(span_labels)

            return Instance(doc_key=ins['doc_key'],
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            spans=spans,
                            span_num=len(spans),
                            prune_labels=[int(x != 0) for x in span_labels],
                            labels=span_labels,
                            label_mask=label_mask,
                            event_id=eventid,
                            trigger_span=trigger_span,
                            other_trigger_spans=other_trigger_spans,
                            start_subwordidx2wordidx=start_subwordidx2wordidx,
                            end_subwordidx2wordidx=end_subwordidx2wordidx,
                            amr_graph=amr_graph,)

        for name in data_bundle.get_dataset_names():
            self.split_name = name
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            for ins in tqdm(ds, total=len(ds), desc=name):
                new_ins = process(ins)
                if new_ins is not None:
                    new_ds.append(new_ins)
            data_bundle.set_dataset(new_ds, name)

        # fastNLP将默认尝试对所有field都进行pad，默认值为0
        # data_bundle.set_ignore('doc_key')
        return data_bundle

    def process_graph(self, graphs, wordidx2subwordidx, snt2span, max_seq_len):
        # gold_span_info和trigger_span都是wordidx而非subwordidx
        assert len(graphs) == len(snt2span)

        graph_spans = []  # context_subword_span
        cur_big_graph = []
        bias_L = 0  # 当前sentence在context中的offset
        LL = [0]  # 每个sentence/local graph的root_id
        for graph_idx, g in enumerate(graphs):  # each sentence
            snt_spans = g.ndata['span']  # torch; word-level for every sentence itself
            for i in range(snt_spans.shape[0]):
                graph_subwords_b = wordidx2subwordidx[snt_spans[i][0] + bias_L][0]
                graph_subwords_e = wordidx2subwordidx[snt_spans[i][1] + bias_L][1]
                if graph_subwords_e >= max_seq_len or graph_subwords_b < 0 or graph_subwords_e < 0:
                    graph_subwords_b = 0
                    graph_subwords_e = max_seq_len - 1
                snt_spans[i][0] = graph_subwords_b
                snt_spans[i][1] = graph_subwords_e

            graph_spans.append(snt_spans)
            cur_big_graph.append(g)
            bias_L += snt2span[graph_idx][1] - snt2span[graph_idx][0] + 1
            LL.append(len(snt_spans) + LL[-1])

        graph_spans = torch.cat(graph_spans, dim=0)
        cur_big_graph = dgl.batch(cur_big_graph)  # 一个example一个graph
        assert cur_big_graph.num_nodes() == graph_spans.shape[0]
        cur_big_graph.ndata['span'] = graph_spans
        num_nodes = cur_big_graph.num_nodes()

        LL = LL[:-1]
        for root_i in LL:
            if root_i >= num_nodes:
                continue
            for root_j in LL:
                if root_j >= num_nodes:
                    continue
                if root_i != root_j:
                    cur_big_graph.add_edges(u=root_i, v=root_j, etype=self.SNT_EDGE_TYPE)

        initial_graph = {}
        for i in range(14):
            initial_graph[('node', str(i), 'node')] = ([], [])
        new_cur_big_graph = dgl.heterograph(initial_graph)

        new_cur_big_graph.add_nodes(num=num_nodes)
        new_cur_big_graph.ndata['span'] = graph_spans
        for etype in cur_big_graph.etypes:
            u, v = cur_big_graph.edges(etype=etype)
            new_cur_big_graph.add_edges(u=u, v=v, etype=etype)

        return new_cur_big_graph

    def process_amrbart_graph(self, graphs, wordidx2subwordidx, max_seq_len):
        cur_big_graph = graphs['graph']
        num_nodes = cur_big_graph.num_nodes()
        amrnode_info = graphs['amrnode_info']
        assert len(amrnode_info) == num_nodes

        features = torch.zeros(num_nodes, 2, dtype=torch.long)
        for i, info in enumerate(amrnode_info):
            if info['span'][0] < 0 or info['span'][1] < 0:
                features[i][0] = 0
                features[i][1] = 0
                continue
            graph_subwords_b = wordidx2subwordidx[info['span'][0]][0]
            graph_subwords_e = wordidx2subwordidx[info['span'][1]][1]
            if graph_subwords_e >= max_seq_len or graph_subwords_b < 0 or graph_subwords_e < 0:
                graph_subwords_b = 0
                graph_subwords_e = 0
            features[i][0] = graph_subwords_b
            features[i][1] = graph_subwords_e

        initial_graph = {}
        for i in range(14):
            initial_graph[('node', str(i), 'node')] = ([], [])
        new_cur_big_graph = dgl.heterograph(initial_graph)

        new_cur_big_graph.add_nodes(num=num_nodes)
        new_cur_big_graph.ndata['span'] = features
        for etype in cur_big_graph.etypes:
            u, v = cur_big_graph.edges(etype=etype)
            new_cur_big_graph.add_edges(u=u, v=v, etype=etype)

        return new_cur_big_graph

    def process_from_file(self, paths):
        dl = AMRSpanLoader().load(paths)
        return self.process(dl)


class AMRSpanLoader(Loader):
    def _load(self, path):
        # path: [data_path, amr_path]
        ds = DataSet()  # dict -> DataSet
        assert not isinstance(path, str) and len(path) == 2

        with open(path[0], 'r') as f:
            all_graphs = torch.load(path[1])
            for line, graphs in zip(f, all_graphs):
                example = json.loads(line)
                # assert len(example['sentences']) == len(graphs)
                ds.append(Instance(doc_key=example['doc_key'],
                                   sentences=example['sentences'],
                                   evt_triggers=example['evt_triggers'],
                                   other_evt_triggers=example['other_evt_triggers'],
                                   gold_evt_links=example['gold_evt_links'],
                                   offset=example['offset'] if 'wikievents' in path[0] else 0,
                                   amr_graph=graphs))
        return ds

    def load(self, paths):
        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
