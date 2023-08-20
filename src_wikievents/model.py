import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel, DebertaV2Model
import dgl
import dgl.nn.pytorch as dglnn
# from dgl.nn import RelGraphConv
from collections import Iterable
from fastNLP import Callback


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class MyModel(nn.Module):
    def __init__(self,
                 config,
                 activation=None,
                 gcn_layer_num=None,
                 ffnn_depth=None,
                 event_embedding_size=None,
                 max_span_width=None,
                 span_width_embedding_size=None,
                 prune_ratio=None,
                 part_ratio=None,
                 max_num_extracted_spans=None,
                 prune_object=None,
                 use_topk=None,
                 span_threshold=None,
                 loss_type=None,
                 use_label_mask=None,
                 pos_loss_weight=None,
                 label_smoothing=None,
                 # pivot_epoch=None,
                 lambda_weight=None, ):
        super().__init__()
        self.dataset_name = config.dataset_name
        self.model_name = config.model_name
        self.num_labels = config.num_labels
        if config.model_name.startswith('bert'):
            self.pretrain_model = BertModel.from_pretrained(config.model_name)
        elif 'roberta' in config.model_name:
            self.pretrain_model = RobertaModel.from_pretrained(config.model_name)
        elif 'deberta' in config.model_name:
            self.pretrain_model = DebertaV2Model.from_pretrained(config.model_name)
        else:
            raise NotImplementedError
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if activation == 'leaky_relu':
            activation_func = nn.LeakyReLU()
        elif activation == 'gelu':
            activation_func = nn.GELU()
        else:
            activation_func = nn.ReLU()

        self.transform_start = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_end = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_span = nn.Linear(config.hidden_size * 3, config.hidden_size)

        node_type_num = 4  # context/trigger/other triggers/argument candidates
        self.node_type_embedding = nn.Embedding(node_type_num, config.hidden_size)
        # self.node_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.scorer = self.make_ffnn(config.hidden_size, output_size=1, hidden_size=[config.hidden_size] * ffnn_depth)
        self.span_width_scorer = self.make_ffnn(span_width_embedding_size, output_size=1,
                                                hidden_size=[config.hidden_size] * ffnn_depth)
        self.max_span_width = max_span_width
        self.span_width_prior_embeddings = nn.Embedding(max_span_width, span_width_embedding_size)
        self.prune_ratio = prune_ratio
        self.part_ratio = part_ratio
        self.max_num_extracted_spans = max_num_extracted_spans
        self.prune_object = prune_object
        self.use_topk = use_topk
        self.span_threshold = span_threshold

        # GRAPH
        self.rel_name_lists = [str(i) for i in range(14)]
        self.CONTEXT_ETYPE = '13'
        self.gcn_layers = nn.ModuleList([RelGraphConvLayer(config.hidden_size, config.hidden_size,
                                                           self.rel_name_lists,
                                                           num_bases=len(self.rel_name_lists),
                                                           activation=nn.ReLU(),
                                                           self_loop=True,
                                                           dropout=config.hidden_dropout_prob * 3) for i in
                                         range(gcn_layer_num)])
        self.after_layer = nn.Sequential(
            nn.Linear(config.hidden_size * (gcn_layer_num + 1), config.hidden_size),
            activation_func,
            nn.Dropout(config.hidden_dropout_prob)
        )

        pair_feature_dim = config.hidden_size * 2
        if event_embedding_size > 0:
            pair_feature_dim += event_embedding_size
            self.event_embedding = nn.Embedding(config.event_num, event_embedding_size)
        else:
            self.event_embedding = None
        self.classifier = nn.Sequential(
            nn.Linear(pair_feature_dim, config.hidden_size),
            activation_func,
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.num_labels)
        )

        self.loss_type = loss_type
        self.use_label_mask = use_label_mask
        if self.loss_type == 'focal':
            self.loss_fct = FocalLoss()
        else:
            self.label_smoothing = label_smoothing
            if pos_loss_weight > 0:
                self.pos_loss_weight = torch.tensor([pos_loss_weight for _ in range(self.num_labels)]).float()
                self.pos_loss_weight[0] = 1.
                self.loss_fct = nn.CrossEntropyLoss(weight=self.pos_loss_weight, label_smoothing=self.label_smoothing)
            else:
                self.loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.lambda_weight = lambda_weight

    def make_ffnn(self, feat_size, output_size, hidden_size=None):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return nn.Linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [nn.Linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [nn.Linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(nn.Linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def select_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B x num
        Returns:
            B x num x dim
        """
        B, L, dim = batch_rep.size()
        _, num = token_pos.size()
        shift = (torch.arange(B).unsqueeze(-1).expand(-1, num) * L).contiguous().view(-1).to(batch_rep.device)
        token_pos = token_pos.contiguous().view(-1)
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res.view(B, num, dim)

    def select_single_token_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B
        Returns:
            B x dim
        """
        B, L, dim = batch_rep.size()
        shift = (torch.arange(B) * L).to(batch_rep.device)
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res

    def extract_top_spans(self, candidate_idx_sorted, candidate_starts, candidate_ends, span_num, top_span_num,
                          no_cross_overlap=True):
        """
        Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of loop
        """
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {}
        for candidate_idx in candidate_idx_sorted:
            if candidate_idx >= span_num:
                continue
            if len(selected_candidate_idx) >= top_span_num:  # 最多num_top_spans个spans
                break
            # Perform overlapping check
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            cross_overlap = False
            for token_idx in range(span_start_idx, span_end_idx + 1):
                max_end = start_to_max_end.get(token_idx, -1)  # 以token_idx为start的max_end
                if token_idx > span_start_idx and max_end > span_end_idx and no_cross_overlap:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end_idx and 0 <= min_start < span_start_idx and no_cross_overlap:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Pass check; select idx and update dict stats
                selected_candidate_idx.append(candidate_idx)
                max_end = start_to_max_end.get(span_start_idx, -1)
                if span_end_idx > max_end:
                    start_to_max_end[span_start_idx] = span_end_idx
                min_start = end_to_min_start.get(span_end_idx, -1)
                if min_start == -1 or span_start_idx < min_start:
                    end_to_min_start[span_end_idx] = span_start_idx
        # Sort selected candidates by span idx
        selected_candidate_idx = sorted(selected_candidate_idx,
                                        key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))

        # if len(selected_candidate_idx) < num_top_spans:  # Padding
        #     print('length of selected candidates (' + str(len(selected_candidate_idx)) +
        #                    ') lower than num_top_spans (' + str(num_top_spans) + ')')
        #     selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
        return selected_candidate_idx

    def get_span_width_scores(self, span_starts, span_ends):
        span_width_idx = torch.clamp(span_ends - span_starts, max=self.max_span_width - 1)  # 只有gold_spans会被clamp
        span_width_embs = self.span_width_prior_embeddings(span_width_idx)
        width_scores = torch.squeeze(self.span_width_scorer(span_width_embs), dim=-1)
        return width_scores

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            spans=None,
            span_nums=None,
            prune_labels=None,
            labels=None,
            label_masks=None,
            event_ids=None,
            trigger_spans=None,
            other_trigger_spans=None,
            graphs=None,
    ):
        # fastNLP使用形参名匹配的方式进行参数传递
        # 没有train_step和evaluate_step时直接调用模型的forward，train_step的返回值必须为dict且包含loss
        # ================= GLOBAL =================
        last_hidden_state = self.pretrain_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)  # bsz * seq_len * hidsize
        bsz, seq_len, hidsize = last_hidden_state.size()
        span_num = spans.size(1)
        span_starts = spans[:, :, 0]
        span_ends = spans[:, :, 1]
        # ================= GLOBAL =================

        # ================= span extractor =================
        start_feature = self.transform_start(last_hidden_state)
        end_feature = self.transform_end(last_hidden_state)
        b_feature = self.select_rep(start_feature, span_starts)
        e_feature = self.select_rep(end_feature, span_ends)

        context = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).repeat(bsz, span_num, 1).to(last_hidden_state.device)
        context_mask = (context >= spans[:, :, 0:1]) & (context <= spans[:, :, 1:])
        context_mask = context_mask.float()
        context_mask /= torch.sum(context_mask, dim=-1, keepdim=True)
        context_feature = torch.bmm(context_mask, last_hidden_state)
        span_feature = torch.cat((b_feature, e_feature, context_feature), dim=-1)
        span_feature = self.transform_span(span_feature)  # bsz * span_num * hidsize
        # ================= span extractor =================

        prune_scores = torch.squeeze(self.scorer(span_feature), -1)  # bsz * span_num
        width_scores = self.get_span_width_scores(span_starts, span_ends)
        prune_scores += width_scores

        # ============== GLOBAL GRAPH ===============
        span_scores = []
        span_labels = []
        pair_labels = []
        all_selected_indices = []
        all_node_idx_list = []

        all_graphs = []  # 考虑batch_size
        all_span_infos = []
        all_node_features = []
        assert len(graphs) == bsz
        for example_idx, cur_big_graph in enumerate(graphs):
            # ================= span pruner =================
            if self.training:
                span_num = span_nums[example_idx].item()
                k = min(int(span_num * self.prune_ratio + 1), self.max_num_extracted_spans)
                selected_indices = torch.topk(prune_scores[example_idx], k)[1]  # 这个k仍然限制了必须是每个example

                if self.prune_object == 'all':
                    span_scores.append(prune_scores[example_idx])
                    span_labels.append(prune_labels[example_idx])
                else:
                    k = int(span_num * self.part_ratio)
                    part_indices = torch.topk(prune_scores[example_idx], k)[1]
                    span_scores.append(prune_scores[example_idx][part_indices])
                    span_labels.append(prune_labels[example_idx][part_indices])
                    # 只关注filter出来的spans的loss

                # if not self.use_topk:
                #     # TODO: 没明白这里是什么意思
                #     # Ignore invalid mentions even during training
                #     selected_indices = torch.nonzero(prune_labels == 1).squeeze()
            else:
                if self.use_topk:
                    k = min(int(span_num * self.prune_ratio) + 1, self.max_num_extracted_spans)
                    selected_indices = torch.topk(prune_scores[example_idx], k)[1]
                else:
                    selected_indices = torch.nonzero(prune_scores[example_idx] >= self.span_threshold).squeeze(-1)
                    if selected_indices.size(0) > self.max_num_extracted_spans:
                        selected_indices = torch.topk(prune_scores[example_idx], self.max_num_extracted_spans)[1]

            selected_indices = selected_indices.clamp(max=span_num - 1)
            pair_labels.append(labels[example_idx][selected_indices])
            all_selected_indices.append(selected_indices)
            # ================= span pruner =================

            # ================= add graph nodes =================
            node_idx_list = []  # pruned_span_list中每个span对应的graph_node_id
            add_span = []
            add_edges_u = []
            add_edges_v = []
            pruned_span_list = spans[example_idx][selected_indices].tolist()
            pruned_span_list = [trigger_spans[example_idx]] + other_trigger_spans[example_idx] + pruned_span_list
            graph_span_list = cur_big_graph.ndata['span'].tolist()
            cur_node_idx = len(graph_span_list) - 1
            for span in pruned_span_list:
                add_node = True
                left_node_idx = -1
                right_node_idx = -1
                for node_idx, graph_span in enumerate(graph_span_list):
                    if span == graph_span:
                        node_idx_list.append(node_idx)
                        add_node = False
                        break
                    if graph_span[1] <= span[1]:
                        if left_node_idx == -1 or graph_span_list[left_node_idx][1] < graph_span[1]:
                            left_node_idx = node_idx
                    if span[0] <= graph_span[0]:
                        if right_node_idx == -1 or graph_span[0] < graph_span_list[right_node_idx][0]:
                            right_node_idx = node_idx
                if add_node:  # TODO: 关于overlap，graph_node本身相同或者overlap怎么处理？ TODO: graph_node中相同的连起来
                    cur_node_idx += 1
                    add_span.append(span)
                    if left_node_idx > 0:
                        add_edges_u += [cur_node_idx, left_node_idx]
                        add_edges_v += [left_node_idx, cur_node_idx]
                    if right_node_idx > 0 and left_node_idx != right_node_idx:
                        add_edges_u += [cur_node_idx, right_node_idx]
                        add_edges_v += [right_node_idx, cur_node_idx]
                    node_idx_list.append(cur_node_idx)
            if add_span:
                add_span = torch.LongTensor(add_span).to(last_hidden_state.device)
                cur_big_graph.add_nodes(add_span.size(0), {'span': add_span})
                cur_big_graph.add_edges(u=add_edges_u, v=add_edges_v, etype=self.CONTEXT_ETYPE)
                cur_big_graph = cur_big_graph.to(last_hidden_state.device)
            all_node_idx_list.append(node_idx_list)
            # ================= add graph nodes =================

            span_info = cur_big_graph.ndata['span']
            node_num = span_info.size(0)
            all_graphs.append(cur_big_graph)
            all_span_infos.append(span_info)

            # obtain initialized node representations from PLM
            node_b_feature = self.select_rep(start_feature[example_idx].unsqueeze(0), span_info[None, :, 0]).squeeze(0)
            node_e_feature = self.select_rep(end_feature[example_idx].unsqueeze(0), span_info[None, :, 1]).squeeze(0)

            graph_span_mask = torch.arange(seq_len).unsqueeze(0).repeat(node_num, 1).to(last_hidden_state.device)
            graph_span_mask = (graph_span_mask >= span_info[:, 0:1]) & (graph_span_mask <= span_info[:, 1:])
            graph_span_mask = graph_span_mask.float()
            graph_span_mask_num = torch.sum(graph_span_mask, dim=-1, keepdim=True)  # node_num * 1
            graph_span_mask /= (graph_span_mask_num == 0).float() + graph_span_mask_num
            node_feature = torch.mm(graph_span_mask, last_hidden_state[example_idx])

            node_feature = torch.cat((node_b_feature, node_e_feature, node_feature), dim=-1)
            node_feature = self.transform_span(node_feature)  # bsz * span_num * hidsize

            # node_type_embedding
            node_type = torch.zeros(node_num, dtype=torch.long, device=last_hidden_state.device)
            node_type[node_idx_list[1 + len(other_trigger_spans[example_idx]):]] = 3  # 可能存在trigger_span被错误预测成candidate spans的情况
            node_type[node_idx_list[0]] = 1
            node_type[node_idx_list[1:1 + len(other_trigger_spans[example_idx])]] = 2
            node_type_embedding = self.node_type_embedding(node_type)

            all_node_features.append(node_feature + node_type_embedding)

        node_features_big = torch.cat(all_node_features, dim=0)
        batched_graph = dgl.batch(all_graphs)

        feature_bank = [node_features_big]
        for gcn_layer in self.gcn_layers:
            node_features_big = gcn_layer(batched_graph, {"node": node_features_big})["node"]
            feature_bank.append(node_features_big)
        feature_bank = torch.cat(feature_bank, dim=-1)
        feature_bank = self.after_layer(feature_bank)  # all_node_num * hidsize

        cur_bias = 0
        # max_graph_span_num = node_labels.size(-1)
        all_span_features = []
        all_trigger_features = []
        for example_idx, (cur_span_info, node_idx_list) in enumerate(zip(all_span_infos, all_node_idx_list)):
            cur_node_num = cur_span_info.size(0)
            cur_features_bank = feature_bank[cur_bias:cur_bias + cur_node_num]  # cur_node_num * hidsize
            cur_bias += cur_node_num

            span_features = cur_features_bank[node_idx_list]
            all_span_features.append(span_features[1 + len(other_trigger_spans[example_idx]):])
            all_trigger_features.append(span_features[0].unsqueeze(0))
        all_span_features = nn.utils.rnn.pad_sequence(all_span_features, batch_first=True)  # bsz * max_span_num * hidsize
        all_trigger_features = torch.stack(all_trigger_features, dim=0)  # bsz * 1 * hidsize

        pair_features = torch.cat((all_span_features, all_trigger_features.expand_as(all_span_features)), dim=-1)
        if self.event_embedding is not None:
            pair_features = torch.cat(
                (pair_features, self.event_embedding(event_ids).unsqueeze(1).expand(-1, all_span_features.size(1), -1)),
                dim=-1)
        pair_logits = self.classifier(pair_features)
        if self.use_label_mask or not self.training:
            label_masks_expand = label_masks.unsqueeze(1).expand_as(pair_logits)
            pair_logits = pair_logits.masked_fill(label_masks_expand == 0, -1e4)

        if self.training:
            prune_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            span_scores = torch.cat(span_scores, dim=-1)
            span_labels = torch.cat(span_labels, dim=-1)
            prune_mask = (span_labels != -100)
            prune_loss = torch.sum(prune_loss_fct(span_scores, span_labels.float()) * prune_mask) / torch.sum(prune_mask)

            pair_labels = nn.utils.rnn.pad_sequence(pair_labels, batch_first=True, padding_value=-100)
            pair_loss = self.loss_fct(pair_logits.view(-1, self.num_labels), pair_labels.view(-1))
            loss = pair_loss + self.lambda_weight * prune_loss
            return {'loss': loss}

        # preds = torch.ones_like(prune_labels)
        # new_preds = torch.zeros_like(prune_labels)
        # index = nn.utils.rnn.pad_sequence(all_selected_indices, batch_first=True, padding_value=(new_preds.size(-1) - 1))
        # new_preds = new_preds.scatter(dim=-1, index=index, src=preds)
        # return {'prune_preds': new_preds, 'prune_labels': prune_labels}

        # preds = torch.argmax(pair_logits, dim=-1)  # bsz * span_num
        # new_preds = torch.zeros_like(labels)
        # # 把preds[i]按照all_selected_indices[i]塞入到new_preds[i]
        # index = nn.utils.rnn.pad_sequence(all_selected_indices, batch_first=True, padding_value=(new_preds.size(-1) - 1))
        # new_preds = new_preds.scatter(dim=-1, index=index, src=preds)
        # return {'preds': new_preds}

        # post-process
        new_preds = torch.zeros_like(labels)
        max_v, max_idx = pair_logits.max(-1)
        sort_idx = max_v.argsort(-1, descending=True)
        for i in range(len(all_span_infos)):
            mask = torch.zeros(input_ids[i].size(0))
            cur_span = spans[i]
            cur_indices = all_selected_indices[i]
            L = cur_indices.size(0)
            cur_preds = sort_idx[i, :L]
            preds = []
            for j in range(len(cur_preds)):
                if max_idx[i][cur_preds[j]] == 0:
                    continue
                a, b = cur_span[cur_indices[cur_preds[j]]]
                id_span = input_ids[i, a:b + 1]
                if mask[a:b + 1].sum() > 0:
                    continue
                else:
                    mask[a:b + 1] += 1
                    preds.append([id_span, min(abs(a - trigger_spans[i][0]), abs(b - trigger_spans[i][1])),
                                  cur_indices[cur_preds[j]], max_idx[i][cur_preds[j]]])
            dels = []
            for _i in range(len(preds) - 1, -1, -1):
                for j in range(_i):
                    if preds[j][0].sum() == preds[_i][0].float().sum():  # and preds[j][1]<preds[_i][1]:
                        dels.append(_i)
            for _i in range(len(preds)):
                if _i not in dels:
                    new_preds[i, preds[_i][2]] = preds[_i][3]
        return {'preds': new_preds}


class PieceWiseTrain(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_begin(self, trainer):
        if trainer.cur_epoch_idx == trainer.model.pivot_epoch:
            trainer.model.use_label_mask = False
            trainer.model.label_smoothing = trainer.model.predefined_label_smoothing  # 前n个epoch用CrossEntropy，后面用LabelSmoothing


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights
        self.gamma = gamma

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
