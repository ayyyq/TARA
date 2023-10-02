import torch
import dgl
import json
from tqdm import tqdm
import re
from unidecode import unidecode


all_edge_type = {}


cmap = {'’':"'", '“': '"', '”': '"', '‘': '"', "–":'-', '—': '-', 'á':'a', 'ó': 'o', 'Ó': 'O', 'é':'e', '▶':'', 'Á': 'A', 'í':'i', 'Ú':'U', '…': '...', '¿': '.', 'г':'r'}
url_extract_pattern1 = "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
url_extract_pattern2 = "^[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
def convert(x):
    x = unidecode(x)
    tt = re.findall(url_extract_pattern1, x)
    tt = tt + [y for y in re.findall(url_extract_pattern2, x) if '/' in y or '@' in y]
    if len(set(tt)) > 0:
        for t in tt:
            x = x.replace(t, '<url>')
        # assert 5==6
    ret = ''
    for c in x:
        ret += cmap.get(c, c)
    return ret


def get_amr_edge_idx(edge_type_str):
    if edge_type_str in ['location', 'localtion-of', 'destination', 'path']:
        return 0
    elif edge_type_str in ['year', 'time', 'duration', 'decade', 'weekday']:
        return 1
    elif edge_type_str in ['instrument', 'manner', 'poss', 'topic', 'medium', 'duration']:
        return 2
    elif edge_type_str in ['mod']:
        return 3
    elif edge_type_str.startswith('prep-'):
        return 4
    elif edge_type_str.startswith('op') and edge_type_str[-1].isdigit():
        return 5
    elif edge_type_str.startswith('snt') or edge_type_str == 'NSENT':
        return 6
    elif edge_type_str == 'ARG0':
        return 7
    elif edge_type_str == 'ARG1':
        return 8
    elif edge_type_str == 'ARG2':
        return 9
    elif edge_type_str == 'ARG3':
        return 10
    elif edge_type_str == 'ARG4':
        return 11
    else:
        return 12


def processing_amr(data, amr_list):
    """
    把每个句子变成一个dglgraph
    编号0的是root节点
    ndata['span']放的是对应的word-level的span信息
    """

    graphs_list = []
    # cur_idx = 0
    initial_graph = {}
    for i in range(13):
        initial_graph[('node', str(i), 'node')] = ([], [])

    global all_edge_type
    assert len(data) == len(amr_list)
    for sentences, amr in tqdm(zip(data, amr_list)):
        # amrs是document-level AMR
        # amrs = amr_list[cur_idx:cur_idx + len(sentences)]
        # cur_idx += len(sentences)

        graph = dgl.heterograph(initial_graph)
        num_nodes = len(amr['nodes'])
        sent_start_nodes = []
        graph.add_nodes(num=num_nodes)
        for edge in amr['edges']:
            edge_start, edge_end, amr_edge_type = edge
            amr_edge_type = amr_edge_type.replace('-of', '').replace('(', '')
            if amr_edge_type == '':  # 这肯定不是一个合理的AMR edge吧
                continue
            # check if the edge type is "ARGx-of", if so, reverse the direction of the edge
            # if amr_edge_type.startswith("ARG") and amr_edge_type.endswith("-of"):
            #     edge_start, edge_end = edge_end, edge_start
            #     amr_edge_type = amr_edge_type[:4]
            if amr_edge_type == 'NSENT':
                if edge_start >= num_nodes or edge_end >= num_nodes:
                    continue  # 去掉dummy node
                if edge_start not in sent_start_nodes:
                    sent_start_nodes.append(edge_start)
                if edge_end not in sent_start_nodes:
                    sent_start_nodes.append(edge_end)
                continue
            # deal with this edge here
            edge_type = str(get_amr_edge_idx(amr_edge_type))
            if amr_edge_type not in all_edge_type:
                all_edge_type[amr_edge_type] = 0
            all_edge_type[amr_edge_type] += 1

            # forward
            graph.add_edges(u=edge_start, v=edge_end, etype=edge_type)
            # also backward
            graph.add_edges(u=edge_end, v=edge_start, etype=edge_type)

        amr_edge_type = 'NSENT'
        edge_type = str(get_amr_edge_idx(amr_edge_type))
        for root_i in sent_start_nodes:
            for root_j in sent_start_nodes:
                if root_i != root_j:
                    graph.add_edges(u=root_i, v=root_j, etype=edge_type)
                    if amr_edge_type not in all_edge_type:
                        all_edge_type[amr_edge_type] = 0
                    all_edge_type[amr_edge_type] += 1

        # 由于AMR node不一定对应连续的span，所以不把对应的text作为node feature [x]
        # 只取连续的span作为AMR node对应的text
        context = [convert(token).strip() for sent in sentences for token in sent]
        # assert len(context) == len(amr['tokens'])
        amr2context = []
        if len(context) < len(amr['tokens']):  # 由于在parse AMR时至多对amr['tokens']的每一个token做了unidecode，所以不影响相对位置
            aidx = 0
            for cidx in range(len(context)):
                if context[cidx] == amr['tokens'][aidx]:
                    amr2context.append(cidx)
                    aidx += 1
                else:
                    amr_token = []
                    while ''.join(amr_token).replace(' ', '') != context[cidx].replace(' ', ''):
                        amr_token.append(amr['tokens'][aidx])
                        amr2context.append(cidx)
                        aidx += 1
                    while len(amr['tokens'][aidx]) == 0:
                        amr2context.append(cidx)
                        aidx += 1
            while aidx < len(amr['tokens']) and len(amr['tokens'][aidx]) == 0:
                amr2context.append(cidx)
                aidx += 1
        elif len(context) > len(amr['tokens']):
            # TODO: check
            print('check')
            cidx = 0
            for aidx in range(len(amr['tokens'])):
                if context[cidx] == amr['tokens'][aidx]:
                    amr2context.append(cidx)
                    cidx += 1
                else:
                    while context[cidx] != amr['tokens'][aidx]:
                        cidx += 1
                    amr2context.append(cidx)
                    cidx += 1
        if amr2context:
            assert len(amr2context) == len(amr['tokens'])
        amrnode_info = []
        for node in amr['nodes']:
            info = {'node': node[0], 'span': []}
            if isinstance(node[1], list):
                # info['span'] = [x for x in node[1] if x != -1]  # TODO: 对于-1的node是应该随机初始化还是用cls？
                index = [x for x in node[1] if x != 1]
                if len(index) == 0:
                    info['span'] = [-1, -1]
                elif len(index) == 1:
                    info['span'] = [index[0], index[0]]
                else:
                    index.sort()
                    start = end = index[0]
                    for i in range(1, len(index)):
                        if index[i] - index[i - 1] == 1:
                            end = index[i]
                    info['span'] = [start, end]  # index可能不是连续的
            else:
                info['span'] = [node[1], node[1]]
            if amr2context:
                info['span'][0] = amr2context[info['span'][0]] if info['span'][0] != -1 else info['span'][0]
                info['span'][1] = amr2context[info['span'][1]] if info['span'][1] != -1 else info['span'][1]
            amrnode_info.append(info)
        # amrnode_info.append({'node': 'dummy', 'test': []})  # TODO: NSENT只连在了相邻两个句子上
        assert len(amrnode_info) == graph.num_nodes()

        graphs_list.append({'graph': graph, 'amrnode_info': amrnode_info})

    print(all_edge_type)
    return graphs_list


def amr2dglgraph(data_path, amr_path, graph_path):
    data = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            sentences = d['sentences']
            data.append(sentences)  # 每个example的sentence
    amr = torch.load(amr_path)  # 所有的sentence
    graphs_list = processing_amr(data, amr['data'])
    torch.save(graphs_list, graph_path)
    print('save to', graph_path)


if __name__ == "__main__":
    amr2dglgraph("data/wikievents/transfer-train.jsonl", "data/wikievents/amrbart/train_save_result.pt0",
                 "data/wikievents/amrbart/amrbart-dglgraph-train.pkl")
    amr2dglgraph("data/wikievents/transfer-dev.jsonl", "data/wikievents/amrbart/dev_save_result.pt0",
                 "data/wikievents/amrbart/amrbart-dglgraph-dev.pkl")
    amr2dglgraph("data/wikievents/transfer-test.jsonl", "data/wikievents/amrbart/test_save_result.pt0",
                 "data/wikievents/amrbart/amrbart-dglgraph-test.pkl")

    # amr2dglgraph("data/rams/train.jsonlines", "data/rams/amrbart/train_save_result.pt0",
    #              "data/rams/amrbart/amrbart-dglgraph-train.pkl")
    # amr2dglgraph("data/rams/dev.jsonlines", "data/rams/amrbart/dev_save_result.pt0",
    #              "data/rams/amrbart/amrbart-dglgraph-dev.pkl")
    # amr2dglgraph("data/rams/test.jsonlines", "data/rams/amrbart/test_save_result.pt0",
    #              "data/rams/amrbart/amrbart-dglgraph-test.pkl")
