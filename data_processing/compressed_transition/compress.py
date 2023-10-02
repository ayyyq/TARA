import torch
import random
from tqdm import tqdm
SENSES = ['COUNTRY', 'QUANTITY', 'ORGANIZATION', 'DATE_ATTRS', 'NATIONALITY', 'LOCATION', \
    'ENTITY', 'MISC', 'ORDINAL_ENTITY', 'IDEOLOGY', 'RELIGION', 'STATE_OR_PROVINCE', 'CAUSE_OF_DEATH',\
    'TITLE', 'DATE', 'NUMBER', 'HANDLE', 'SCORE_ENTITY', 'DURATION', 'ORDINAL', 'MONEY', 'CRIMINAL_CHARGE', \
    'PERSON', 'THING', 'STATE', 'DATE-ENTITY', 'NAME', 'PUBLICATION', 'PROVINCE', 'GOVERNMENT-ORGANIZATION', 'CITY-DISTRICT', 'CITY', \
    'CRIMINAL-ORGANIZATION', 'GROUP', 'RELIGIOUS-GROUP', 'STRING-ENTITY', 'POLITICAL-PARTY', 'WORLD-REGION', 'COUNTRY-REGION', 'STRING-NAME', "URL-ENTITY", \
    'FESTIVAL', 'COMPANY', 'BOARDCAST-PROGRAM'
    ]


def in_edge(n, nodes, edges):
    ret = []
    for e in edges:
        if nodes[e[1]] == n:
            ret.append(e)
    return ret


def out_edge(n, nodes, edges):
    ret = []
    for e in edges:
        if nodes[e[0]] == n:
            ret.append(e)
    return ret


def graph2str(nodes, edges, token_str, root_str):
    ret = token_str + '\n'
    for n in nodes:
        ret = ret + '\t'.join(['# ::node', n[0], n[1], n[2] if n[2]!='0-0' else ' '])+'\n'
    if root_str is not None:
        ret = ret + root_str + '\n'
    for e in edges:
        st = nodes[e[0]]
        ed = nodes[e[1]]
        str1 = ['# ::edge',st[1], e[2], ed[1], st[0], ed[0]]
        ret = ret + '\t'.join(str1) + '\n'
    return ret.strip()


def str2graph(amr_str):
    nodes, edges = [], []
    token_str, root_str = None, None
    for line in amr_str.split('\n'):
        if line.startswith('# ::tok'):
            #tokens = line[7:].split()[:-1] # remove <ROOT>
            token_str = line.strip()
        if line.startswith('# ::node'):
            tt = line[8:].strip().split('\t')
            if len(tt)<3:
                node_id, node_str = tt
                node_pos = '0-0'
            else:
                node_id, node_str, node_pos = tt
            nodes.append([node_id, node_str, node_pos])
        if line.startswith('# ::edge'):
            edge_st, edge_rel, edge_ed, edge_stid, edge_edid = line[8:].strip().split('\t')
            st = [x[0] for x in nodes].index(edge_stid)
            ed = [x[0] for x in nodes].index(edge_edid)
            edges.append([st, ed, edge_rel])
        if line.startswith('# ::root'):
            root_str = line.strip()
    return nodes, edges, token_str, root_str


def update_graph(rm_edges, add_edges, amr_nodes, amr_edges, only_remove_dup=False):
    count = {}
    all_edges = [x for x in amr_edges + add_edges if x not in rm_edges]
    nmap = {}
    new_nodes = []
    st, ed = 0, 1
    que = [0]
    if not only_remove_dup:
        while st<ed:
            q = que[st]
            for e in all_edges:
                if q==e[0] and e[1] not in que:
                    que.append(e[1])
                    ed += 1
                if q==e[1] and e[0] not in que:
                    que.append(e[0])
                    ed += 1
            st+= 1
    for i, n in enumerate(amr_nodes):
        if i in que or only_remove_dup:
            if n in new_nodes:
                nmap[i] = new_nodes.index(n)
            else:
                nmap[i] = len(new_nodes)
                new_nodes.append(n)
    new_edges = []
    for i, e in enumerate(all_edges):
        if e[0] in nmap and e[1] in nmap:
            ee = [nmap[e[0]], nmap[e[1]], e[2]]
            if ee not in new_edges and ee[0]!=ee[1]:
                new_edges.append(ee)

    return new_nodes, new_edges


def post_process(amr_str):
    amr_nodes, amr_edges, token_str, root_str = str2graph(amr_str)
    rm_edges, add_edges = [], []
    for _i in range(len(amr_nodes)):
        n = amr_nodes[_i]
        new_label = []
        new_align = []
        new_cnt = 1 + max([int(x[0]) for x in amr_nodes])
        scanned = []

        if n[1].upper() in SENSES:

            edges = out_edge(n, amr_nodes, amr_edges)
            for e in edges:
                if 'name' in e[2]:
                    _edges = out_edge(amr_nodes[e[1]], amr_nodes, amr_edges)
                    if len(_edges)==0:
                        for ee in edges:
                            if ee[2] == 'wiki':
                                new_label.append(amr_nodes[ee[1]][1])
                                new_align.append(amr_nodes[ee[1]][2])
                    for _e in sorted(_edges, key=lambda x:x[2]): 
                        if _e[2] in ['op1', 'op2', 'op3', 'op4', 'op5', 'op6', 'op7', 'op8', 'op9']:
                            new_label.append(amr_nodes[_e[1]][1])
                            new_align.append(amr_nodes[_e[1]][2])
                    scanned.append(e[2])
                    break
                if len(new_label)>0:
                    break
        if len(new_label)>0:
            tt = []
            for x in new_align:
                a, b = x.split('-')
                tt.append(int(a))
                tt.append(int(b))
            new_node = [str(new_cnt), ' '.join(new_label), str(min(tt))+'-'+str(max(tt))]
            new_cnt += 1

            in_edges = in_edge(n, amr_nodes, amr_edges)
            out_edges = out_edge(n, amr_nodes, amr_edges)
            new_idx = len(amr_nodes)
            amr_nodes.append(new_node)
            for e in in_edges:
                rm_edges.append(e)
                add_edges.append([e[0], new_idx, e[2]])
            for e in out_edges:
                rm_edges.append(e)
                if e[2] not in scanned:
                    add_edges.append([new_idx, e[1], e[2]])

    for e in add_edges+amr_edges:
        if e[2] == 'wiki':
            rm_edges.append(e)

    amr_nodes, amr_edges = update_graph(rm_edges, add_edges, amr_nodes, amr_edges)    
    return graph2str(amr_nodes, amr_edges, token_str, root_str)


if __name__ == '__main__':
    for split in ['train', 'dev', 'test']:
        data = torch.load(f'data/wikievents/transition/amr-wikievent-{split}.pkl')
        compressed_amr_list = []
        len1, len2 = [], []
        for i, d in tqdm(enumerate(data)):
            dd = post_process(d)
            compressed_amr_list.append(dd)
            len1.append(len([x for x in d.strip().split('\n') if x[0] == '#']))
            len2.append(len(dd.split('\n')))
        assert len(data) == len(compressed_amr_list)
        torch.save(compressed_amr_list, f'data/wikievents/transition/compressed-amr-wikievent-{split}.pkl')
        print(sum(len1), sum(len2))
