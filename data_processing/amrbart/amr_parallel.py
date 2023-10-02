from amr import *
import tqdm
import time
import os


SENSES = ['COUNTRY', 'QUANTITY', 'ORGANIZATION', 'DATE_ATTRS', 'NATIONALITY', 'LOCATION', \
    'ENTITY', 'MISC', 'ORDINAL_ENTITY', 'IDEOLOGY', 'RELIGION', 'STATE_OR_PROVINCE', 'CAUSE_OF_DEATH',\
    'TITLE', 'DATE', 'NUMBER', 'HANDLE', 'SCORE_ENTITY', 'DURATION', 'ORDINAL', 'MONEY', 'CRIMINAL_CHARGE', \
    'PERSON', 'THING', 'STATE', 'DATE-ENTITY', 'NAME', 'PUBLICATION', 'PROVINCE', 'GOVERNMENT-ORGANIZATION', 'CITY-DISTRICT', 'CITY', \
    'CRIMINAL-ORGANIZATION', 'GROUP', 'RELIGIOUS-GROUP', 'STRING-ENTITY', 'POLITICAL-PARTY', 'WORLD-REGION', 'COUNTRY-REGION', 'STRING-NAME', "URL-ENTITY", \
    'FESTIVAL', 'COMPANY', 'BOARDCAST-PROGRAM'
    ]


EMB_ID_UNK = 0
EMB_ID_LB = 1
EMB_ID_PNEW = 2
EMB_ID_POLD = 3
EMB_ID_SENSE = 4
EMB_ID_RB = 5
EMB_ID_REL = 6
EMB_ID_LLIT = 7
EMB_ID_RLIT = 8
gpdict = {}
gidict = {}
import pydot


def draw_graph(nodes, edges, tokens, fname):
    global gdict, gidict  
    graph = pydot.Dot('my_graph', graph_type='graph', bgcolor='white')
    for _i, _n in enumerate(nodes):
        if isinstance(_n[1], list):
            text = ' '.join([tokens[x] for x in _n[1] if x!=-1])
        else:
            text = tokens[_n[1]] if _n[1]!=-1 else ' '
        gpdict[_n[0]+' | '+text] = gpdict.get(_n[0]+' | '+text, 0) + 1
        gidict[_n[0]] = gidict.get(_n[0], 0) + 1
        gidict[text] = gidict.get(text, 0) + 1 
        graph.add_node(pydot.Node(str(_i), shape='ellipse', label=_n[0]+'\n'+text))

    for _e in edges: 
        graph.add_edge(pydot.Edge(str(_e[0]), str(_e[1]), color='blue', label=str(_e[2])))
    text = ''
    last = 0
    for t in tokens:
        if len(text)-last<40:
            text += t+' '
        else:
            text += t+'\n'
            last = len(text)
    graph.add_node(pydot.Node('9999', label=str(len(nodes))+'_'+str(len(edges))+text))
    graph.write_png('imgs/{0:}.png'.format(fname), prog='fdp')


def within(x, arrs):
    for arr in arrs:
        if len(arr)==1:
            if x == arr[0]:
                return True
        if len(arr)==2:
            if x>=arr[0] and x<arr[1]:
                return True
    return False


class MinLengthLogitsProcessor(object):
    def __init__(self, tokenizer, input_ids):
        self.tokenizer = tokenizer
        self.rules = {
            EMB_ID_LB: [EMB_ID_PNEW, EMB_ID_POLD],
            EMB_ID_PNEW: [EMB_ID_SENSE],
            EMB_ID_POLD: [EMB_ID_RB, EMB_ID_REL],
            EMB_ID_SENSE: [EMB_ID_SENSE, EMB_ID_REL, EMB_ID_RB, EMB_ID_RLIT],
            EMB_ID_RB: [EMB_ID_RB, EMB_ID_REL],
            EMB_ID_REL: [EMB_ID_LB, EMB_ID_LLIT, EMB_ID_PNEW, EMB_ID_POLD],
            EMB_ID_LLIT: [EMB_ID_SENSE],
            EMB_ID_RLIT: [EMB_ID_RB]
        }
        self.basic_mask = [[53842], [53843], [0, 10], [53054, 53069]]
        self.score_masks = {
            EMB_ID_LB: [[36]],
            EMB_ID_RB: [[4839]],
            EMB_ID_PNEW : [[53069,53581]],
            EMB_ID_POLD : [],
            EMB_ID_REL: [[52938, 53032]],
            EMB_ID_LLIT: [[53838]],
            EMB_ID_RLIT: [[53839]],
            EMB_ID_SENSE: [[53032, 53054], [50265, 52938], [10,36], [37, 4839], [4840, 50000]],
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i, b in enumerate(input_ids):
            if len(b)<2:
                continue
            if within(b[-1].item(), self.basic_mask):
                continue
            find_k = None
            for k,v in self.score_masks.items():
                if k!=EMB_ID_SENSE:
                    if within(b[-1].item(), v):
                        find_k = k
                        break
            if find_k is None:
                find_k = EMB_ID_SENSE
                continue
            next_k = self.rules[find_k]
            allow_mask = sum([self.score_masks[k] for k in next_k], [])+self.basic_mask
        return scores


def parse_amr_skip(amrs):
    stack = []
    nodes = []
    visit = []
    edges = []
    node_cnt = 0
    fcnt = 0
    buffer = []
    pointers = {}
    i = 0
    lit_flag = False
    while i < len(amrs):
        w = amrs[i]
        if len(w)<1:
            i += 1
            continue
        if w in ['</s>', '<AMR>', '</AMR>', '<s>', '<lit>']:
            if w == '<lit>':
                lit_flag = True
            i += 1
            continue
        if w == '</lit>':
            lit_flag = False

        if w[0]==':':
            buffer.append([':', w, None])
        elif w[0]=='<' and w[-1]=='>' and w not in ['</lit>']:
            if w in pointers:
                buffer.append(['c', w, None])
            else:
                buffer.append(['nc', w, None])
        elif w in ['(', ')']:
            if w==')':
                if len(stack)>0:
                    stack = stack[:-1]
        elif w not in ['</lit>']:
            if w == 'multi-sentence':
                buffer.append(['t', w, None])
            else:
                if lit_flag and len(buffer)>0 and buffer[-1][0]=='t':
                    buffer[-1].extend([w, None])
                else:
                    buffer.append(['t', w, None])
        if len(buffer)>0 and not lit_flag:
            if len(buffer)>2 and buffer[-1][0]=='t' and buffer[-2][0] in ['nc'] and buffer[-3][0]==':':
                if not (buffer[-1][1] in STOPs):
                    n = [buffer[-1][1], buffer[-1][2]]
                    if len(stack)>0:
                        e = [stack[-1], node_cnt, buffer[-3][1][1:]]
                        edges.append(e)
                    stack.append(node_cnt)
                    pointers[buffer[-2][1]] = node_cnt
                    nodes.append(n)
                    node_cnt += 1
                buffer = []
            if len(buffer)>1 and buffer[-1][0] in ['c'] and buffer[-2][0]==':':
                if len(stack)>0:
                    e = [stack[-1], pointers[buffer[-1][1]], buffer[-2][1][1:]]
                    edges.append(e)
                buffer = []
            if len(buffer)>1 and buffer[-1][0] == 't' and buffer[-2][0] in ['nc']:
                if not (buffer[-1][1] in STOPs):
                    n = [buffer[-1][1], buffer[-1][2]]

                    nodes.append(n)
                    stack.append(node_cnt)
                    pointers[buffer[-2][1]] = node_cnt
                    node_cnt += 1
                buffer = []
            if len(buffer)>1 and buffer[-1][0] == 't' and buffer[-2][0] in [':']:
                _align = [buffer[-1][_i+1] for _i in range(1, len(buffer[-1]), 2)]
                n = [' '.join([buffer[-1][_i] for _i in range(1, len(buffer[-1]), 2)]), _align]
                nodes.append(n)

                if len(stack)>0:
                    e = [stack[-1], node_cnt, buffer[-2][1][1:]]
                    edges.append(e)
                node_cnt += 1
                buffer = []
        i += 1
    if len(buffer)>0:
        if len(buffer)>2 and buffer[-1][0]=='t' and buffer[-2][0] in ['nc'] and buffer[-3][0]==':':
            if not (buffer[-1][1] in STOPs or buffer[-1][2]==-1):
                n = [buffer[-1][1], buffer[-1][2]]
                if len(stack)>0:
                    e = [stack[-1], node_cnt, buffer[-3][1][1:]]
                    edges.append(e)
                pointers[buffer[-2][1]] = node_cnt
                nodes.append(n)
                node_cnt += 1
            buffer = []
        if len(buffer)>1 and buffer[-1][0] in ['c'] and buffer[-2][0]==':':
            if len(stack)>0:
                e = [stack[-1], pointers[buffer[-1][1]], buffer[-2][1][1:]]
                edges.append(e)
            buffer = []
        if len(buffer)>1 and buffer[-1][0] == 't' and buffer[-2][0] in ['nc']:
            if not (buffer[-1][1] in STOPs):
                n = [buffer[-1][1], buffer[-1][2]]
                nodes.append(n)
                pointers[buffer[-2][1]] = node_cnt
                node_cnt += 1
            buffer = []

    return nodes, edges


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


draw_cnt = 0


class AMRParser(object):
    def __init__(self, device):
        self.tokenizer = PENMANBartTokenizer.from_pretrained(
            "facebook/bart-large",
            collapse_name_ops=False,
            use_pointer_tokens=True,
            raw_graph=False,
        )
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.model = AutoModelForSeq2SeqLM.from_pretrained("xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing") 
        self.model.config.eos_token_id = self.tokenizer.amr_eos_token_id
        self.model.config.forced_eos_token_id = self.tokenizer.amr_eos_token_id
        self.model.config.decoder_start_token_id = self.tokenizer.amr_bos_token_id
        self.model = self.model.to(device)
        self.device = device
        self.word_dict = {}

        self.rules = {}
        with open('data_processing/amrbart/v1.rules', 'r') as f:
            for line in f.readlines():
                if '||' in line:
                    if len(line.split('||'))==2:
                        a, b = line.split('||')
                        a, b = a.strip(), b.strip()
                        if a not in self.rules:
                            self.rules[a] = {}
                        self.rules[a][b] = 'NEG'
                else:
                    if len(line.split('|'))==2:
                        a, b = line.split('|')
                        a, b = a.strip(), b.strip()
                        if a not in self.rules:
                            self.rules[a] = {}

                        if len(b)>0 and b!='NONE':
                            self.rules[a][b] = 'Strong POS'
                        elif b=='NONE':
                            self.rules[a][b] = 'NEG'

    def __call__(self, batch_inp, skip_align=False):
        log1 = MinLengthLogitsProcessor(self.tokenizer,None)
        punkts = [',', '.', '!', '?', ':']
        batch = []
        for inp in batch_inp:
            if isinstance(inp, list):
                inp = ' '.join(inp)
            else:
                tinp = []
                for x in inp.split():
                    if x[-1] in punkts:
                        tinp.append(x[:-1])
                        tinp.append(x[-1])
                    else:
                        tinp.append(x)
                inp = ' '.join(tinp)
            _inp = self.tokenizer([inp])['input_ids'][0]        
            _inp = torch.LongTensor(_inp + [self.tokenizer.amr_bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.amr_eos_token_id])            
            batch.append(_inp)
        max_len = max([len(x) for x in batch])
        batch = [torch.cat([x, self.tokenizer.pad_token_id+torch.zeros(max_len-len(x)).long()],0) for x in batch]
        _inp = torch.stack(batch,0).to(self.device)
        if skip_align:
            ret = self.model.generate(_inp, num_beams=1, length_penalty=1.0, max_length=768, return_dict_in_generate=True, logits_processor=[log1])
        else:
            ret = self.model.generate(_inp, num_beams=1, length_penalty=1.0, max_length=768, output_attentions=True, return_dict_in_generate=True, logits_processor=[log1])#, min_length=0, no_repeat_ngram_size=0)
        all_res = []
        for i in range(len(ret['sequences'])):
            tt = self.tokenizer.convert_ids_to_tokens(ret['sequences'][i])
            inps = self.tokenizer.convert_ids_to_tokens(_inp[i])
            if skip_align:
                last_str = []
                last_score = []
                inp_splits = []
                last = -1
                show_inp = []
                for i, x in enumerate(inps[1:-4]):
                    if x[0]=='Ġ':
                        if last>=0:
                            inp_splits.append([last, i])
                            show_inp.append(inps[1:-4][last:i])
                            last = -1
                    if x=='-':
                        if last>=0:
                            inp_splits.append([last, i])
                            show_inp.append(inps[1:-4][last:i])
                            last = -1
                        inp_splits.append([i, i+1])
                        show_inp.append(['-'])
                        last = i+1

                    if last == -1:
                        last = i
                if last>=0:
                    inp_splits.append([last, len(inps[1:-4])])
                    show_inp.append(inps[1:-4][last:])
                amrs = []
                for i in range(len(tt)-1): 
                    if tt[i][0]=='Ġ':
                        if len(last_str)>0:
                            amrs.append(''.join(last_str))
                            last_str = []
                        last_str.append(tt[i][1:])
                    else:
                        last_str.append(tt[i])
                if len(last_str)>0:
                    amrs.append(''.join(last_str))

                for j in range(len(amrs)):
                    amrs[j] = amrs[j].replace('âĢĿ', '"').replace('âĢĻ', "'")
                tokens = [''.join(x).replace('Ġ', '').replace('âĢĿ', '"').replace('âĢĻ', "'")  for x in show_inp]
                amr_nodes, amr_edges = parse_amr_skip(amrs)
                all_res.append({'nodes': amr_nodes, 'edges': amr_edges, 'tokens': tokens})
            else:
                last_str = []
                last_score = []
                inp_splits = []
                last = -1
                show_inp = []
                for i, x in enumerate(inps[1:-4]):
                    if x[0]=='Ġ':
                        if last>=0:
                            inp_splits.append([last, i])
                            show_inp.append(inps[1:-4][last:i])
                            last = -1
                    if x=='-':
                        if last>=0:
                            inp_splits.append([last, i])
                            show_inp.append(inps[1:-4][last:i])
                            last = -1
                        inp_splits.append([i, i+1])
                        show_inp.append(['-'])
                        last = i+1

                    if last == -1:
                        last = i
                if last>=0:
                    inp_splits.append([last, len(inps[1:-4])])
                    show_inp.append(inps[1:-4][last:])
                cor_matrix = []
                for i in range(len(tt)-1): 
                    step = ret['cross_attentions'][i]
                    ss = step[0][0].sum([0,1])[1:-4]#.cpu()
                    ss = (ss/ss.sum()).cpu().numpy()
                    if tt[i][0]=='Ġ':
                        if len(last_str)>0:
                            cor_matrix.append([''.join(last_str), np.array(last_score).mean(0)])
                            last_str = []
                            last_score = []
                        last_str.append(tt[i][1:])
                    else:
                        last_str.append(tt[i])
                    _score = []
                    for sp in inp_splits:
                        _score.append(sum(ss[sp[0]:sp[1]]))
                    last_score.append(_score)
                if len(last_str)>0:
                    cor_matrix.append([''.join(last_str), np.array(last_score).mean(0)])
                for i, x in enumerate(show_inp):
                    if len(x)==0 or ''.join(x)[-1] in punkts:
                        for j in range(len(cor_matrix)):
                            cor_matrix[j][1][i] -= 100
                if False:
                    for row in cor_matrix:
                        print(row[0], np.round(row[1], 2))

                tokens = [''.join(x).replace('Ġ', '').replace('âĢĿ', ')').replace('âĢĻ', "'").replace('âĢľ', "(")  for x in show_inp]
                tokens = [x for x in tokens if x not in ['</s>', '<AMR>', '<mask>', '</AMR>', '<pad>']]
                for j in range(len(cor_matrix)):
                    cor_matrix[j][0] = cor_matrix[j][0].replace('âĢĿ', ')').replace('âĢĻ', "'").replace('âĢľ', "(")
                    cor_matrix[j][1] = cor_matrix[j][1][:len(tokens)]
                    if cor_matrix[j][0]=='<pad>':
                        cor_matrix = cor_matrix[:j]
                        break
                amr_nodes, amr_edges = parse_amr(cor_matrix, tokens, self.word_dict, self.rules)

                rm_edges, add_edges = [], []
                for _i in range(len(amr_nodes)):
                    n = amr_nodes[_i]
                    new_label = []
                    new_align = []
                    if n[0] in ['date-entity', 'percentage-entity', 'date-interval', 'mass-quantity', 'distance-quantity']:
                        edges = out_edge(n, amr_nodes, amr_edges)
                        for e in edges:
                            new_label.append(e[2] + ' ' + amr_nodes[e[1]][0])
                            if isinstance(amr_nodes[e[1]][1], list):
                                new_align.extend(amr_nodes[e[1]][1])
                            else:
                                new_align.append(amr_nodes[e[1]][1])
                    elif n[0] in ['monetary-quantity', 'temporal-quantity', 'truth-value']:
                        edges = out_edge(n, amr_nodes, amr_edges)
                        for e in edges:
                            new_label.append(amr_nodes[e[1]][0])
                            if isinstance(amr_nodes[e[1]][1], list):
                                new_align.extend(amr_nodes[e[1]][1])
                            else:
                                new_align.append(amr_nodes[e[1]][1])
                        new_label = list(sorted(new_label))
                    elif n[0] == 'ordinal-entity':
                        edges = out_edge(n, amr_nodes, amr_edges)
                        for e in edges:
                            new_label.append(amr_nodes[e[1]][0])
                            if isinstance(amr_nodes[e[1]][1], list):
                                new_align.extend(amr_nodes[e[1]][1])
                            else:
                                new_align.append(amr_nodes[e[1]][1])
                        new_label = list(sorted(new_label))
                    elif n[0].upper() in SENSES:
                        edges = out_edge(n, amr_nodes, amr_edges)
                        for e in edges:
                            if 'name' in e[2]:
                                _edges = out_edge(amr_nodes[e[1]], amr_nodes, amr_edges)
                                if len(_edges)==0:
                                    for ee in edges:
                                        if ee[2] == 'wiki':
                                            new_label.append(amr_nodes[ee[1]][0])
                                            if isinstance(amr_nodes[ee[1]][1], list):
                                                new_align.extend(amr_nodes[ee[1]][1])
                                            else:
                                                new_align.append(amr_nodes[ee[1]][1])
                                for _e in sorted(_edges, key=lambda x:x[2]): 
                                    if _e[2] in ['op1', 'op2', 'op3', 'op4', 'op5', 'op6', 'op7', 'op8', 'op9']:
                                        new_label.append(amr_nodes[_e[1]][0])
                                        if isinstance(amr_nodes[_e[1]][1], list):
                                            new_align.extend(amr_nodes[_e[1]][1])
                                        else:
                                            new_align.append(amr_nodes[_e[1]][1])
                                break
                            if len(new_label)>0:
                                break
                    if len(new_label)>0:
                        if all([x==-1 for x in new_align]):
                            new_node = [' '.join(new_label), -1]
                        else:
                            new_node = [' '.join(new_label), list(sorted(list(set([x for x in new_align if x!=-1]))))]

                        in_edges = in_edge(n, amr_nodes, amr_edges)
                        out_edges = out_edge(n, amr_nodes, amr_edges)
                        new_idx = len(amr_nodes)
                        amr_nodes.append(new_node)
                        for e in in_edges:
                            rm_edges.append(e)
                            add_edges.append([e[0], new_idx, e[2]])
                        for e in out_edges:
                            rm_edges.append(e)
                            if e[2]!='name':
                                add_edges.append([new_idx, e[1], e[2]])

                for e in add_edges+amr_edges:
                    if e[2] == 'wiki':
                        rm_edges.append(e)

                amr_nodes, amr_edges = update_graph(rm_edges, add_edges, amr_nodes, amr_edges)
                all_res.append({'nodes': amr_nodes, 'edges': amr_edges, 'tokens': tokens})
        return all_res


def batch_inference(data_dir, proc_id, devices, batch_size, trunc_length, extra_coref=True, split=None):
    device = devices[proc_id]
    model = AMRParser(device)
    data = torch.load(os.path.join(data_dir, split + '_' + 'tmp_data.pt' + str(proc_id)))
    docs = data['docs']
    if extra_coref:
        doc_ids = data['doc_ids']
    else:
        doc_ids = np.arange(len(docs))
    batchs = []
    _batch = []
    assert len(docs) == len(doc_ids)
    print(doc_ids[-1] + 1, len(docs))
    for d, did in zip(docs, doc_ids):
        if len(d) > trunc_length:
            # 为了便于align，不再trunct，而是对于过长的句子直接减小batch_size以防止OOM
            if _batch:
                batchs.append(_batch)
                _batch = []

            if extra_coref:
                _batch.append([d, did])
            else:
                _batch.append(d)
            batchs.append(_batch)
            _batch = []
            continue

        if extra_coref:
            _batch.append([d, did])
        else:
            _batch.append(d)
        if len(_batch) == batch_size:
            batchs.append(_batch)
            _batch = []
    if _batch:
        batchs.append(_batch)
    tq = tqdm.tqdm(batchs) if proc_id == 0 else batchs
    ret = []
    buff = []
    for i, batch in enumerate(tq):
        _rets = model([x[0] for x in batch], skip_align=False)
        if extra_coref:
            for r, (_d, did) in zip(_rets, batch):
                if r['tokens'] != _d:
                    print(did)
                if len(buff) == 0 or buff[-1][0] == did:
                    buff.append([did, r])
                else:
                    token_offset = 0
                    node_offset = 0
                    m_nodes, m_edges, m_tokens = [], [], []
                    for d in [x[1] for x in buff]:
                        for n in d['nodes']:
                            if isinstance(n[1], list):
                                for _i in range(len(n[1])):
                                    n[1][_i] += token_offset if n[1][_i] != -1 else 0
                            else:
                                n[1] += token_offset if n[1] != -1 else 0
                        for _i in range(len(d['edges'])):
                            e = d['edges'][_i]
                            d['edges'][_i] = [e[0] + node_offset, e[1] + node_offset, e[2]]
                        m_edges.append([node_offset, node_offset + len(d['nodes']), 'NSENT'])
                        token_offset += len(d['tokens'])
                        node_offset += len(d['nodes'])
                        m_nodes.extend(d['nodes'])
                        m_edges.extend(d['edges'])
                        m_tokens.extend(d['tokens'])
                    # draw_graph(m_nodes, m_edges, m_tokens, str(i) + 'cc')
                    ret.append({'nodes': m_nodes, 'edges': m_edges, 'tokens': m_tokens, 'did': buff[0][0]})
                    buff = [[did, r]]
        else:
            ret.extend(_rets)

    if buff:
        token_offset = 0
        node_offset = 0
        m_nodes, m_edges, m_tokens = [], [], []
        for d in [x[1] for x in buff]:
            for n in d['nodes']:
                if isinstance(n[1], list):
                    for _i in range(len(n[1])):
                        n[1][_i] += token_offset if n[1][_i] != -1 else 0
                else:
                    n[1] += token_offset if n[1] != -1 else 0
            for _i in range(len(d['edges'])):
                e = d['edges'][_i]
                d['edges'][_i] = [e[0] + node_offset, e[1] + node_offset, e[2]]
            m_edges.append([node_offset, node_offset + len(d['nodes']), 'NSENT'])
            token_offset += len(d['tokens'])
            node_offset += len(d['nodes'])
            m_nodes.extend(d['nodes'])
            m_edges.extend(d['edges'])
            m_tokens.extend(d['tokens'])
        # draw_graph(m_nodes, m_edges, m_tokens, str(i) + 'cc')
        ret.append({'nodes': m_nodes, 'edges': m_edges, 'tokens': m_tokens, 'did': buff[0][0]})

    if split is not None:
        save_path = os.path.join(data_dir, split + '_' + 'save_result.pt' + str(proc_id))
        torch.save({'data': ret}, save_path)
        print('save to', save_path)
    else:
        save_path = os.path.join(data_dir, 'save_result.pt' + str(proc_id))
        torch.save({'data': ret}, save_path)
        print('save to', save_path)


from unidecode import unidecode


def single_gpu_batch_inference(data_dir, docs, batch_size=64, trunc_length=60, sent_tok=False, split=None):
    from spacy.lang.en import English
    nlp = English()
    nlp.add_pipe("sentencizer")
    _docs = []
    doc_ids = []

    cmap = {'’': "'", '“': '"', '”': '"', '‘': '"', "–": '-', '—': '-', 'á': 'a', 'ó': 'o', 'Ó': 'O', 'é': 'e', '▶': '',
            'Á': 'A', 'í': 'i', 'Ú': 'U', '…': '...', '¿': '.', 'г': 'r'}
    url_extract_pattern1 = "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
    url_extract_pattern2 = "^[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"

    def convert(x):
        x = unidecode(x)
        tt = re.findall(url_extract_pattern1, x) 
        tt = tt + [y for y in re.findall(url_extract_pattern2, x) if '/' in y or '@' in y]
        if len(set(tt)) > 0:
            for t in tt:
                x = x.replace(t, '<url>')
        ret = ''
        for c in x:
            ret += cmap.get(c, c)
        assert isinstance(ret, str)
        return ret

    all_sent_num = 0
    for i, doc in enumerate(docs):
        if sent_tok:
            for sent in nlp(doc).sents:
                _docs.append([convert(x.text) for x in sent])
                doc_ids.append(i)
        else:
            # 已有句子划分，不需要再划分句子
            for sent in doc:
                all_sent_num += 1
                _docs.append([convert(x) for x in sent])
                doc_ids.append(i)
    print('ALL_SENT_NUM', all_sent_num)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    torch.save({'docs': _docs, 'doc_ids': doc_ids}, os.path.join(data_dir, split + '_' + 'tmp_data.pt0'))
    batch_inference(data_dir, 0, ['cuda:0'], batch_size=batch_size, trunc_length=trunc_length, extra_coref=True, split=split)


if __name__ == '__main__':
    data = []
    import json
    for split in ['train', 'dev', 'test']:
        with open('data/wikievents/transfer-{}.jsonl'.format(split), 'r') as f:
            for line in f:
                example = json.loads(line)
                data.append(example['sentences'])
        print('Total', len(data))

        single_gpu_batch_inference('data/wikievents/amrbart', data[:10], batch_size=16, trunc_length=100, sent_tok=False, split=split)
