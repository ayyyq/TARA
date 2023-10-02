# coding:utf-8
import copy
import sys
from pathlib import Path
import pydot

STOPs = ['', '.', ',', 'of', 'the', 'in', 'at', 'to', 'is', 'be', 'passage', '<pad>', '(', ')', '-', 'on']
ROOT = Path(__file__).parent.parent.parent
print(ROOT)

import numpy as np
import torch
from transformers import BartTokenizer
import random

import penman
from penman import load as load_, Graph, Triple
from penman import loads as loads_
from penman import encode as encode_
from penman.model import Model
from penman.models.noop import NoOpModel
from penman.models import amr

import abc
import itertools
from collections import deque, defaultdict, Counter
import re
import regex as ree
from typing import List, Optional, Dict, Any, Set, TypeVar

from cached_property import cached_property
from dataclasses import dataclass
import networkx as nx
import enum


BACKOFF = penman.Graph(
    [
        penman.Triple("d2", ":instance", "dog"),
        penman.Triple("b1", ":instance", "bark-01"),
        penman.Triple("b1", ":ARG0", "d2"),
    ]
)


def commonAnc(str1, str2):
    pt = 0
    while pt<len(str1) and pt<len(str2):
        if str1[pt]!=str2[pt]:
            break
        pt += 1
    return pt 


def token_processing(tok):
    if tok is None:
        return None
    elif tok.isdigit():
        try:
            return eval(tok)
        except:
            return tok
    elif tok.startswith('"') and (not tok.endswith('"')):
        return tok + '"'
    elif tok.endswith('"') and (not tok.startswith('"')):
        return '"' + tok
    else:
        return tok


def decode_into_node_and_backreferences(subtoken_ids, tokenizer):
    rex_arg = re.compile(f"^{tokenizer.INIT}(op|snt|conj|prep)")
    rex_spc = re.compile(r"<(s|/s|lit|/lit|stop|unk|pad|mask)>")

    # get strings
    subtokens = [tokenizer.decoder.get(t) for t in subtoken_ids]
    # fix backreferences
    subtoken_backreferences = [max(t - len(tokenizer.encoder), -1) for t in subtoken_ids]
    # strip padding
    subtokens, subtoken_backreferences = zip(
        *[
            (s, b)
            for s, b in zip(subtokens, subtoken_backreferences)
            if s != (tokenizer.INIT + "<pad>")
        ]
    )

    # subword collapse
    tokens = []
    backreferences = []
    subword_to_token_map = {}
    current_token_i = 0
    for subw_i, (subw_backr, subtok) in enumerate(zip(subtoken_backreferences, subtokens)):
        subword_to_token_map[subw_i] = current_token_i

        # if empty you cannot do anything but add a new word
        if not tokens:
            tokens.append(subtok.lstrip(tokenizer.INIT))
            backreferences.append(-1)
            current_token_i += 1

        # backref can't be splitted
        elif subw_backr > -1:
            tokens.append(None)
            backreferences.append(subword_to_token_map[subw_backr])
            current_token_i += 1

        # after a special token release
        elif isinstance(tokens[-1], str) and rex_spc.match(tokens[-1]):
            tokens.append(subtok.lstrip(tokenizer.INIT))
            backreferences.append(-1)
            current_token_i += 1

        # after a subtoken ':' (which should be followed by the rest of the edge) ignore tokenizer.INIT
        # TODO: this is an ugly patch due to the fact that BART tokenizer splits after ':'
        elif (tokens[-1] == ":") and rex_arg.match(subtok):
            tokens[-1] = tokens[-1] + subtok[1:]

        # leading tokenizer.INIT
        elif subtok.startswith(tokenizer.INIT):
            tokens.append(subtok.lstrip(tokenizer.INIT))
            backreferences.append(-1)
            current_token_i += 1

        # very ugly patch for some cases in which tokenizer.INIT is not in the following token to the edge
        elif (
            isinstance(tokens[-1], str)
            and tokens[-1].startswith(":")
            and tokens[-1][-1].isdigit()
            and (subtok != "-of")
        ):
            tokens.append(subtok.lstrip(tokenizer.INIT))
            backreferences.append(-1)
            current_token_i += 1

        # in any other case attach to the previous
        else:
            tokens[-1] = tokens[-1] + subtok

    # strip INIT and fix byte-level
    tokens = [
        tokenizer.convert_tokens_to_string(list(t)).lstrip() if isinstance(t, str) else t
        for t in tokens
    ]
    # tokens = [t.replace(tokenizer.INIT, '') if isinstance(t, str) else t for t in tokens]

    # unks are substituted with thing
    tokens = [t if t != "<unk>" else "thing" for t in tokens]

    old_tokens = tokens
    old_backreferences = backreferences

    # <lit> Barack Obama </lit> -> "Barack Obama"
    tokens = []
    backreferences = []
    token_to_token_map = {}
    start_search = 0
    removed = 0
    while True:
        try:

            lit_start = old_tokens.index("<lit>", start_search)
            token_addition = old_tokens[start_search:lit_start]
            for i, t in enumerate(token_addition, start=start_search):
                token_to_token_map[i] = i - removed
            tokens += token_addition

            backreferences_addition = [
                token_to_token_map[b] if b > -1 else -1
                for b in old_backreferences[start_search:lit_start]
            ]
            backreferences += backreferences_addition

            lit_end = min(lit_start + 2, len(old_tokens) - 1)

            while lit_end < len(old_tokens):
                old_tok = old_tokens[lit_end]

                if isinstance(old_tok, str) and (
                    (old_tok.startswith(":") and len(old_tok) > 3) or (old_tok == "<stop>")
                ):
                    res_tok = old_tokens[lit_start + 1 : lit_end]
                    for i in range(lit_start, lit_end):
                        token_to_token_map[i] = len(tokens)

                    # Remove possible wrong None
                    res = old_tokens[lit_start + 1 : lit_end]
                    res = [str(r) for r in res if r is not None]
                    res = '"' + "_".join(res) + '"'

                    removed += len(res_tok)
                    start_search = lit_end
                    tokens += [res, old_tok]
                    backreferences += [-1, -1]
                    break

                elif old_tok == "</lit>":
                    res_tok = old_tokens[lit_start + 1 : lit_end]
                    for i in range(lit_start, lit_end + 1):
                        token_to_token_map[i] = len(tokens)

                    # Remove possible wrong None
                    res = old_tokens[lit_start + 1 : lit_end]
                    res = [str(r) for r in res if r is not None]
                    res = '"' + "_".join(res) + '"'

                    removed += len(res_tok) + 1
                    start_search = lit_end + 1
                    tokens.append(res)
                    backreferences.append(-1)
                    break

                else:
                    lit_end += 1
                    start_search = lit_end

        except ValueError:
            token_addition = old_tokens[start_search:]
            for i, t in enumerate(token_addition, start=start_search):
                token_to_token_map[i] = i - removed
            backreferences_addition = [
                token_to_token_map[b] if b > -1 else b for b in old_backreferences[start_search:]
            ]
            tokens += token_addition
            backreferences += backreferences_addition
            break

    tokens = [token_processing(t) for t in tokens]

    shift = 1
    if tokens[1] == "<s>":
        shift = 2

    tokens = tokens[shift:]
    backreferences = [b if b == -1 else b - shift for b in backreferences[shift:]]

    if tokens[-1] == "</s>":
        tokens.pop()
        backreferences.pop()

    return tokens, backreferences


def index_of(element, iterable, default=None, start=None, end=None):
    if not callable(element):

        def check(x):
            return element == x

    else:
        check = element
    if start is None:
        start = 0
    if end is None:
        end = len(iterable)
    item = start
    while item < end:
        if check(iterable[item]):
            return item
        item += 1
    return default


def separate_edges_nodes(edges_nodes_slice, *other):
    is_arg = lambda x: isinstance(x, str) and x.startswith(":")
    start = 0
    edges = []
    nodes = []
    l = len(edges_nodes_slice)
    while start < l:
        edge_index = index_of(is_arg, edges_nodes_slice, start=start)
        if edge_index is None or edge_index == (l - 1):
            break
        if is_arg(edges_nodes_slice[edge_index + 1]):
            start = edge_index + 1
            continue
        edges.append(edge_index)
        nodes.append(edge_index + 1)
        start = edge_index + 2
    ret = []
    for oth in other:
        edges_oth = [oth[i] for i in edges]
        nodes_oth = [oth[i] for i in nodes]
        ret.append((edges_oth, nodes_oth))
    return ret


def _split_name_ops(graph):
    # identify name triples
    name_vars = {}
    for i, (v1, rel, v2) in enumerate(graph.triples):
        if rel == ":instance" and v2 == "name":
            name_vars[v1] = 1

    # check if they have ops
    name_vars_to_ops = defaultdict(list)
    for i, (v1, rel, v2) in enumerate(graph.triples):
        if v1 in name_vars and rel.startswith(":op"):
            name_vars_to_ops[v1].append((i, rel, v2.strip('"')))

    triples = graph.triples.copy()
    for nv, ops in name_vars_to_ops.items():
        ops = sorted(ops, key=lambda x: int(x[1][3:]))
        idx, _, lits = zip(*ops)
        for i in idx:
            triples[i] = None

        lits = ['"' + l + '"' for lit in lits for l in lit.split("_")]

        tt = []
        for i, l in enumerate(lits, start=1):
            rel = ":op" + str(i)
            tt.append(penman.Triple(nv, rel, l))

        triples[min(idx)] = tt

    triples = [t if isinstance(t, list) else [t] for t in triples if t is not None]
    triples = [t for tt in triples for t in tt]

    graph_ = penman.Graph(triples)
    graph_.metadata = graph.metadata
    return graph_


def _reconstruct_graph_from_nodes(nodes, backreferences):
    triples = []
    triples_added = set()

    variable2index = {}
    index2variable = {}
    start_index = 0

    cnt = defaultdict(Counter)

    while start_index < len(nodes):
        stop_index = index_of("<stop>", nodes, default=len(nodes) + 1, start=start_index)
        old_start_index = start_index
        start_index = stop_index + 1

        src_node, src_backr = nodes[old_start_index], backreferences[old_start_index]

        if src_node == "<stop>":
            continue

        trg_nodes_edges = nodes[old_start_index:stop_index]
        trg_nodes_edges_backr = backreferences[old_start_index:stop_index]
        trg_nodes_edges_indices = list(range(old_start_index, stop_index))

        if isinstance(src_node, str):
            if src_node in ("<s>", "</s>", "<stop>"):
                continue
            elif ("/" in src_node) or (":" in src_node) or ("(" in src_node) or (")" in src_node):
                src_node = "thing"

        if src_node is not None:
            src_node = str(src_node)
            src_var = src_node[0].lower()
            if not src_var not in "abcdefghijklmnopqrstuvwxyz":
                src_var = "x"
            # src_var = f'{src_var}_{len(variable2index)}'
            src_var = f"{src_var}{len(variable2index)}"
            src_var_i = old_start_index
            variable2index[src_var] = src_var_i
            index2variable[src_var_i] = src_var
            triple = penman.Triple(src_var, ":instance", src_node)
            if triple not in triples_added:
                triples.append(triple)
                triples_added.add(triple)
        else:
            if src_backr in index2variable:
                src_var = index2variable[src_backr]
        # more resilient logic here
        (trg_edges, trg_nodes), (_, trg_nodes_backr), (_, trg_nodes_indices) = separate_edges_nodes(
            trg_nodes_edges, trg_nodes_edges, trg_nodes_edges_backr, trg_nodes_edges_indices
        )

        for n, e, nb, ni in zip(trg_nodes, trg_edges, trg_nodes_backr, trg_nodes_indices):

            if isinstance(n, str) and n.startswith(":"):
                continue
            if isinstance(n, str) and n.startswith("<") and n.endswith(">"):
                continue
            if e == ":li":
                pass
            elif len(e) < 4 or (not e.startswith(":")):
                continue

            # same edge more than once
            num = cnt[src_var][e]
            # num = 0
            if num:

                if e.startswith(":op") or e.startswith(":snt"):
                    continue
                # elif e.startswith(':ARG'):
                #    continue
                elif num > 3:
                    continue

            if n is None:
                if nb not in index2variable:
                    continue
                trg_var = index2variable[nb]
                trg = trg_var
            elif e == ":mode":
                trg = n
            elif (
                (not isinstance(n, str))
                or re.match(r"^[+-]?\d+\.?\d*$", n)
                or (n == "-")
                or (n == "+")
            ):
                trg = str(n)
            elif n.startswith('"') and n.endswith('"') and len(n) > 2:
                trg = '"' + n.replace('"', "") + '"'
            elif ("/" in n) or (":" in n) or ("(" in n) or (")" in n) or ("=" in n):
                trg = f'"{n}"'
            elif n == '"':
                continue
            elif (
                (n.startswith('"') and (not n.endswith('"')))
                or (not n.startswith('"') and (n.endswith('"')))
                or ('"' in n)
            ):
                trg = '"' + n.replace('"', "") + '"'
            else:
                trg_var = n[0].lower()
                if trg_var not in "abcdefghijklmnopqrstuvwxyz":
                    trg_var = "x"
                # trg_var = f'{trg_var}_{len(variable2index)}'
                trg_var = f"{trg_var}{len(variable2index)}"
                trg_var_i = ni
                variable2index[trg_var] = trg_var_i
                index2variable[trg_var_i] = trg_var
                triple = penman.Triple(trg_var, ":instance", n)
                if triple not in triples_added:
                    triples.append(triple)
                    triples_added.add(triple)
                trg = trg_var

            triple = penman.Triple(src_var, e, trg)
            if triple not in triples_added:
                triples.append(triple)
                triples_added.add(triple)

            cnt[src_var][e] += 1

    return penman.Graph(triples)


def build_graph(nodes, backreferences, restore_name_ops=False):
    graph = _reconstruct_graph_from_nodes(nodes, backreferences)
    if restore_name_ops:
        graph = _split_name_ops(graph)
    return graph


class ParsedStatus(enum.Enum):
    OK = 0
    FIXED = 1
    BACKOFF = 2


def connect_graph_if_not_connected(graph):

    try:
        encoded = encode(graph)
        return graph, ParsedStatus.OK
    except:
        pass

    nxgraph = nx.MultiGraph()
    variables = graph.variables()
    for v1, _, v2 in graph.triples:
        if v1 in variables and v2 in variables:
            nxgraph.add_edge(v1, v2)
        elif v1 in variables:
            nxgraph.add_edge(v1, v1)

    triples = graph.triples.copy()
    new_triples = []
    addition = f"a{len(variables) + 1}"
    triples.append(penman.Triple(addition, ":instance", "and"))
    for i, conn_set in enumerate(nx.connected_components(nxgraph), start=1):
        edge = f":op{i}"
        conn_set = sorted(conn_set, key=lambda x: int(x[1:]))
        conn_set = [c for c in conn_set if c in variables]
        node = conn_set[0]
        new_triples.append(penman.Triple(addition, edge, node))
    triples = new_triples + triples
    metadata = graph.metadata
    graph = penman.Graph(triples)
    graph.metadata.update(metadata)
    encode(graph)

    return graph, ParsedStatus.FIXED


def restore_backreferences_from_pointers(nodes):
    new_nodes, new_backreferences = [], []
    prev_pointer = None
    pointer2i = {}
    for n in nodes:
        is_pointer = isinstance(n, str) and n.startswith("<pointer:") and n.endswith(">")

        if not is_pointer:
            if prev_pointer is not None:
                if prev_pointer in pointer2i:
                    new_nodes.append(None)
                    new_backreferences.append(pointer2i[prev_pointer])
                    new_nodes.append(n)
                    new_backreferences.append(-1)

                else:
                    pointer2i[prev_pointer] = len(new_nodes)
                    new_nodes.append(n)
                    new_backreferences.append(-1)
            else:
                new_nodes.append(n)
                new_backreferences.append(-1)

            prev_pointer = None
        else:
            prev_pointer = n
    return new_nodes, new_backreferences


@dataclass
class SemanticGraph:

    nodes_var: List[str]
    """
    List of linearized nodes, with special tokens.
    """
    edges: Optional[List[str]]
    """
    List of linearized edges, with special tokens.
    """
    backreferences: List[int]
    """
    List of backpointers to handle rentrancies and cycles.
    """
    var2instance: Dict[str, str]
    """
    Dict from var ids to 'lemmatized' readable strings qualifying the node (collapsing the :instance edge for AMR).
    """
    extra: Dict[str, Any]
    """
    Holds extra stuff that might be useful, e.g. alignments, NER, EL.
    """

    @cached_property
    def variables(self) -> Set[str]:
        """Set of variables in this semantic graph"""
        variables = {v for v in self.nodes_var if not v.startswith('<')}
        return variables

    @property
    def resolved_nodes_var(self) -> List[str]:
        return [self.nodes_var[b] for b in self.backreferences]

    @cached_property
    def nodes(self) -> List[str]:
        """Linearized nodes with varids replaced by instances"""
        return [self.var2instance.get(node, node) for node in self.nodes_var]

    @property
    def resolved_nodes(self) -> List[str]:
        return [self.nodes[b] for b in self.backreferences]

    def src_occurrence(self, var: str) -> int:
        pass


class BaseLinearizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def linearize(self, *args, **kwargs) -> SemanticGraph:
        pass


class AMRTokens:

    START, END = '<', '>'
    _TEMPL = START + '{}' + END

    BOS_N   = _TEMPL.format('s')
    EOS_N   = _TEMPL.format('/s')
    START_N = _TEMPL.format('start')
    STOP_N  = _TEMPL.format('stop')
    PNTR_N  = _TEMPL.format('pointer')

    LIT_START = _TEMPL.format( 'lit')
    LIT_END   = _TEMPL.format('/lit')

    BACKR_SRC_N = _TEMPL.format('backr:src:XXX')
    BACKR_TRG_N = _TEMPL.format('backr:trg:XXX')

    BOS_E   = _TEMPL.format('s')
    EOS_E   = _TEMPL.format('/s')
    START_E = _TEMPL.format('start')
    STOP_E  = _TEMPL.format('stop')

    _FIXED_SPECIAL_TOKENS_N = {
        BOS_N, EOS_N, START_N, STOP_N}
    _FIXED_SPECIAL_TOKENS_E = {
        BOS_E, EOS_E, START_E, STOP_E}
    _FIXED_SPECIAL_TOKENS = _FIXED_SPECIAL_TOKENS_N | _FIXED_SPECIAL_TOKENS_E

    # match and read backreferences
    _re_BACKR_SRC_N = re.compile(BACKR_SRC_N.replace('XXX', r'([0-9]+)'))
    _re_BACKR_TRG_N = re.compile(BACKR_TRG_N.replace('XXX', r'([0-9]+)'))

    @classmethod
    def is_node(cls, string: str) -> bool:
        if isinstance(string, str) and string.startswith(':'):
            return False
        elif string in cls._FIXED_SPECIAL_TOKENS_E:
            return False
        return True

    @classmethod
    def read_backr(cls, string: str) -> Optional:
        m_src = cls._re_BACKR_SRC_N.search(string)
        if m_src is not None:
            return m_src
        m_trg = cls._re_BACKR_TRG_N.search(string)
        if m_trg is not None:
            return m_trg
        return None


T = TypeVar('T')


def index_default(
        item: T, list_: List[T],
        start: Optional[int] = None,
        stop: Optional[int] = None,
        default: Optional[int] = None
):
    if start is None:
        start = 0
    if stop is None:
        stop = len(list_)
    return next((i for i, x in enumerate(list_[start:stop], start=start) if x == item), default)


class AMRLinearizer(BaseLinearizer):

    def __init__(
            self,
            use_pointer_tokens: bool = True,
            collapse_name_ops: bool = False,
    ):
        self.collapse_name_ops = collapse_name_ops
        self.interleave_edges = False
        self.use_pointer_tokens = use_pointer_tokens

    def _collapse_name_ops(self, amr):
        # identify name triples
        name_vars = {}
        for i, (v1, rel, v2) in enumerate(amr.triples):
            if rel == ':instance' and v2 == 'name':
                name_vars[v1] = 1

        # check if they have ops
        name_vars_to_ops = defaultdict(list)
        for i, (v1, rel, v2) in enumerate(amr.triples):
            if v1 in name_vars and rel.startswith(':op'):
                name_vars_to_ops[v1].append((i, rel, v2.strip('"')))

        triples = amr.triples.copy()
        for nv, ops in name_vars_to_ops.items():
            ops = sorted(ops, key=lambda x: int(x[1][3:]))
            idx, _, lits = zip(*ops)
            for i in idx:
                triples[i] = None
            lit = '"' + '_'.join(lits) + '"'
            triples[min(idx)] = penman.Triple(nv, ':op1', lit)

        triples = [t for t in triples if t is not None]
        amr_ = penman.Graph(triples)
        amr_.metadata = amr.metadata
        return amr_

    def linearize(self, amr: penman.Graph) -> SemanticGraph:
        if self.collapse_name_ops:
            amr = self._collapse_name_ops(amr)
        linearized = self._linearize(amr)
        linearized = self._interleave(linearized)
        if self.use_pointer_tokens:
            linearized = self._add_pointer_tokens(linearized)
        return linearized

    def _linearize(self, amr: penman.Graph) -> SemanticGraph:
        variables = set(amr.variables())
        variables = {'var:' + v for v in variables}
        var2instance = {}

        graph = nx.MultiDiGraph()

        triples2order = {k: i for i, k in enumerate(amr.triples)}

        for triple in amr.triples:
            var, rel, instance = triple
            order = triples2order[triple]
            if rel != ':instance':
                continue
            for expansion_candidate in itertools.chain(range(order - 1, -1), range(order + 1, len(amr.triples))):
                if var == amr.triples[expansion_candidate][2]:
                    expansion = expansion_candidate
                    break
            else:
                expansion = 0
            var = 'var:' + var
            var2instance[var] = instance
            graph.add_node(var, instance=instance, order=order, expansion=expansion)

        for triple in amr.edges():
            var1, rel, var2 = triple
            order = triples2order[triple]
            if rel == ':instance':
                continue
            var1 = 'var:' + var1
            var2 = 'var:' + var2
            graph.add_edge(var1, var2, rel=rel, order=order)

        for triple in amr.attributes():
            var, rel, attr = triple
            order = triples2order[triple]
            if rel == ':instance':
                continue
            var = 'var:' + var
            graph.add_edge(var, attr, rel=rel, order=order)

        # nodes that are not reachable from the root (e.g. because of reification)
        # will be present in the not_explored queue
        # undirected_graph = graph.to_undirected()
        # print(amr.variables())
        not_explored = deque(sorted(variables, key=lambda x: nx.get_node_attributes(graph, 'order')[x]))
        # (
        #     len(nx.shortest_path(undirected_graph, 'var:' + amr.top, x)),
        #     -graph.out_degree(x),
        # )

        first_index = {}
        explored = set()
        added_to_queue = set()
        nodes_visit = [AMRTokens.BOS_N]
        edges_visit = [AMRTokens.BOS_E]
        backreferences = [0]
        queue = deque()
        queue.append('var:' + amr.top)

        while queue or not_explored:

            if queue:
                node1 = queue.popleft()
            else:
                node1 = not_explored.popleft()
                if node1 in added_to_queue:
                    continue
                if not list(graph.successors(node1)):
                    continue

            if node1 in variables:
                if node1 in explored:
                    continue
                if node1 in first_index:
                    nodes_visit.append(AMRTokens.BACKR_TRG_N)
                    backreferences.append(first_index[node1])
                else:
                    backreferences.append(len(nodes_visit))
                    first_index[node1] = len(nodes_visit)
                    nodes_visit.append(node1)
                edges_visit.append(AMRTokens.START_E)

                successors = []
                for node2 in graph.successors(node1):
                    for edge_data in graph.get_edge_data(node1, node2).values():
                        rel = edge_data['rel']
                        order = edge_data['order']
                        successors.append((order, rel, node2))
                successors = sorted(successors)

                for order, rel, node2 in successors:
                    edges_visit.append(rel)

                    # node2 is a variable
                    if node2 in variables:
                        # ... which was mentioned before
                        if node2 in first_index:
                            nodes_visit.append(AMRTokens.BACKR_TRG_N)
                            backreferences.append(first_index[node2])

                        # .. which is mentioned for the first time
                        else:
                            backreferences.append(len(nodes_visit))
                            first_index[node2] = len(nodes_visit)
                            nodes_visit.append(node2)

                        # 1) not already in Q
                        # 2) has children
                        # 3) the edge right before its expansion has been encountered
                        if (node2 not in added_to_queue) and list(graph.successors(node2)) and (nx.get_node_attributes(graph, 'expansion')[node2] <= order):
                            queue.append(node2)
                            added_to_queue.add(node2)

                    # node2 is a constant
                    else:
                        backreferences.append(len(nodes_visit))
                        nodes_visit.append(node2)

                backreferences.append(len(nodes_visit))
                nodes_visit.append(AMRTokens.STOP_N)
                edges_visit.append(AMRTokens.STOP_E)
                explored.add(node1)

            else:
                backreferences.append(len(nodes_visit))
                nodes_visit.append(node1)
                explored.add(node1)

        backreferences.append(len(nodes_visit))
        nodes_visit.append(AMRTokens.EOS_N)
        edges_visit.append(AMRTokens.EOS_E)
        assert len(nodes_visit) == len(edges_visit) == len(backreferences)
        return SemanticGraph(
            nodes_visit,
            edges_visit,
            backreferences,
            var2instance,
            extra={'graph': graph, 'amr': amr}
        )

    def _interleave(self, graph: SemanticGraph) -> SemanticGraph:

        new_backreferences_map = []
        new_nodes = []
        new_edges = None
        new_backreferences = []

        # to isolate sublist to the stop token
        start_i = 1
        end_i = index_default(AMRTokens.STOP_N, graph.nodes_var, start_i, -1, -1)

        def add_node(node, backr = None):
            old_n_node = len(new_backreferences_map)
            new_n_node = len(new_nodes)

            if backr is None:
                backr = old_n_node

            new_backreferences_map.append(new_n_node)
            new_nodes.append(node)
            if old_n_node == backr:
                new_backreferences.append(new_n_node)
            else:
                new_backreferences.append(new_backreferences_map[backr])

        def add_edge(edge):
            new_nodes.append(edge)
            new_backreferences.append(len(new_backreferences))

        add_node(AMRTokens.BOS_N)

        while end_i > -1:

            # src node
            add_node(graph.nodes_var[start_i], graph.backreferences[start_i])

            # edges and trg nodes, interleaved
            nodes = graph.nodes_var[start_i+1:end_i]
            edges = graph.edges[start_i+1:end_i]
            backr = graph.backreferences[start_i+1:end_i]
            for n, e, b in zip(nodes, edges, backr):
                add_edge(e)
                add_node(n, b)

            # stop
            add_node(graph.nodes_var[end_i], graph.backreferences[end_i])

            start_i = end_i + 1
            end_i = index_default(AMRTokens.STOP_N, graph.nodes_var, start_i, -1, -1)

        add_node(AMRTokens.EOS_N)

        new_graph = SemanticGraph(
            new_nodes,
            None,
            new_backreferences,
            graph.var2instance,
            extra=graph.extra,
        )
        return new_graph

    def _add_pointer_tokens(self, graph: SemanticGraph) -> SemanticGraph:
        new_nodes = []
        var2pointer = {}
        for node, backr in zip(graph.nodes_var, graph.backreferences):

            if node == AMRTokens.BACKR_TRG_N:
                node = graph.nodes_var[backr]
                pointer = var2pointer[node]
                new_nodes.append(pointer)
            elif node in graph.var2instance:
                pointer = var2pointer.setdefault(node, f"<pointer:{len(var2pointer)}>")
                new_nodes.append(pointer)
                new_nodes.append(node)
            else:
                new_nodes.append(node)

        new_backreferences = list(range(len(new_nodes)))
        new_graph = SemanticGraph(
            new_nodes,
            None,
            new_backreferences,
            graph.var2instance,
            extra=graph.extra,
        )
        return new_graph


op_model = Model()
noop_model = NoOpModel()
amr_model = amr.model
DEFAULT = op_model


def _get_model(dereify):
    if dereify is None:
        return DEFAULT
    elif dereify:
        return op_model
    else:
        return noop_model


def _remove_wiki(graph):
    metadata = graph.metadata
    triples = []
    for t in graph.triples:
        v1, rel, v2 = t
        if rel == ':wiki':
            t = Triple(v1, rel, '+')
        triples.append(t)
    graph = Graph(triples)
    graph.metadata = metadata
    return graph


def load(source, dereify=None, remove_wiki=False):
    model = _get_model(dereify)
    out = load_(source=source, model=model)
    if remove_wiki:
        for i in range(len(out)):
            out[i] = _remove_wiki(out[i])
    return out


def loads(string, dereify=None, remove_wiki=False):
    model = _get_model(dereify)
    out = loads_(string=string, model=model)
    if remove_wiki:
        for i in range(len(out)):
            out[i] = _remove_wiki(out[i])
    return out


def encode(g, top=None, indent=-1, compact=False):
    model = amr_model
    return encode_(g=g, top=top, indent=indent, compact=compact, model=model)


class AMRBartTokenizer(BartTokenizer):

    INIT = 'Ġ'

    ADDITIONAL = [
        AMRTokens.PNTR_N,
        AMRTokens.STOP_N,
        AMRTokens.LIT_START,
        AMRTokens.LIT_END,
        AMRTokens.BACKR_SRC_N,
        AMRTokens.BACKR_TRG_N,
        "<AMR>",
        "</AMR>"]

    def __init__(self, *args, use_pointer_tokens=False, collapse_name_ops=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.patterns = ree.compile(
            r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.linearizer = AMRLinearizer(use_pointer_tokens=use_pointer_tokens, collapse_name_ops=collapse_name_ops)
        self.use_pointer_tokens = use_pointer_tokens
        self.collapse_name_ops = collapse_name_ops
        self.recategorizations = set()
        self.modified = 0

    @classmethod
    def from_pretrained(cls, pretrained_model_path, pred_min=5, *args, **kwargs):
        inst = super().from_pretrained(pretrained_model_path, *args, **kwargs)
        inst.init_amr_vocabulary(pred_min=pred_min)
        return inst

    def init_amr_vocabulary(self, pred_min=5):
        for tok in [self.bos_token, self.eos_token, self.pad_token, '<mask>', '<unk>']:
            ntok = self.INIT + tok
            i = self.encoder[tok]
            self.decoder[i] = ntok
            del self.encoder[tok]
            self.encoder[ntok] = i

        tokens = []
        for line in Path('data_processing/amrbart/amr_predicates.txt').read_text().strip().splitlines():
            tok, count = line.split()
            if int(count) >= pred_min:
                tokens.append(tok)
                
        for tok in Path('data_processing/amrbart/amr_additions.txt').read_text().strip().splitlines():
            tokens.append(tok)

        for tok in Path('data_processing/amrbart/amr_recategorizations.txt').read_text().strip().splitlines():
            if not tok.startswith('_'):
                self.recategorizations.add(tok)
            tokens.append(tok)

        if self.use_pointer_tokens:
            for cnt in range(512):
                tokens.append(f"<pointer:{cnt}>")
        
        for cnt in range(1, 256):
            tokens.append(f"<mask{cnt}>")

        tokens += self.ADDITIONAL
        tokens = [self.INIT + t if t[0] not in ('_', '-') else t for t in tokens]
        tokens = [t for t in tokens if t not in self.encoder]
        
        self.old_enc_size = old_enc_size = len(self.encoder)
        for i, t in enumerate(tokens, start=old_enc_size):
            self.encoder[t] = i

        self.encoder = {k: i for i, (k,v) in enumerate(sorted(self.encoder.items(), key=lambda x: x[1]))}
        self.decoder = {v: k for k, v in sorted(self.encoder.items(), key=lambda x: x[1])}
        self.modified = len(tokens)
        
        self.bos_token = self.INIT + '<s>'
        self.pad_token = self.INIT + '<pad>'
        self.eos_token = self.INIT + '</s>'
        self.mask_token = self.INIT + '<mask>'
        self.unk_token = self.INIT + '<unk>'
        # self.text_bos_token = self.INIT + "ToText"
        # self.text_bos_token_id = self.encoder[self.text_bos_token]
        self.amr_bos_token = self.INIT + "<AMR>"
        self.amr_bos_token_id = self.encoder[self.amr_bos_token]
        self.amr_eos_token = self.INIT + "</AMR>"
        self.amr_eos_token_id = self.encoder[self.amr_eos_token]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def _tokenize(self, text):
        """ Tokenize a string. Modified in order to handle sentences with recategorization pointers"""
        bpe_tokens = []
        for tok_span in text.lstrip().split(' '):
            tok_span = tok_span.strip()
            recats = tok_span.rsplit('_', 1)
            if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
                bpe_tokens.extend([self.INIT + recats[0], '_' + recats[1]])
            else:
                for token in ree.findall(self.pat, ' ' + tok_span):
                    token = "".join(
                        self.byte_encoder[b] for b in token.encode("utf-8")
                    )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
                    bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

    def _tok_bpe(self, token, add_space=True):
        # if add_space:
        #     token = ' ' + token.lstrip()
        tokk = []
        tok = token.strip()
        recats = tok.rsplit('_', 1)
        if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
            tokk.extend([self.INIT + recats[0], '_' + recats[1]])
        else:
            for tok in self.patterns.findall(' ' + token):
                tok = "".join(
                    self.byte_encoder[b] for b in tok.encode("utf-8"))
                toks = self.bpe(tok).split(' ')
                tokk.extend(toks)
        return tokk

    def _get_nodes_and_backreferences(self, graph):
        lin = self.linearizer.linearize(graph)
        linearized_nodes, backreferences = lin.nodes, lin.backreferences
        return linearized_nodes, backreferences

    def tokenize_amr(self, graph):
        linearized_nodes, backreferences = self._get_nodes_and_backreferences(graph)

        bpe_tokens = []
        bpe_backreferences = []
        counter = 0
        
        for i, (backr, tokk) in enumerate(zip(backreferences, linearized_nodes)):
            is_in_enc = self.INIT + tokk in self.encoder
            is_rel = tokk.startswith(':') and len(tokk) > 1
            is_spc = tokk.startswith('<') and tokk.endswith('>')
            is_of  = tokk.startswith(':') and tokk.endswith('-of')
            is_frame = ree.match(r'.+-\d\d', tokk) is not None

            if tokk.startswith('"') and tokk.endswith('"'):
                tokk = tokk[1:-1].replace('_', ' ')
                bpe_toks = [self.INIT + AMRTokens.LIT_START]
                bpe_toks += self._tok_bpe(tokk, add_space=True)
                bpe_toks.append(self.INIT + AMRTokens.LIT_END)

            elif (is_rel or is_spc or is_frame or is_of):
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                elif is_frame:
                    bpe_toks = self._tok_bpe(tokk[:-3], add_space=True) + [tokk[-3:]]
                elif is_of:
                    rel = tokk[:-3]
                    if self.INIT + rel in self.encoder:
                        bpe_toks = [self.INIT + rel, '-of']
                    else:
                        bpe_toks = [self.INIT + ':'] + self._tok_bpe(rel[1:], add_space=True) + ['-of']
                elif is_rel:
                    bpe_toks = [self.INIT + ':'] + self._tok_bpe(tokk[1:], add_space=True)
                else:
                    raise

            else:
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                else:
                    bpe_toks = self._tok_bpe(tokk, add_space=True)

            bpe_tokens.append(bpe_toks)

            if i == backr:
                bpe_backr = list(range(counter, counter + len(bpe_toks)))
                counter += len(bpe_toks)
                bpe_backreferences.append(bpe_backr)
            else:
                bpe_backreferences.append(bpe_backreferences[backr][0:1])
                counter += 1               
        bpe_tokens = [b for bb in bpe_tokens for b in bb]
        bpe_token_ids = [self.encoder.get(b, self.unk_token_id) for b in bpe_tokens]
        bpe_backreferences = [b for bb in bpe_backreferences for b in bb]
        return bpe_tokens, bpe_token_ids, bpe_backreferences

    def get_ids(self, tokens, max_length=0, pad_to_max_length=False):
        token_ids = [self.encoder.get(b, self.unk_token_id) for b in tokens]
        if pad_to_max_length:
            assert max_length > 0, "Invalid max-length: {}".format(max_length)
            pad_ids = [self.encoder.get(self.pad_token) for _ in range(max_length)]
            len_tok = len(token_ids)
            if max_length > len_tok:
                pad_ids[:len_tok] = map(int, token_ids)
            else:
                pad_ids = token_ids[:max_length]
            return torch.tensor(pad_ids, dtype=torch.long)
        return torch.tensor(token_ids, dtype=torch.long)

    def batch_encode_sentences(self, sentences, device=torch.device('cpu')):
        sentences = [s for s in sentences]
        extra = {'sentences': sentences}
        batch = super().batch_encode_plus(sentences, return_tensors='pt', pad_to_max_length=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch, extra
    
    def linearize(self, graph):
        shift = len(self.encoder)
        tokens, token_ids, backreferences = self.tokenize_amr(graph)
        extra = {'linearized_graphs': tokens, 'graphs': graph}
        token_uni_ids = \
            [idx if i == b else b + shift for i, (idx, b) in enumerate(zip(token_ids, backreferences))]
        if token_uni_ids[-1] != (self.INIT + AMRTokens.EOS_N):
            tokens.append(self.INIT + AMRTokens.EOS_N)
            token_ids.append(self.eos_token_id)
            token_uni_ids.append(self.eos_token_id)
            backreferences.append(len(backreferences))
        return token_uni_ids, extra
        
    def batch_encode_graphs(self, graphs, device=torch.device('cpu')):
        linearized, extras = zip(*[self.linearize(g) for g in graphs])
        return self.batch_encode_graphs_from_linearized(linearized, extras, device=device)
    
    def batch_encode_graphs_from_linearized(self, linearized, extras=None, device=torch.device('cpu')):
        if extras is not None:
            batch_extra = {'linearized_graphs': [], 'graphs': []}
            for extra in extras:
                batch_extra['graphs'].append(extra['graphs'])
                batch_extra['linearized_graphs'].append(extra['linearized_graphs'])
        else:
            batch_extra = {}
        maxlen = 0
        batch = []
        for token_uni_ids in linearized:
            maxlen = max(len(token_uni_ids), maxlen)
            batch.append(token_uni_ids)
        batch = [x + [self.pad_token_id] * (maxlen - len(x)) for x in batch]
        batch = torch.tensor(batch).to(device)
        batch = {'decoder_input_ids': batch[:, :-1], 'lm_labels': batch[:, 1:]}
        return batch, batch_extra

    def decode_amr(self, tokens, restore_name_ops=False):
        try:
            nodes, backreferences =  decode_into_node_and_backreferences(tokens, self)
        except Exception as e:
            print('Decoding failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            return  BACKOFF,  ParsedStatus.BACKOFF, (None, None)
        if self.use_pointer_tokens:
            nodes, backreferences =  restore_backreferences_from_pointers(nodes)
        try:
            graph_ = graph =  build_graph(nodes, backreferences, restore_name_ops=restore_name_ops)
        except Exception as e:
            print('Building failure:', file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(e, file=sys.stderr)
            return  BACKOFF,  ParsedStatus.BACKOFF, (None, None)
        try:
            graph, status =  connect_graph_if_not_connected(graph)
            if status ==  ParsedStatus.BACKOFF:
                print('Reconnection 1 failure:')
                print(nodes, file=sys.stderr)
                print(backreferences, file=sys.stderr)
                print(graph_, file=sys.stderr)
            return graph, status, (nodes, backreferences)
        except Exception as e:
            print('Reconnction 2 failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(graph_, file=sys.stderr)
            return  BACKOFF,  ParsedStatus.BACKOFF, (nodes, backreferences)


class PENMANBartTokenizer(AMRBartTokenizer):

    def __init__(self, *args, raw_graph=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.linearizer = None
        self.remove_pars = False
        self.raw_graph = raw_graph

    def _tokenize_encoded_graph(self, encoded):
        linearized = ree.sub(r"(\".+?\")", r' \1 ', encoded)
        pieces = []
        for piece in linearized.split():
            if piece.startswith('"') and piece.endswith('"'):
                pieces.append(piece)
            else:
                piece = piece.replace('(', ' ( ')
                piece = piece.replace(')', ' ) ')
                piece = piece.replace(':', ' :')
                piece = piece.replace('/', ' / ')
                piece = piece.strip()
                pieces.append(piece)
        linearized = ree.sub(r'\s+', ' ', ' '.join(pieces)).strip()
        linearized_nodes = [AMRTokens.BOS_N] + linearized.split(' ')
        return linearized_nodes

    def tokenize_amr(self, graph):
        if self.raw_graph:
            graph_ = copy.deepcopy(graph)
            graph_.metadata = {}
            linearized = penman.encode(graph_)
            linearized = ree.sub(r"\s+", ' ', linearized)
            bpe_tokens = [self.bos_token] + self._tokenize(linearized)[:1022]
            bpe_token_ids = [self.encoder.get(b, self.unk_token_id) for b in bpe_tokens]
            bpe_backreferences = list(range(len(bpe_token_ids)))
            return bpe_tokens, bpe_token_ids, bpe_backreferences
        else:
            return super().tokenize_amr(graph)

    def _get_nodes_and_backreferences(self, graph):
        graph_ = copy.deepcopy(graph)
        graph_.metadata = {}
        linearized = penman.encode(graph_)
        linearized_nodes = self._tokenize_encoded_graph(linearized)

        if self.use_pointer_tokens:
            remap = {}
            for i in range(1, len(linearized_nodes)):
                nxt = linearized_nodes[i]
                lst = linearized_nodes[i-1]
                if nxt == '/':
                    remap[lst] = f'<pointer:{len(remap)}>'
            i = 1
            linearized_nodes_ = [linearized_nodes[0]]
            while i < (len(linearized_nodes)):
                nxt = linearized_nodes[i]
                lst = linearized_nodes_[-1]
                if nxt in remap:
                    if lst == '(' and linearized_nodes[i+1] == '/':
                        nxt = remap[nxt]
                        i += 1
                    elif lst.startswith(':'):
                        nxt = remap[nxt]
                linearized_nodes_.append(nxt)
                i += 1
            linearized_nodes = linearized_nodes_
            if self.remove_pars:
                linearized_nodes = [n for n in linearized_nodes if n != '(']
        backreferences = list(range(len(linearized_nodes)))
        return linearized_nodes, backreferences

    def _classify(self, node):
        if not isinstance(node, str):
            return "CONST"
        elif node == 'i':
            return "I"
        elif ree.match(r'^[a-z]\d*$', node) is not None:
            return "VAR"
        elif node[0].isdigit():
            return "CONST"
        elif node.startswith('"') and node.endswith('"'):
            return "CONST"
        elif node in ('+', '-'):
            return "CONST"
        elif node == ':mode':
            return 'MODE'
        elif node.startswith(':'):
            return "EDGE"
        elif node in ['/', '(', ')']:
            return node
        elif node[0].isalpha():
            for char in (',', ':', '/', '(', ')', '.', '!', '?', '\\'):
                if char in node:
                    return "CONST"
            return "INST"
        else:
            return 'CONST'

    def _fix_and_make_graph(self, nodes):

        nodes_ = []
        for n in nodes:
            if isinstance(n, str):
                if n.startswith('<') and n.endswith('>') and (not n.startswith('<pointer:')):
                    pass
                else:
                    nodes_.append(n)
            else:
                nodes_.append(n)
        nodes = nodes_

        if self.use_pointer_tokens:

            i = 0
            nodes_ = []
            while i < len(nodes):
                nxt = nodes[i]
                pst = None
                if isinstance(nxt, str) and nxt.startswith('<pointer:'):
                    e = nxt.find('>')
                    if e != len(nxt) -1:
                        pst = nxt[e+1:]
                        nxt = nxt[:e+1]
                    nodes_.append(nxt)
                    if pst is not None:
                        nodes_.append(pst)
                else:
                    nodes_.append(nxt)
                i += 1
            nodes = nodes_

            i = 1
            nodes_ = [nodes[0]]
            while i < len(nodes):
                nxt = nodes[i]
                if isinstance(nxt, str) and nxt.startswith('<pointer:'):
                    nxt = 'z' + nxt[9:-1]
                    fol = nodes[i+1]
                    # is not expansion
                    if isinstance(fol, str) and (fol.startswith(':') or (fol == ')')):
                        nodes_.append(nxt)
                    else:
                        if self.remove_pars:
                            nodes_.append('(')
                        else:
                            if nodes_[-1] != '(':
                                nodes_.append('(')
                                #pass
                        nodes_.append(nxt)
                        nodes_.append('/')
                else:
                    nodes_.append(nxt)
                i += 1
            nodes = nodes_

        i = 0
        nodes_ = []
        while i < (len(nodes) - 1):
            if nodes[i] == ':':
                nodes_.append(nodes[i] + nodes[i+1])
                i += 2
                last = False
            else:
                nodes_.append(nodes[i])
                i += 1
                last = True
        if last:
            nodes_.append(nodes[-1])
        nodes = nodes_

        i = 0
        nodes_ = []
        while i < (len(nodes)):
            if i < 2:
                nodes_.append(nodes[i])
                i += 1
            elif nodes_[-2] == '/' and nodes[i] == '/':
                i += 2
            else:
                nodes_.append(nodes[i])
                i += 1
        nodes = nodes_

        i = 0
        newvars = 0
        variables = set()
        remap = {}
        nodes_ = []
        while i < (len(nodes)):

            next = nodes[i]

            if next == '/':
                last = nodes_[-1]
                if last in variables:
                    last_remap = f"z{newvars+1000}"
                    newvars += 1
                    nodes_[-1] = last_remap
                    remap[last] = last_remap
                variables.add(last)
                nodes_.append(next)

            elif self._classify(next) == 'VAR' and next in remap and (i < len(nodes) - 1) and nodes[i+1] != '/':
                next = remap[next]
                nodes_.append(next)

            else:
                nodes_.append(next)

            i += 1

        nodes = nodes_
        pieces_ = []
        open_cnt = 0
        closed_cnt = 0
        if nodes[0] != '(':
            pieces_.append('(')
            open_cnt += 1
        for p in nodes:
            if p == '(':
                open_cnt += 1
            elif p == ')':
                closed_cnt += 1
            pieces_.append(p)
            if open_cnt == closed_cnt:
                break
        nodes = pieces_ + [')'] * (open_cnt - closed_cnt)

        pieces = []
        for piece in nodes:
            if not pieces:
                pieces.append('(')
            else:
                piece = str(piece)
                if piece.startswith('"') or piece.startswith('"') or '"' in piece.strip('"'):
                    piece = '"' + piece.replace('"', '') + '"'

                prev = self._classify(pieces[-1])
                next = self._classify(piece)

                if next == 'CONST':
                    quote = False
                    for char in (',', ':', '/', '(', ')', '.', '!', '?', '\\', '_', '='):
                        if char in piece:
                            quote = True
                            break
                    if quote:
                        piece = '"' + piece.strip('"') + '"'

                if  prev == '(':
                    if next in ('VAR', 'I'):
                        pieces.append(piece)
                elif prev == ')':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'VAR':
                    if next in ('/', 'EDGE', 'MODE', ')'):
                        pieces.append(piece)
                elif prev == '/':
                    if next in ('INST', 'I'):
                        pieces.append(piece)
                elif prev == 'INST':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'I':
                    if next in ('/', ')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'EDGE':
                    if next in ('(', 'VAR', 'CONST', 'I'):
                        pieces.append(piece)
                    elif next == ')':
                        pieces[-1] = piece
                    elif next in ('EDGE', 'MODE'):
                        pieces[-1] = piece
                elif prev == 'MODE':
                    if next == 'INST':
                        pieces.append(piece)
                elif prev == 'CONST':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)

        pieces_ = []
        open_cnt = 0
        closed_cnt = 0
        if pieces[0] != '(':
            pieces_.append('(')
            open_cnt += 1
        for p in pieces:
            if p == '(':
                open_cnt += 1
            elif p == ')':
                closed_cnt += 1
            pieces_.append(p)
            if open_cnt == closed_cnt:
                break
        pieces = pieces_ + [')'] * (open_cnt - closed_cnt)

        linearized = ree.sub(r'\s+', ' ', ' '.join(pieces)).strip()

        """
        line = linearized
        # make sure parentheses match
        # copied from https://github.com/RikVN/AMR/blob/master/restoreAMR/restore_amr.py
        open_count = 0
        close_count = 0
        for i, c in enumerate(line):
            if c == '(':
                open_count += 1
            elif c == ')':
                close_count += 1
            if open_count == close_count and open_count > 0:
                line = line[:i].strip()
                break
        old_line = line
        while True:
            open_count = len(re.findall(r'\(', line))
            close_count = len(re.findall(r'\)', line))
            if open_count > close_count:
                line += ')' * (open_count - close_count)
            elif close_count > open_count:
                for i in range(close_count - open_count):
                    line = line.rstrip(')')
                    line = line.rstrip(' ')
            if old_line == line:
                break
            old_line = line
        """

        graph = penman.decode(linearized + ' ')
        triples = []
        newvars = 2000
        for triple in graph.triples:
            x, rel, y = triple
            if x is None:
                pass
            elif rel == ':instance' and y is None:
                triples.append(penman.Triple(x, rel, 'thing'))
            elif y is None:
                var = f'z{newvars}'
                newvars += 1
                triples.append(penman.Triple(x, rel, var))
                triples.append(penman.Triple(var, ':instance', 'thing'))
            else:
                triples.append(triple)
        graph = penman.Graph(triples)
        linearized = encode(graph)

        def fix_text(linearized=linearized):
            n = 0
            def _repl1(match):
                nonlocal n
                out = match.group(1) + match.group(2) + str(3000 + n) + ' / ' + match.group(2) + match.group(3)
                n += 1
                return out
            linearized = ree.sub(r'(\(\s?)([a-z])([^\/:\)]+[:\)])', _repl1, linearized,
                                flags=re.IGNORECASE | re.MULTILINE)

            def _repl2(match):
                return match.group(1)
            linearized = ree.sub(r'(\(\s*[a-z][\d+]\s*\/\s*[^\s\)\(:\/]+\s*)((?:/\s*[^\s\)\(:\/]+\s*)+)', _repl2,
                                linearized,
                                flags=ree.IGNORECASE | ree.MULTILINE)

            # adds a ':' to args w/o it
            linearized = ree.sub(r'([^:])(ARG)', r'\1 :\2', linearized)

            # removes edges with no node
            # linearized = re.sub(r':[^\s\)\(:\/]+?\s*\)', ')', linearized, flags=re.MULTILINE)

            return linearized

        linearized = fix_text(linearized)

        g = penman.decode(linearized)
        return g

    def decode_amr(self, tokens, restore_name_ops=None):
        try:
            if self.raw_graph:
                nodes = self._tokenize_encoded_graph(self.decode(tokens))
                backreferences = list(range(len(nodes)))
            else:
                nodes, backreferences =  decode_into_node_and_backreferences(tokens, self)
            nodes_ = nodes
        except Exception as e:
            print('Decoding failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            return  BACKOFF,  ParsedStatus.BACKOFF, (None, None)
        try:
            graph_ = graph = self._fix_and_make_graph(nodes)
            if self.collapse_name_ops:
                graph_ = graph =  _split_name_ops(graph)
        except Exception as e:
            print('Building failure:', file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(e, file=sys.stderr)
            return  BACKOFF,  ParsedStatus.BACKOFF, (None, None)
        try:
            graph, status =  connect_graph_if_not_connected(graph)
            if status ==  ParsedStatus.BACKOFF:
                print('Reconnection 1 failure:')
                print(nodes, file=sys.stderr)
                print(backreferences, file=sys.stderr)
                print(graph_, file=sys.stderr)
            return graph, status, (nodes_, backreferences)
        except Exception as e:
            print('Reconnction 2 failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(graph_, file=sys.stderr)
            return  BACKOFF,  ParsedStatus.BACKOFF, (nodes_, backreferences)


class AMRParser(object):
    def __init__(self):
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

        self.rules = {}
        with open('v1.rules', 'r') as f:
            for line in f.readlines():
                if '||' in line:
                    a, b = line.split('||')
                    a, b = a.strip(), b.strip()
                    self.rules[a][b] = 'NEG'
                else:
                    a, b = line.split('|')
                    a, b = a.strip(), b.strip()
                    if len(b)>0 and b!='NONE':
                        self.rules[a][b] = 'Strong POS'
                    elif b=='NONE':
                        self.rules[a][b] = 'NEG'

    def __call__(self, inp):
        punkts = [',', '.', '!', '?', ':']
        tinp = []
        for x in inp.split():
            if x[-1] in punkts:
                tinp.append(x[:-1])
                tinp.append(x[-1])
            else:
                tinp.append(x)
        inp = ' '.join(tinp)
        _inp = self.tokenizer([inp])['input_ids'][0]        
        _inp = torch.LongTensor([_inp + [self.tokenizer.amr_bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.amr_eos_token_id]])
        ret = self.model.generate(_inp, num_beams=4, length_penalty=1.0, max_length=768, output_attentions=True, return_dict_in_generate=True)#, min_length=0, no_repeat_ngram_size=0)
        tt = self.tokenizer.convert_ids_to_tokens(ret['sequences'][0])
        inps = self.tokenizer.convert_ids_to_tokens(_inp[0])

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
            ss = step[0][0].sum([0,1])[1:-4]
            ss = ss/ss.sum().numpy()
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

        tokens = [''.join(x).replace('Ġ', '') for x in show_inp]
        print(self.tokenizer.batch_decode(ret['sequences']))
        amr_nodes, amr_edges = parse_amr(cor_matrix, tokens, word_dict={}, rules=self.rules)
        return amr_nodes, amr_edges, tokens


from word_forms.word_forms import get_word_forms
from Levenshtein import distance


def match_rules(str1, str2, rules):
    if rules is None:
        return 'None'
    ret = 'None'
    if str1 in rules:
        if ret != 'NEG' and str2 in rules[str1]:
            ret = rules[str1][str2]
    return ret


def parse_amr(amrs, tokens, word_dict, rules=None):
    stack = []
    nodes = []
    visit = []
    edges = []
    node_cnt = 0
    fcnt = 0
    buffer = []
    pointers = {}
    for i in range(len(amrs)):
        for j in range(len(amrs[i][1])):
            tt = amrs[i][0].lower()
            if len(tt)>3 and tt[-3]=='-' and tt[-2:].isdigit():
                tt = tt[:-3]
            if tt in word_dict:
                all_words = word_dict[tt]
            else:
                all_words = get_word_forms(tt)
                all_words = list(set(sum([list(x) for x in all_words.values()], [])))
                word_dict[tt] = all_words
            l1 = max(len(tokens[j].lower()), len(tt))
            l2 = commonAnc(tokens[j].lower(), tt)
            l3 = distance(tt, tokens[j].lower())
            _rule = match_rules(tt, tokens[j].lower(), rules)
            if (l1>10 and abs(l1-l2)<=4) or (l1<=10 and l2/(l1+1e-6)>0.7) or (tt==tokens[j].lower()) or (tokens[j].lower() in all_words) or (1.0*l3/l1<0.46):
                _rule = 'POS'
            if _rule == 'Strong POS':
                amrs[i][1][j] += 200
            if _rule == 'POS':
                amrs[i][1][j] += 100
            if _rule == 'NEG':
                amrs[i][1][j] -= 100

    amr_matrix = np.stack([x[1] for x in amrs], 0)
    aligns = {}
    sort_adj = np.argsort(amr_matrix, 0)[::-1, :]
    part_sort = np.argsort(sort_adj[:5, :].flatten())[::-1]
    sp = len(part_sort)//5
    for j in range(len(part_sort)):
        x, y = j//sp, j%sp
        x = sort_adj[x, y]
        if len(amrs[x][0])<1 or (amrs[x][0][0]=='<' and amrs[x][0][-1]=='>') or amrs[x][0][0]==':' or tokens[y].lower() in STOPs or amrs[x][0].lower() in STOPs:
            continue
        if x not in aligns and amr_matrix[x, y]>0.15:
            aligns[x] = y

    i = 0

    lit_flag = False
    while i < len(amrs):
        w = amrs[i][0]
        if len(w)<1 or w=='-':
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
                    buffer[-1].extend([w, aligns.get(i, -1)])
                else:
                    buffer.append(['t', w, aligns.get(i, -1)])

        if len(buffer)>0 and not lit_flag:
            if len(buffer)>2 and buffer[-1][0]=='t' and buffer[-2][0] in ['nc'] and buffer[-3][0]==':':
                if not (buffer[-1][1] in STOPs or buffer[-1][2] is None):
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
                if not (buffer[-1][1] in STOPs or buffer[-1][2] is None):
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
            if not (buffer[-1][1] in STOPs or buffer[-1][2]==-1 or buffer[-1][2] is None):
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
            if not (buffer[-1][1] in STOPs or buffer[-1][2] is None):
                n = [buffer[-1][1], buffer[-1][2]]
                nodes.append(n)
                pointers[buffer[-2][1]] = node_cnt
                node_cnt += 1
            buffer = []

    return nodes, edges


def parse_amr_bak(amr_str):
    amr_str = amr_str.replace('<lit>', '').replace('</lit>', '')
    ws = amr_str.split()
    stack = []
    nodes = []
    node_cnt = 0
    child = {} 
    i = 0
    fcnt = 0
    while i < len(ws):
        w = ws[i]
        if w in ['</s>', '<AMR>', '</AMR>', '<s>']:
            i += 1
            continue
        if w[0]==':':
            if i<len(ws)-3 and ws[i+1][0] == '(' and ws[i+2][0]=='<' and ws[i+3][0] not in ['<', ':']:
                tlist = [x[2] for x in nodes]
                if w[1:-1] in tlist:
                    t = tlist.index(w[1:-1])
                    if len(stack)>0:
                        if stack[-1] not in child:
                            child[stack[-1]] = []
                        child[stack[-1]].append(t)
                    stack.append(t)                                       
                else:
                    n = [node_cnt, w[1:], ws[i+2][1:-1], ws[i+3]]
                    if len(stack)>0:
                        if stack[-1] not in child:
                            child[stack[-1]] = []
                        child[stack[-1]].append(node_cnt)
                    stack.append(node_cnt)
                    nodes.append(n)
                    node_cnt += 1
                i+=3
            elif i<len(ws)-1:
                n = [node_cnt, w[1:], None, ws[i+1]]
                if len(stack)>0:
                    if stack[-1] not in child:
                        child[stack[-1]] = []
                    child[stack[-1]].append(node_cnt)
                #stack.append(node_cnt)
                nodes.append(n)
                node_cnt += 1
                i+=1

        elif w==')':
            if len(stack)>0:
                stack = stack[:-1]
        elif w[0] == '<':
            if i < len(ws)-1:
                if ws[i+1][0] not in ['<', ':']:
                    n = [node_cnt, None, ws[i][1:-1], ws[i+1]] 
                    stack.append(node_cnt)
                    nodes.append(n)
                    node_cnt += 1 
                else:
                    tlist = [x[2] for x in nodes]
                    if w[1:-1] in tlist:
                        t = tlist.index(w[1:-1])
                        stack.append(t) 
                if i>2:
                    fcnt += 1
                i += 1
                
        i += 1
    return nodes, child, fcnt


def parse_file():
    nc = []
    ff = False
    lc = 0
    with open('amr-bank-struct-v3.0.txt', 'r') as f, open('amr_lp_sents.txt', 'w') as wf:
        for line in f.readlines():
            line = line.strip()
            if len(line)>2:
                if ff:
                    lc += 1
                elif line.startswith('# ::snt'):
                    wf.write(line[7:].strip().replace('"', '')+'\n')
                elif line[0]=='(' and not ff:
                    ff = True
                    lc += 1
            elif ff:
                ff = False
                nc.append(lc)
                lc = 0
    return nc


def parse_sst2():
    with open('SST-2/test.tsv', 'r') as f, open('amr_sst2_sents.txt', 'w') as wf:
        for line in f.readlines()[1:]:
            if len(line)>2:
                wf.write(line.split('\t')[1].strip()+'\n')

