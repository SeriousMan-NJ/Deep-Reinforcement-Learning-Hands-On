import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from lib import utils
import math
import copy
import time
import os
import random

NUM_PHYS = len(utils.X86.F_ALL)
# isAllocated, weight, size, isPhysReg, isIntReg, isFloatReg, isVecReg, isNext
# allocation: NUM_PHYS
NUM_NODE_FEATURES = 8 + NUM_PHYS

WIN = 0
LOSE = 1
DRAW = 2

NodeMap = {}

Files = []

def _init_files():
    DIR = "/home/ywshin/Deep-Reinforcement-Learning-Hands-On/SVNet/data"
    ALL = [
        os.path.join(DIR, "600.perlbench_s_filtered"),
        os.path.join(DIR, "602.gcc_s_filtered"),
        os.path.join(DIR, "605.mcf_s_filtered"),
        os.path.join(DIR, "620.omnetpp_s_filtered"),
        os.path.join(DIR, "623.xalancbmk_s_filtered"),
        os.path.join(DIR, "625.x264_s_filtered"),
        os.path.join(DIR, "631.deepsjeng_s_filtered"),
        os.path.join(DIR, "641.leela_s_filtered"),
        # os.path.join(DIR, "648.exchange2_s_filtered"),
        os.path.join(DIR, "657.xz_s_filtered"),
        os.path.join(DIR, "998.specrand_is_filtered"),
    ]
    for p in ALL:
        Files.extend([os.path.join(p, name) for name in os.listdir(p) if os.path.isfile(os.path.join(p, name))])

_init_files()

def get_file_index():
    return random.randrange(len(Files))

def set_node_features(G, filepath):
    bound = 0
    if filepath not in NodeMap:
        NodeMap[filepath[:-7]] = {}
    with open(filepath, "r") as f:
        lines = f.readlines()
        for l in lines:
            features = l.split(",")
            if len(l) < 2:
                break
            nid = int(features[0])
            if nid not in NodeMap[filepath[:-7]]:
                NodeMap[filepath[:-7]][nid] = bound
                bound += 1
            G.add_node(NodeMap[filepath[:-7]][nid],
                nid=nid,
                isNext=0,
                isAllocated = int(features[1]),
                allocation = int(features[2]),
                weight = 100 if float(features[3]) > 100 else float(features[3]), # TODO
                size = int(features[4]),
                isPhysReg = int(features[5]),
                isIntReg = int(features[6]),
                isFloatReg = int(features[7]),
                isVecReg = int(features[8]),
                allocOrder = list(map(lambda x: int(x), features[9].split()))
            )

def add_edges(G, filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
        for l in lines:
            node = NodeMap[filepath[:-7]][int(l.split()[0])]
            edges = list(map(lambda x: NodeMap[filepath[:-7]][int(x)], l.split()[1:]))
            if len(edges) < 1:
                continue
            for e in edges:
                G.add_edge(node, e)

def get_next_node_id(G):
    w = -np.inf
    N = None
    for nid, attr in G.nodes(data=True):
        if attr['isAllocated']:
            continue

        if w < attr['weight']:
            w = attr['weight']
            N = nid

    return N

File_index = 0
def change_file_index():
    global File_index
    File_index = get_file_index()

def get_initial_state():
    while True:
        filepath = Files[File_index]
        n = sum(1 for line in open(filepath))
        if n > 200:
            change_file_index()
            continue
        else:
            break
    nf_filepath = filepath.split('.')
    nf_filepath[-2] = 'nf'
    nf_filepath = '.'.join(nf_filepath)

    if_filepath = filepath.split('.')
    if_filepath[-2] = 'if'
    if_filepath = '.'.join(if_filepath)

    G = nx.Graph()
    set_node_features(G, nf_filepath)
    add_edges(G, if_filepath)
    nid = get_next_node_id(G)
    G.string_id = None
    G.nnid = nid
    G.nodes[nid]['isNext'] = 1
    return G

def state_to_int(G):
    if G.string_id is None:
        s = []
        for _, attr in G.nodes(data=True):
            s.append(str(attr['allocation']))
        G.string_id = ','.join(s)
    return G.string_id

INITIAL_STATE = get_initial_state()

def possible_moves(state, nid=None):
    """
    :param state_int: field representation
    :return: the list of columns which we can make a move
    """
    assert isinstance(state, nx.Graph)
    if nid is None:
        nid = get_next_node_id(state)
    N = state.nodes(data=True)[nid]

    return {utils.renumber_reg(idx) for idx in N['allocOrder']}

def _get_spill_costs(G):
    spill_costs = 0.0
    for _, attr in G.nodes(data=True):
        if len(attr['allocOrder']) > 0:
            return -1
        if attr['allocation'] == -1:
            spill_costs += attr['weight']
    return spill_costs

def deepcopy(state):
    state_new = copy.deepcopy(state)
    state_new.string_id = None
    return state_new

def deepcopy_light(state):
    state_new = state.copy()
    state_new.string_id = None
    for nid, attr in state.nodes(data=True):
        state_new.nodes[nid]['allocOrder'] = attr['allocOrder'][:]
    return state_new

def move(state, reg, nid=None):
    """
    Perform move into given register. Assume the move could be performed, otherwise, assertion will be raised
    :param state: current state
    :param reg: register to make a move
    :return: tuple of (state_new, won). Value won is bool, True if this move lead
    to victory or False otherwise (but it could be a draw)
    """
    # TODO: assert isinstance(state, nx.Graph)
    assert isinstance(reg, int)
    # assert 0 <= reg < len(utils.X86.F_ALL)

    state_new = deepcopy_light(state) # MUST create deep copy of the graph

    if nid is None:
        nid = get_next_node_id(state_new)
    state_new.nnid = nid
    assert state_new.nodes[state.nnid]['isNext'] == 1
    state_new.nodes[state.nnid]['isNext'] = 0
    state_new.nodes[nid]['isNext'] = 1
    state_new.nodes[nid]['isAllocated'] = 1
    state_new.nodes[nid]['allocation'] = reg
    state_new.nodes[nid]['allocOrder'] = []
    neighbors = state_new.neighbors(nid)
    if reg != -1:
        for n in neighbors:
            # neighboring nodes에 이미 할당되어 있으면 안 된다.
            assert utils.renumber_reg(state_new.nodes[n]['allocation']) != utils.renumber_reg(reg)

            state_new.nodes[n]['allocOrder'] = list(filter(lambda x: utils.renumber_reg(reg) != utils.renumber_reg(x), state_new.nodes[n]['allocOrder']))

    spill_costs = _get_spill_costs(state_new)

    return state_new, spill_costs

# TODO: used in play.py
def update_counts(counts_dict, key, counts):
    v = counts_dict.get(key, (0, 0, 0))
    res = (v[0] + counts[0], v[1] + counts[1], v[2] + counts[2])
    counts_dict[key] = res
