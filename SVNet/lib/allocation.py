import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from lib import utils
import math
import copy
import time

NUM_PHYS = len(utils.X86.F_ALL)
# isAllocated, weight, size, isPhysReg, isIntReg, isFloatReg, isNext
# allocation: NUM_PHYS
NUM_NODE_FEATURES = 7 + NUM_PHYS

WIN = 0
LOSE = 1
DRAW = 2

NodeMap = {}
bound = 0

def set_node_features(G, filepath):
    global bound # TODO
    if filepath is None:
        filepath = "/home/ywshin/Deep-Reinforcement-Learning-Hands-On/SVNet/tests/nf.txt" # TODO
    with open(filepath, "r") as f:
        lines = f.readlines()
        for l in lines:
            features = l.split(",")
            if len(l) < 2:
                break
            nid = int(features[0])
            if nid not in NodeMap:
                NodeMap[nid] = bound
                bound += 1
            G.add_node(NodeMap[nid],
                nid=nid,
                isNext=0,
                isAllocated = int(features[1]),
                allocation = int(features[2]),
                weight = 100 if float(features[3]) > 100 else float(features[3]), # TODO
                size = int(features[4]),
                isPhysReg = int(features[5]),
                isIntReg = int(features[6]),
                isFloatReg = int(features[7]),
                allocOrder = list(map(lambda x: int(x), features[8].split()))
            )

def add_edges(G, filepath):
    if filepath is None:
        filepath = "/home/ywshin/Deep-Reinforcement-Learning-Hands-On/SVNet/tests/if.txt" # TODO
    with open(filepath, "r") as f:
        lines = f.readlines()
        for l in lines:
            node = NodeMap[int(l.split()[0])]
            edges = list(map(lambda x: NodeMap[int(x)], l.split()[1:]))
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

def get_initial_state():
    G = nx.Graph()
    set_node_features(G, None)
    add_edges(G, None)
    nid = get_next_node_id(G)
    G.nnid = nid
    G.nodes[nid]['isNext'] = 1
    return G

def state_to_int(G):
    s = []
    for _, attr in G.nodes(data=True):
        s.append(str(attr['allocation']))
    return ','.join(s)

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
    return state_new

def deepcopy_light(state):
    state_new = state.copy()
    # state_new.nnid = state.nnid
    for nid, attr in state.nodes(data=True):
        # state_new.nodes[nid]['nid'] = nid
        # state_new.nodes[nid]['isNext'] = attr['isNext']
        # state_new.nodes[nid]['isAllocated'] = attr['isAllocated']
        # state_new.nodes[nid]['allocation'] = attr['allocation']
        # state_new.nodes[nid]['weight'] = attr['weight']
        # state_new.nodes[nid]['size'] = attr['size']
        # state_new.nodes[nid]['isPhysReg'] = attr['isPhysReg']
        # state_new.nodes[nid]['isIntReg'] = attr['isIntReg']
        # state_new.nodes[nid]['isFloatReg'] = attr['isFloatReg']
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

    # start = time.time()
    # deepcopy(state)
    # end = time.time()
    # print("D:", end - start)
    # start = time.time()
    state_new = deepcopy_light(state) # MUST create deep copy of the graph
    # end = time.time()
    # print("S:", end - start)
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
