import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MFConv, global_add_pool, global_mean_pool
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader
import math

from lib import allocation, mcts, utils

OBS_SHAPE = (allocation.NUM_NODE_FEATURES, allocation.NUM_NODE_FEATURES, allocation.NUM_NODE_FEATURES)
NUM_FILTERS = allocation.NUM_NODE_FEATURES

class Net(nn.Module):
    def __init__(self, input_shape, actions_n):
        super(Net, self).__init__()

        # TODO: Layer
        self.conv_in = GCNConv(input_shape[0], NUM_FILTERS)
        self.conv_1 = GCNConv(NUM_FILTERS, NUM_FILTERS)

        self.conv_val = MFConv(NUM_FILTERS, 1)
        self.conv_policy = GCNConv(NUM_FILTERS, allocation.NUM_PHYS)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv_in(x, edge_index)
        x = self.conv_1(x, edge_index)
        val = self.conv_val(x, edge_index)
        val = global_add_pool(val, data.batch)
        pol = self.conv_policy(x, edge_index)
        pol = global_mean_pool(pol, data.batch)
        return pol, val

def _convert_state(state):
    """
    In-place encodes list state into the zero numpy array
    :param dest_np: dest array, expected to be zero
    :param state: state
    """
    # assert dest_np.shape == OBS_SHAPE
    G = state
    edge_index = []
    x = []
    for N in G.nodes:
        for n in G.neighbors(N):
            edge_index.append([N, n])
        nf = [0] * allocation.NUM_NODE_FEATURES
        nf[0] = G.nodes[N]['isAllocated']
        for i in range(allocation.NUM_PHYS):
            reg = utils.renumber_reg(G.nodes[N]['allocation'])
            if reg == i:
                nf[1 + i] = 1
            else:
                nf[1 + i] = 0
        nf[1 + allocation.NUM_PHYS] = G.nodes[N]['weight']
        nf[2 + allocation.NUM_PHYS] = G.nodes[N]['size']
        nf[3 + allocation.NUM_PHYS] = G.nodes[N]['isPhysReg']
        nf[4 + allocation.NUM_PHYS] = G.nodes[N]['isIntReg']
        nf[5 + allocation.NUM_PHYS] = G.nodes[N]['isFloatReg']
        x.append(nf)
    x = torch.tensor(x, dtype=torch.float32)
    edge_index = torch.tensor(np.transpose(edge_index), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def state_list_to_batch(state_list, device="cpu"):
    """
    Convert list of list states to batch for network
    :param state_lists: list of 'list states'
    :return Variable with observations
    """
    return next(iter(DataLoader(list(map(lambda x: _convert_state(x), state_list)), batch_size=len(state_list)))).to(device) # TODO

def convert_action(G, action):
    nid = allocation.get_next_node_id(G)
    N = G.nodes(data=True)[nid]

    if len(N['allocOrder']) == 0:
        return -1

    found = False
    for r in N['allocOrder']:
        if utils.renumber_reg(r) == action:
            action = r
            found = True
            break
    assert found
    return action

def play_game(mcts_store, replay_buffer, net1, net2, steps_before_tau_0, mcts_searches, mcts_batch_size, device="cpu", isTest=False):
    """
    Play one single allocation, memorizing transitions into the replay buffer
    :param mcts_store: could be None or single MCTS or two MCTSes for individual net
    :param replay_buffer: queue with (state, probs, values), if None, nothing is stored
    :param net: allocator
    :return: value for the allocation in respect to allocator (+1 if allocator won, -1 if lost, 0 if draw)
    """
    assert isinstance(replay_buffer, (collections.deque, type(None)))
    assert isinstance(mcts_store, (mcts.MCTS, type(None), list))
    assert isinstance(net1, Net)
    assert isinstance(net2, Net)
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0
    assert isinstance(mcts_searches, int) and mcts_searches > 0
    assert isinstance(mcts_batch_size, int) and mcts_batch_size > 0

    if mcts_store is None:
        mcts_store = [mcts.MCTS(), mcts.MCTS()]

    states = [allocation.INITIAL_STATE, allocation.INITIAL_STATE] # TODO

    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    tau = 10000 if isTest else tau
    allocation_history = []

    results = [None, None]

    # get spill cost of net1
    while results[0] is None:
        mcts_store[0].search_batch(mcts_searches, mcts_batch_size, states[0], net1, device=device)
        probs, _ = mcts_store[0].get_policy_value(states[0], tau=tau)
        allocation_history.append((states[0], probs))
        action = np.random.choice(allocation.NUM_PHYS, p=probs)
        if len(allocation.possible_moves(states[0])) > 0:
            if action not in allocation.possible_moves(states[0]):
                print("debug")
                print(action)
                print(allocation.possible_moves(states[0]))
            assert action in allocation.possible_moves(states[0])
        else:
            action = 0

        # convert
        action = convert_action(states[0], action)

        states[0], spill_costs = allocation.move(states[0], action)
        if spill_costs >= 0:
            results[0] = spill_costs
            break
        step += 1
        if step >= steps_before_tau_0:
            tau = 0

    step = 0
    allocation_history = []

    # get spill cost of net2
    while results[1] is None:
        mcts_store[1].search_batch(mcts_searches, mcts_batch_size, states[1], net2, device=device)
        probs, _ = mcts_store[1].get_policy_value(states[1], tau=tau)
        allocation_history.append((states[1], probs))
        action = np.random.choice(allocation.NUM_PHYS, p=probs)
        if len(allocation.possible_moves(states[1])) > 0:
            assert action in allocation.possible_moves(states[1])
        else:
            action = 0

        # convert
        action = convert_action(states[1], action)

        states[1], spill_costs = allocation.move(states[1], action)
        if spill_costs >= 0:
            results[1] = spill_costs
            break
        step += 1
        if step >= steps_before_tau_0:
            tau = 0

    result = None
    if math.isclose(results[0], results[1]):
        result = 0
    elif results[0] < results[1]:
        result = 1
    else:
        result = -1

    if replay_buffer is not None:
        for state, probs in reversed(allocation_history):
            replay_buffer.append((state, probs, result))
    return result, step
