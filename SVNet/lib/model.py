import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MFConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader
import math

from lib import allocation, mcts, utils

OBS_SHAPE = (allocation.NUM_NODE_FEATURES, allocation.NUM_NODE_FEATURES, allocation.NUM_NODE_FEATURES)
NUM_FILTERS = 32

class Net(nn.Module):
    def __init__(self, input_shape, actions_n):
        super(Net, self).__init__()

        # TODO: Layer
        self.conv_0 = GCNConv(input_shape[0], NUM_FILTERS)
        self.conv_1 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_2 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_3 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_4 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_5 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_6 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_7 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_8 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_9 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_10 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_11 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_12 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_13 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_14 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_15 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_16 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_17 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_18 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_19 = GCNConv(NUM_FILTERS, NUM_FILTERS)
        self.conv_20 = GCNConv(NUM_FILTERS, NUM_FILTERS)

        self.conv_val = MFConv(NUM_FILTERS, NUM_FILTERS)
        self.value = nn.Sequential(
            nn.Linear(NUM_FILTERS, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Tanh()
        )
        self.conv_policy = GCNConv(NUM_FILTERS, 128)
        self.policy = nn.Sequential(
            nn.Linear(128, allocation.NUM_PHYS),
            nn.ReLU(),
            nn.Linear(allocation.NUM_PHYS, allocation.NUM_PHYS),
            nn.ReLU()
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = x.size()[0]
        x = self.conv_0(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_6(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_7(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_8(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_9(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_10(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_11(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_12(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_13(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_14(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_15(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_16(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_17(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_18(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv_19(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        val = self.conv_val(x, edge_index)
        val = F.relu(val)
        val = F.dropout(val, training=self.training)
        val = global_max_pool(val, data.batch)
        val = F.relu(val)
        val = self.value(val)

        pol = self.conv_policy(x, edge_index)
        pol = F.relu(pol)
        pol = F.dropout(pol, training=self.training)
        pol = global_mean_pool(pol, data.batch)
        val = F.relu(val)
        pol = self.policy(pol)

        return pol, val

ConvertedStates = {}

def _convert_state(state):
    """
    In-place encodes list state into the zero numpy array
    :param dest_np: dest array, expected to be zero
    :param state: state
    """
    state_int = allocation.state_to_int(state)
    if ConvertedStates.get(state_int, None) is not None:
        return Data(x=ConvertedStates[state_int][0].clone().detach(), edge_index=ConvertedStates[state_int][1].clone().detach())
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
        nf[6 + allocation.NUM_PHYS] = G.nodes[N]['isVecReg']
        nf[7 + allocation.NUM_PHYS] = G.nodes[N]['isNext']
        x.append(nf)
    x = torch.tensor(x, dtype=torch.float32)
    edge_index = torch.tensor(np.transpose(edge_index), dtype=torch.long)

    ConvertedStates[state_int] = (x, edge_index)

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

    initial_state = allocation.get_initial_state()
    # states = [allocation.INITIAL_STATE, allocation.INITIAL_STATE] # TODO
    states = [initial_state, initial_state] # TODO

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
