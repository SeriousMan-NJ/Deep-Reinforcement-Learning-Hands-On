"""
Monte-Carlo Tree Search
"""
import math as m
import time
import numpy as np

from lib import allocation, model, utils

import torch.nn.functional as F


class MCTS:
    """
    Class keeps statistics for every state encountered during the search
    """
    def __init__(self, c_puct=1.0):
        self.c_puct = c_puct
        # count of visits, state_int -> [N(s, a)]
        self.visit_count = {}
        # total value of the state's action, state_int -> [W(s, a)]
        self.value = {}
        # average value of actions, state_int -> [Q(s, a)]
        self.value_avg = {}
        # prior probability of actions, state_int -> [P(s,a)]
        self.probs = {}
        self.best_costs = np.inf
        self.visited_states = set()

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()

    def __len__(self):
        return len(self.value)

    def find_leaf(self, state):
        """
        Traverse the tree until the end of game or leaf node
        :param state: root node state
        :return: tuple of (value, leaf_state, states, actions)
        1. value: None if leaf node, otherwise equals to the allocation outcome at leaf
        2. leaf_state: state of the last state
        3. states: list of states traversed
        4. list of actions taken
        """
        states = []
        actions = []
        cur_state = state
        cur_state_int = allocation.state_to_int(state)
        root_state_int = allocation.state_to_int(state)
        value = None

        while not self.is_leaf(cur_state_int):
            states.append(cur_state)

            counts = self.visit_count[cur_state_int]
            total_sqrt = m.sqrt(sum(counts))
            probs = self.probs[cur_state_int]
            values_avg = self.value_avg[cur_state_int]

            # choose action to take, in the root node add the Dirichlet noise to the probs
            if cur_state_int == root_state_int:
                noises = np.random.dirichlet([0.03] * allocation.NUM_PHYS)
                probs = [0.75 * prob + 0.25 * noise for prob, noise in zip(probs, noises)]
            score = [value + self.c_puct * prob * total_sqrt / (1 + count)
                     for value, prob, count in zip(values_avg, probs, counts)]
            invalid_actions = set(range(len(utils.X86.F_ALL))) - set(allocation.possible_moves(cur_state))
            for invalid in invalid_actions:
                score[invalid] = -np.inf
            action = int(np.argmax(score))
            actions.append(action)

            # convert
            action = model.convert_action(cur_state, action)

            cur_state, spill_costs = allocation.move(cur_state, action)
            cur_state_int = allocation.state_to_int(cur_state)

            if spill_costs >= 0:
                if m.isclose(spill_costs, self.best_costs):
                    value = 0.0
                elif spill_costs < self.best_costs:
                    self.best_costs = spill_costs
                    value = 1.0
                else:
                    value = -1.0

        return value, cur_state, states, actions

    def is_leaf(self, state_int):
        return state_int not in self.probs

    def search_batch(self, count, batch_size, state, net, device="cpu"):
        for _ in range(count):
            self.search_minibatch(batch_size, state, net, device)

    def search_minibatch(self, count, state, net, device="cpu"):
        """
        Perform several MCTS searches.
        """
        backup_queue = []
        expand_states = []
        expand_queue = []
        planned = set()
        for _ in range(count):
            value, leaf_state, states, actions = self.find_leaf(state)
            leaf_state_int = allocation.state_to_int(leaf_state)
            if leaf_state_int not in self.visited_states:
                self.visited_states.add(leaf_state_int)
            if value is not None:
                backup_queue.append((value, states, actions))
            else:
                if leaf_state_int not in planned:
                    planned.add(leaf_state_int)
                    expand_states.append(leaf_state)
                    expand_queue.append((leaf_state, states, actions))

        # do expansion of nodes
        if len(expand_queue) > 0:
            batch_v = model.state_list_to_batch(expand_states, device)
            start = time.time()
            logits_v, values_v = net(batch_v)
            end = time.time()
            print(end - start)
            probs_v = F.softmax(logits_v, dim=1)
            values = values_v.data.cpu().numpy()[:, 0]
            probs = probs_v.data.cpu().numpy()

            # create the nodes
            for (leaf_state, states, actions), value, prob in zip(expand_queue, values, probs):
                leaf_state_int = allocation.state_to_int(leaf_state)
                self.visit_count[leaf_state_int] = [0] * allocation.NUM_PHYS
                self.value[leaf_state_int] = [0.0] * allocation.NUM_PHYS
                self.value_avg[leaf_state_int] = [0.0] * allocation.NUM_PHYS
                self.probs[leaf_state_int] = prob
                backup_queue.append((value, states, actions))

        # perform backup of the searches
        for value, states, actions in backup_queue:
            # leaf state is not stored in states and actions, so the value of the leaf will be the value of the opponent
            for state, action in zip(states[::-1], actions[::-1]):
                state_int = allocation.state_to_int(state)
                self.visit_count[state_int][action] += 1
                self.value[state_int][action] += value
                self.value_avg[state_int][action] = self.value[state_int][action] / self.visit_count[state_int][action]

    def get_policy_value(self, state, tau=1):
        """
        Extract policy and action-values by the state
        :param state_int: state of the board
        :return: (probs, values)
        """
        state_int = allocation.state_to_int(state)
        counts = self.visit_count[state_int]
        if tau == 0:
            probs = [0.0] * allocation.NUM_PHYS
            probs[np.argmax(counts)] = 1.0
        else:
            counts = [count ** (1.0 / tau) for count in counts]
            total = sum(counts)
            probs = [count / total for count in counts]
        values = self.value_avg[state_int]
        return probs, values
