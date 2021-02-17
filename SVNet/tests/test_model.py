import unittest

import numpy as np
from lib import allocation, model, utils, mcts
import ptan
import networkx as nx
import os
import torch
import collections

class TestSimpleModel(unittest.TestCase):
    def setUp(self):
        self.G = nx.Graph()
        allocation.set_node_features(self.G, os.path.join(os.path.dirname(__file__), "nf.txt"))
        allocation.add_edges(self.G, os.path.join(os.path.dirname(__file__), "if.txt"))

    def _check_data(self, data):
        # check for data.x
        self.assertAlmostEqual(data.x[3][0], 0)
        for i in range(allocation.NUM_PHYS):
            self.assertAlmostEqual(data.x[3][1 + i], 0)
        self.assertAlmostEqual(data.x[3][1 + allocation.NUM_PHYS], 1)
        self.assertAlmostEqual(data.x[3][2 + allocation.NUM_PHYS], 2)
        self.assertAlmostEqual(data.x[3][3 + allocation.NUM_PHYS], 0)
        self.assertAlmostEqual(data.x[3][4 + allocation.NUM_PHYS], 1)
        self.assertAlmostEqual(data.x[3][5 + allocation.NUM_PHYS], 0)

        # check for data.edge_index
        l = []
        for x in {int(x) for x in '1 4 5'.split()}:
            l.append([3, x])
        l = np.transpose(l)
        t = np.transpose(data.edge_index.numpy())
        t = list(filter(lambda x: x[0] == 3, t))
        t = np.transpose(t)
        self.assertEqual(sorted(l[1]), sorted(t[1]))

    def test__convert_state(self):
        G = self.G
        data = model._convert_state(G)
        self._check_data(data)

    def test_state_list_to_batch(self):
        G = self.G
        batch = model.state_list_to_batch([G, G])
        self._check_data(batch)

    def test_simple_play_game(self):
        device = "cpu"
        net = model.Net(input_shape=model.OBS_SHAPE, actions_n=allocation.NUM_PHYS).to(device)
        best_net = ptan.agent.TargetNet(net)
        mcts_store = [mcts.MCTS(), mcts.MCTS()]
        replay_buffer = collections.deque(maxlen=100)

        result, step = model.play_game(mcts_store, replay_buffer, best_net.target_model, best_net.target_model,
                        steps_before_tau_0=10, mcts_searches=10,
                        mcts_batch_size=8, device=device)
        self.assertTrue(result == -1 or result == 0 or result == 1)
        self.assertGreater(step, 0)
        self.assertGreater(len(mcts_store), 0)
        self.assertGreater(len(replay_buffer), 0)

class TestModel(unittest.TestCase):
    def setUp(self):
        self.G = nx.Graph()
        allocation.set_node_features(self.G, os.path.join(os.path.dirname(__file__), "nf_bicubicKernel.txt"))
        allocation.add_edges(self.G, os.path.join(os.path.dirname(__file__), "if_bicubicKernel.txt"))

    def _check_data(self, data):
        # check for data.x
        self.assertAlmostEqual(data.x[46][0], 0)
        for i in range(allocation.NUM_PHYS):
            self.assertAlmostEqual(data.x[46][1 + i], 0)
        self.assertAlmostEqual(data.x[46][1 + allocation.NUM_PHYS], 0.054353)
        self.assertAlmostEqual(data.x[46][2 + allocation.NUM_PHYS], 2)
        self.assertAlmostEqual(data.x[46][3 + allocation.NUM_PHYS], 0)
        self.assertAlmostEqual(data.x[46][4 + allocation.NUM_PHYS], 1)
        self.assertAlmostEqual(data.x[46][5 + allocation.NUM_PHYS], 0)

        # check for data.edge_index
        l = []
        for x in {int(x) for x in '2147483650 2147483651 2147483654 2147483655 2147483656 2147483657 2147483658 2147483659 2147483666 2147483674 2147483681 2147483685 2147483688 2147483689 2147483690 2147483692 2147483694 2147483695 2147483696 2147483698 2147483700 2147483701 2147483702 2147483704 2147483706 2147483707 2147483708 2147483710 2147483711 2147483713 2147483714 2147483715 2147483717 2147483718 2147483719 2147483720 2147483722 2147483725 2147483726 2147483735 2147483737 2147483738 2147483740 2147483743 2147483752 2147483754 2147483755 2147483757 2147483760 2147483769 2147483771 2147483772 2147483774 2147483777 2147483786 2147483788 2147483789 2147483794 2147483803 2147483806 2147483807 2147483810 2147483811 2147483812 2147483813 2147483650 2147483651 2147483654 2147483655 2147483656 2147483657 2147483658 2147483659 2147483666 2147483674 2147483681 2147483685 2147483688 2147483689 2147483690 2147483692 2147483694 2147483695 2147483696 2147483698 2147483700 2147483701 2147483702 2147483704 2147483706 2147483707 2147483708 2147483710 2147483711 2147483713 2147483714 2147483715 2147483717 2147483718 2147483719 2147483720 2147483722 2147483725 2147483726 2147483735 2147483737 2147483738 2147483740 2147483743 2147483752 2147483754 2147483755 2147483757 2147483760 2147483769 2147483771 2147483772 2147483774 2147483777 2147483786 2147483788 2147483789 2147483794 2147483803 2147483806 2147483807 2147483810 2147483811 2147483812 2147483813'.split()}:
            l.append([2147483649, x])
        l = np.transpose(l)
        t = np.transpose(data.edge_index.numpy())
        t = list(filter(lambda x: x[0] == 2147483649, t))
        t = np.transpose(t)
        self.assertEqual(sorted(l[1]), sorted(t[1]))

    def test__convert_state(self):
        G = self.G
        data = model._convert_state(G)
        self._check_data(data)

    def test_state_list_to_batch(self):
        G = self.G

        batch = model.state_list_to_batch([G, G])
        self._check_data(batch)
