import unittest
import networkx as nx
import numpy as np
import math
import os

from lib import allocation, utils

class TestSimpleGraph(unittest.TestCase):
    def setUp(self):
        self.G = nx.Graph()
        allocation.set_node_features(self.G, os.path.join(os.path.dirname(__file__), "nf.txt"))
        allocation.add_edges(self.G, os.path.join(os.path.dirname(__file__), "if.txt"))

    def test_set_node_features(self):
        G = self.G
        N = G.nodes[3]
        self.assertEqual(N['isAllocated'], 0)
        self.assertEqual(N['allocation'], 0)
        self.assertTrue(math.isclose(N['weight'], 1))
        self.assertEqual(N['size'], 2)
        self.assertEqual(N['isPhysReg'], 0)
        self.assertEqual(N['isIntReg'], 1)
        self.assertEqual(N['isFloatReg'], 0)
        self.assertListEqual(sorted(N['allocOrder']), sorted([int(x) for x in '52 54 57'.split()]))

        N = G.nodes[1]
        self.assertEqual(N['isAllocated'], 1)
        self.assertEqual(N['allocation'], 49)
        self.assertTrue(math.isclose(N['weight'], 1))
        self.assertEqual(N['size'], 1)
        self.assertEqual(N['isPhysReg'], 1)
        self.assertEqual(N['isIntReg'], 1)
        self.assertEqual(N['isFloatReg'], 0)
        self.assertListEqual(sorted(N['allocOrder']), sorted([int(x) for x in ''.split()]))

    def test_add_edges(self):
        G = self.G
        self.assertListEqual(sorted(list(G.neighbors(3))), sorted(list({int(x) for x in '1 4 5'.split()})))

    def test_get_next_node_id(self):
        G = self.G
        nodes = list(filter(lambda x: not G.nodes[x]['isAllocated'], G.nodes))
        n = len(nodes)
        for _ in range(n):
            N = allocation.get_next_node_id(G)
            self.assertTrue(N in nodes)
            nodes.remove(N)
            G.remove_node(N)
        self.assertEqual(len(nodes), 0)

    def test_move_and_possible_moves(self):
        G = self.G
        l1 = allocation.possible_moves(G, 5)
        l2 = [utils.renumber_reg(idx) for idx in [52, 54]]
        self.assertListEqual(sorted(l1), sorted(l2))

        G, c = allocation.move(G, 52, 3)
        self.assertTrue(isinstance(G, nx.Graph))
        self.assertEqual(c, -1)
        l1 = allocation.possible_moves(G, 5)
        l2 = [utils.renumber_reg(idx) for idx in [54]]
        self.assertListEqual(sorted(l1), sorted(l2))

        # force node(5) spilled
        G, c = allocation.move(G, 54, 4)
        G, c = allocation.move(G, 57, 6)
        G, c = allocation.move(G, 54, 0)
        self.assertEqual(c, 1)

    def test_state_to_int(self):
        G = self.G
        self.assertEqual('0,49,52,0,0,0,0', allocation.state_to_int(G))

class TestGraph(unittest.TestCase):
    def setUp(self):
        self.G = nx.Graph()
        allocation.set_node_features(self.G, os.path.join(os.path.dirname(__file__), "nf_bicubicKernel.txt"))
        allocation.add_edges(self.G, os.path.join(os.path.dirname(__file__), "if_bicubicKernel.txt"))

    def test_set_node_features(self):
        G = self.G
        N = G.nodes[2147483685]
        self.assertEqual(N['isAllocated'], 0)
        self.assertEqual(N['allocation'], 0)
        self.assertTrue(math.isclose(N['weight'], 4.03901))
        self.assertEqual(N['size'], 3)
        self.assertEqual(N['isPhysReg'], 0)
        self.assertEqual(N['isIntReg'], 1)
        self.assertEqual(N['isFloatReg'], 0)
        self.assertListEqual(sorted(N['allocOrder']), sorted([int(x) for x in '49 52 54 57 53 127 128 129 130 51 133 134 131 132 50'.split()]))

    def test_add_edges(self):
        G = self.G
        self.assertListEqual(sorted(list(G.neighbors(2147483666))), sorted(list({int(x) for x in '2147483649 2147483650 2147483651 2147483654 2147483655 2147483656 2147483657 2147483658 2147483659 2147483649 2147483650 2147483651 2147483654 2147483655 2147483656 2147483657 2147483658 2147483659 2147483672 2147483674 2147483675 2147483681 2147483685 2147483688 2147483689 2147483690 2147483692 2147483694 2147483695 2147483696 2147483698 2147483700 2147483701 2147483702 2147483704 2147483706 2147483707 2147483708 2147483710 2147483711 2147483713 2147483714 2147483715 2147483717 2147483718 2147483719 2147483720 2147483722 2147483725 2147483726 2147483735 2147483737 2147483738 2147483740 2147483743 2147483752 2147483754 2147483755 2147483757 2147483760 2147483769 2147483771 2147483772 2147483774 2147483777 2147483786 2147483788 2147483789 2147483794 2147483803 2147483806 2147483807 2147483810 2147483811 2147483812 2147483813 57 53 2147483672 2147483674 2147483675 2147483681 2147483685 2147483688 2147483689 2147483690 2147483692 2147483694 2147483695 2147483696 2147483698 2147483700 2147483701 2147483702 2147483704 2147483706 2147483707 2147483708 2147483710 2147483711 2147483713 2147483714 2147483715 2147483717 2147483718 2147483719 2147483720 2147483722 2147483725 2147483726 2147483735 2147483737 2147483738 2147483740 2147483743 2147483752 2147483754 2147483755 2147483757 2147483760 2147483769 2147483771 2147483772 2147483774 2147483777 2147483786 2147483788 2147483789 2147483794 2147483803 2147483806 2147483807 2147483810 2147483811 2147483812 2147483813'.split()})))

    def test_get_next_node_id(self):
        G = self.G
        inf_nodes = [2147483675, 2147483681, 2147483711, 2147483754, 2147483771, 2147483788, 2147483803]
        n = len(inf_nodes)
        for _ in range(n):
            N = allocation.get_next_node_id(G)
            self.assertTrue(N in inf_nodes)
            inf_nodes.remove(N)
            G.remove_node(N)

        N = allocation.get_next_node_id(G)
        self.assertEqual(N, 2147483794)
        G.remove_node(N)

        N = allocation.get_next_node_id(G)
        self.assertEqual(N, 2147483777)
        G.remove_node(N)

# TODO: test for simple cases
