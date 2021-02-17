import unittest

import numpy as np
from lib import allocation, model, utils, mcts
import ptan
import networkx as nx
import os
import torch
import collections

class TestSimpleIntegration(unittest.TestCase):
  def test_simple_integration(self):
    device = "cpu"
    net = model.Net(input_shape=model.OBS_SHAPE, actions_n=allocation.NUM_PHYS).to(device)
    best_net = ptan.agent.TargetNet(net)
    mcts_store = [mcts.MCTS(), mcts.MCTS()]
    replay_buffer = collections.deque(maxlen=100)
    results = set()

    for _ in range(30):
      result, step = model.play_game(mcts_store, replay_buffer, best_net.target_model, best_net.target_model,
                                      steps_before_tau_0=10, mcts_searches=10,
                                      mcts_batch_size=8, device=device, isTest=True)
      self.assertTrue(result == -1 or result == 0 or result == 1)
      self.assertGreater(step, 0)
      if result not in results:
        results.add(result)
      print("STEP: ", step)

    # print(mcts_store[0].visited_states)
    self.assertEqual(len(list(filter(lambda x: '-1' not in x.split(',') and '0' not in x.split(','), mcts_store[0].visited_states))), 26) # MCTS shoud visit all states with very high probability
    self.assertEqual(len(results), 3) # three results(win/draw/lose) should be occurred with very high probability
    self.assertGreater(len(replay_buffer), 0)
