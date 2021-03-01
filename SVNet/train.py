#!/usr/bin/env python3
import os
import time
import ptan
import random
import argparse
import collections

from lib import allocation, model, mcts

from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import multiprocessing
import traceback


PLAY_EPISODES = 64  # TODO: 25
MCTS_SEARCHES = 2 # TODO 10
MCTS_BATCH_SIZE = 8 # TODO: 8
REPLAY_BUFFER = 1000000
LEARNING_RATE = 0.01
BATCH_SIZE = 16
TRAIN_ROUNDS = 100
MIN_REPLAY_TO_TRAIN = 20 # TODO: 10000

BEST_NET_WIN_RATIO = 0.55

EVALUATE_EVERY_STEP = 10 # TODO: 200
EVALUATION_ROUNDS = 10 # TODO: 20
STEPS_BEFORE_TAU_0 = 10

def error_callback(e):
    traceback.print_exception(type(e), e, e.__traceback__)

def evaluate(net1, net2, rounds, device="cpu"):
    n1_win, n2_win = 0, 0

    for i in range(rounds):
        allocation.change_file_index()
        mcts_store = [mcts.MCTS(), mcts.MCTS()]
        r, _ = model.play_game(mcts_store=mcts_store, replay_buffer=None, net1=net1, net2=net2,
                               steps_before_tau_0=0, mcts_searches=2, mcts_batch_size=1,
                               device=device)
        print("Step {} (EVAL), result: {}".format(i, r))
        if r < -0.5:
            n2_win += 1
        elif r > 0.5:
            n1_win += 1

    if n1_win + n2_win == 0:
        return 0

    return n1_win / (n1_win + n2_win)


if __name__ == "__main__":
    manager = multiprocessing.Manager()

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = os.path.join("saves", args.name)
    os.makedirs(saves_path, exist_ok=True)
    writer = SummaryWriter(comment="-" + args.name)

    net = model.Net(input_shape=model.OBS_SHAPE, actions_n=allocation.NUM_PHYS).to(device)
    best_net = ptan.agent.TargetNet(net)
    print(net)

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)
    step_idx = 0
    best_idx = 0

    with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
        while True:
            pool = multiprocessing.Pool(PLAY_EPISODES)
            rb = manager.list()
            print(allocation.Files[allocation.File_index])
            mcts_store = [mcts.MCTS(), mcts.MCTS()]

            t = time.time()
            prev_nodes = len(mcts_store[0])
            game_steps = 0
            for i in range(PLAY_EPISODES):
                pool.apply_async(model.play_game, args=(None, rb, best_net.target_model, best_net.target_model,
                                STEPS_BEFORE_TAU_0, MCTS_SEARCHES, MCTS_BATCH_SIZE, device), error_callback=error_callback)
                # _, steps = model.play_game(mcts_store, replay_buffer, best_net.target_model, best_net.target_model,
                #                            steps_before_tau_0=STEPS_BEFORE_TAU_0, mcts_searches=MCTS_SEARCHES,
                #                            mcts_batch_size=MCTS_BATCH_SIZE, lock=lock, device=device)
                # game_steps += steps
            pool.close()
            pool.join()

            replay_buffer += rb

            game_nodes = len(mcts_store[0]) - prev_nodes
            dt = time.time() - t
            speed_steps = game_steps / dt
            speed_nodes = game_nodes / dt
            tb_tracker.track("speed_steps", speed_steps, step_idx)
            tb_tracker.track("speed_nodes", speed_nodes, step_idx)
            print("Step %d, best_idx %d, replay %d" % (step_idx, best_idx, len(replay_buffer)))
            step_idx += 1

            if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
                continue

            # train
            sum_loss = 0.0
            sum_value_loss = 0.0
            sum_policy_loss = 0.0

            for _ in range(TRAIN_ROUNDS):
                batch = random.sample(replay_buffer, BATCH_SIZE)
                batch_states, batch_probs, batch_values = zip(*batch)
                batch_states_list = [state for state in batch_states]
                states_v = model.state_list_to_batch(batch_states_list, device)

                optimizer.zero_grad()
                probs_v = torch.FloatTensor(batch_probs).to(device)
                values_v = torch.FloatTensor(batch_values).to(device)
                out_logits_v, out_values_v = net(states_v)

                loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)

                loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
                loss_policy_v = loss_policy_v.sum(dim=1).mean()

                # print(out_values_v.squeeze(-1))
                # print(values_v)
                # print(out_logits_v)
                # print(probs_v)

                loss_v = loss_policy_v + loss_value_v
                loss_v.backward()
                optimizer.step()
                sum_loss += loss_v.item()
                sum_value_loss += loss_value_v.item()
                sum_policy_loss += loss_policy_v.item()

            print("loss_total:  {}".format(sum_loss / TRAIN_ROUNDS))
            print("loss_value:  {}".format(sum_value_loss / TRAIN_ROUNDS))
            print("loss_policy: {}".format(sum_policy_loss / TRAIN_ROUNDS))

            tb_tracker.track("loss_total", sum_loss / TRAIN_ROUNDS, step_idx)
            tb_tracker.track("loss_value", sum_value_loss / TRAIN_ROUNDS, step_idx)
            tb_tracker.track("loss_policy", sum_policy_loss / TRAIN_ROUNDS, step_idx)

            # evaluate net
            if step_idx % EVALUATE_EVERY_STEP == 0:
                win_ratio = evaluate(net, best_net.target_model, rounds=EVALUATION_ROUNDS, device=device)
                print("Net evaluated, win ratio = %.2f" % win_ratio)
                writer.add_scalar("eval_win_ratio", win_ratio, step_idx)
                if win_ratio > BEST_NET_WIN_RATIO:
                    print("Net is better than cur best, sync")
                    best_net.sync()
                    best_idx += 1
                    file_name = os.path.join(saves_path, "best_%03d_%05d.dat" % (best_idx, step_idx))
                    torch.save(net.state_dict(), file_name)
                    # for s in mcts_store:
                    #     s.clear()

            for s in mcts_store:
                s.clear()

            allocation.change_file_index()
