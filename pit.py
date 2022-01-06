import os

import Arena
from MCTS import MCTS
from gobang.GobangGame import GobangGame
from gobang.GobangPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet
import re
import argparse
import numpy as np
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('-r', help="是否人类对局", action='store_true', required=False)
parser.add_argument('--times', help="对局次数", required=False, type=int, default=3)


args = parser.parse_args()

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = args.r

if mini_othello:
    g = GobangGame(6)
else:
    g = GobangGame(6)

# all players
rp = RandomPlayer(g).play
gp = GreedyGobangPlayer(g).play
hp = HumanGobangPlayer(g).play


def get_model(times: int):
    """
    给定times，输出模型名词
    @param times:
    @return:
    """
    return f"best_{times}.pth.tar"

# nnet players
n1 = NNet(g)
if mini_othello:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
else:
    pattern = re.compile(r"best_(\d+).pth.tar")
    bestModels = os.listdir('./temp/')
    bestModels.sort(key=lambda x: int(pattern.findall(x)[0]), reverse=True)

    n1.load_checkpoint('./temp/', bestModels[0])
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./temp/', get_model(0))
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

# 正常玩
# arena = Arena.Arena(n1p, player2, g, display=GobangGame.display)

# 跟随机的玩
arena = Arena.Arena(n1p, rp, g, display=GobangGame.display)

# verbose：显示交战画面
print(arena.playGames(args.times, verbose=False))
