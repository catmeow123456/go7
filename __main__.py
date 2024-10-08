import json
import torch
from game import Board
from model import NNet, MCTS
TIME_LIMIT = 3.5


def get_input():
    return json.loads(input())


# 处理输入，还原棋盘
def initBoard(full_input) -> Board:
    requests = full_input["requests"]
    responses = full_input["responses"]
    board = Board()
    if requests[0]["x"] >= 0:
        board.place(requests[0]["x"], requests[0]["y"])
    turn = len(responses)
    for i in range(turn):
        board.place(responses[i]["x"], responses[i]["y"])
        board.place(requests[i + 1]["x"], requests[i + 1]["y"])
    return board


full_input = get_input()
board = initBoard(full_input)

saved_state = torch.load("data/cnn.pt", map_location="cpu")
nnet = NNet(0, 128, 256)
nnet.load_state_dict(saved_state)
mcts: MCTS = MCTS(nnet)
while True:
    if board.is_game_ended():
        break
    action = mcts.best_move(board, TIME_LIMIT)
    x, y = board.int2move(action)
    board.place(x, y)
    print(json.dumps({"response": {"x": x, "y": y}}))
    print("\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n", flush=True)

    if board.is_game_ended():
        break
    # print(board)
    full_input = get_input()
    board.place(full_input["x"], full_input["y"])
    # print(board)
