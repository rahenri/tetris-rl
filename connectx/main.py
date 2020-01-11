#!/usr/bin/env python3

import connectx_env
import agents


def main():
    board = connectx_env.Board(7, 7, 4)

    agent1 = agents.RandomAgent(1)
    agent2 = agents.RandomAgent(2)
    print(board)

    while True:
        move = agent1.act(board)
        over = board.move(move)
        print(board)
        if over:
            break

        move = agent2.act(board)
        over = board.move(move)
        print(board)
        if over:
            break

    print("game over")
    print(f"winner is {board.winner()}")


if __name__ == "__main__":
    main()
