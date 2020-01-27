#!/usr/bin/env python3

import train
import nnagent


def main():
    board_shape = [6, 7]

    agent = nnagent.NNAgent(board_shape)
    agent.load_model("experiments/test7/model_snapshots/model.npz")
    env = train.Env(board_shape, k=4)

    while True:
        action, action_score = agent.act(env, randomize=False, verbose=True)
        print(env)
        print(f"action: {action} score:{action_score}")

        reward, done = env.step(action)
        if done:
            print(reward)
            break


if __name__ == "__main__":
    main()
