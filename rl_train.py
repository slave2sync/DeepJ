import math
import os
import sys

import math
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from dataset import *
from constants import *
from model import DeepJ
from torch.autograd import Variable

num_steps = 8
# Discount factor
discount = 0.99
# GAE parameter
tau = 1.00

def g_rollout(model):
    """
    Rollout a sequence
    """
    # Init state to zero vector
    state = None
    # Reset memory
    memory = None
    # TODO: Do we need noise regularization?
    # Pick a new noise vector (until next optimisation step)
    # model.sample_noise()

    states = []
    values = []
    log_probs = []
    entropies = []

    # Perform sequence generation #
    for step in range(num_steps):
        state_vec = torch.zeros(BATCH_SIZE, NUM_ACTIONS) if state is None else one_hot_batch(state, NUM_ACTIONS)
        state_vec = state_vec.unsqueeze(1)
        value, logit, memory = model(var(state_vec), None, memory)
        value = value.squeeze(1)
        logit = logit.squeeze(1)

        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)

        entropy = -(log_prob * prob).sum(1, keepdim=True)
        entropies.append(entropy)

        # Sample actions
        action = prob.multinomial().data
        log_prob = log_prob.gather(1, var(action))

        # Action become's next state
        state = action.cpu()
        states.append(state)

        values.append(value)
        log_probs.append(log_prob)
    return states, values, log_probs, entropies

def g_train(model, optimizer, plot, gen_rate):
    with tqdm() as tq:
        running_reward = 0

        while True:
            model.train()
            optimizer.zero_grad()
            # Perform a rollout #
            states, values, log_probs, entropies = g_rollout(model)

            # Finished sequence generation. Now compute rewards!
            # TODO: Simple reward scheme
            R = var(torch.zeros(BATCH_SIZE, 1))

            for state in states[1:]:
                for i, b in enumerate(state):
                    if b[0] > state[i - 1][0]:
                        R[i, 0] = 1

            values.append(R)

            policy_loss = 0
            value_loss = 0

            gae = var(torch.zeros(BATCH_SIZE, 1))

            # Standardize rewards
            mean_rewards = R.mean()
            std_rewards = R.std()
            R -= mean_rewards
            if std_rewards.data[0] != 0:
                R /= std_rewards

            for i in reversed(range(num_steps)):
                R = discount * R
                advantage = R - values[i]
                value_loss += 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = discount * values[i + 1] - values[i]
                gae = gae * discount * tau + delta_t
                # TODO: Need entropy?
                policy_loss -= log_probs[i] * gae + 0.01 * entropies[i]

            loss = torch.sum(policy_loss + 0.5 * value_loss)
            loss.backward()
            # TODO: Tune gradient clipping parameter?
            torch.nn.utils.clip_grad_norm(model.parameters(), GRADIENT_CLIP)

            optimizer.step()

            running_reward = mean_rewards.data[0] * 0.01 + running_reward * 0.99
            tq.set_postfix(loss=loss.data[0], reward=running_reward)
            tq.update(BATCH_SIZE)
def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--path', help='Load existing model?')
    parser.add_argument('--gen', default=0, type=int, help='Generate per how many epochs?')
    parser.add_argument('--noplot', default=False, action='store_true', help='Do not plot training/loss graphs')
    args = parser.parse_args()

    print('=== Loading Model ===')
    print('GPU: {}'.format(torch.cuda.is_available()))
    model = DeepJ()

    if torch.cuda.is_available():
        model.cuda()

    if args.path:
        model.load_state_dict(torch.load(args.path))
        print('Restored model from checkpoint.')

    # Construct optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print()

    print('=== Dataset ===')
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Loading data...')
    data = process(load())
    print()
    print('Creating data generators...')
    train_data, val_data = validation_split(data)
    train_generator = lambda: batcher(sampler(train_data))
    val_generator = lambda: batcher(sampler(val_data))

    """
    # Checks if training data sounds right.
    for i, (train_seq, *_) in enumerate(train_generator()):
        save_midi('train_seq_{}'.format(i), train_seq[0].cpu().numpy())
    """

    print('Training Sequences:', len(train_data[0]), 'Validation Sequences:', len(val_data[0]))
    print()

    print('=== Training ===')
    g_train(model, optimizer, plot=not args.noplot, gen_rate=args.gen)

if __name__ == '__main__':
    main()
