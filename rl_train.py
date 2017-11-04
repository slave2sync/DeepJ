import math
import os
import sys
import random
import math
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from dataset import *
from constants import *
from model import *
from torch.autograd import Variable

# Discount factor
discount = 0.99
# GAE parameter
tau = 1.00

min_num_steps = 4

def g_rollout(model, num_steps):
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

def g_train(generator, optimizer, num_steps):
    generator.train()
    optimizer.zero_grad()
    # Perform a rollout #
    states, values, log_probs, entropies = g_rollout(generator, num_steps)

    # Finished sequence generation. Now compute rewards!
    # TODO: Simple reward scheme
    R = var(torch.zeros(BATCH_SIZE, 1))

    for state in states:
        for i, b in enumerate(state):
            if b[0] == state[i - 1][0] + 1:
                R[i, 0] = R[i, 0] + 1

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
    torch.nn.utils.clip_grad_norm(generator.parameters(), GRADIENT_CLIP)

    optimizer.step()
    return mean_rewards.data[0], states

def d_train(discriminator, optimizer, fake_seqs, real_seqs):
    optimizer.zero_grad()
    criterion = nn.BCEWithLogitsLoss()

    fake_seqs = one_hot_seq(torch.cat(fake_seqs, dim=1), NUM_ACTIONS)
    real_seqs = one_hot_seq(real_seqs, NUM_ACTIONS)
    input_batch = var(torch.cat((fake_seqs, real_seqs), dim=0))
    outputs, _ = discriminator(input_batch, None)

    # Create classes the first half the batch are fake = 0. Second half are real = 1.
    targets = var(torch.cat((torch.zeros(BATCH_SIZE, 1), torch.ones(BATCH_SIZE, 1))))
    accuracy = (outputs.round() == targets).sum().data[0] / (BATCH_SIZE * 2)

    loss = criterion(outputs, targets)
    loss.backward()

    optimizer.step()
    return accuracy

def train(generator, discriminator, train_generator, val_generator, plot, gen_rate):
    # Construct optimizer
    g_opt = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    d_opt = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    train_gen = train_generator()

    with tqdm() as tq:
        iteration = 0
        running_reward = 0

        for real_seqs, styles in train_gen:
            iteration += 1

            target_steps = min_num_steps

            # Gradually increase the number of steps
            # TODO: Raise sequence length bounds
            target_steps = min(max(int(running_reward * SEQ_LEN), min_num_steps), SEQ_LEN)
            num_steps = random.randint(min_num_steps, target_steps)

            # Train the generator
            avg_reward, fake_seqs = g_train(generator, g_opt, num_steps)
            # Train the discriminator
            real_seqs = real_seqs[:, :num_steps]
            accuracy = d_train(discriminator, d_opt, fake_seqs, real_seqs)

            if iteration == 1:
                running_reward = avg_reward
            else:
                running_reward = avg_reward * 0.01 + running_reward * 0.99

            tq.set_postfix(len=num_steps, reward=running_reward, d_acc=accuracy)
            tq.update(BATCH_SIZE)

def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--path', help='Load existing model?')
    parser.add_argument('--gen', default=0, type=int, help='Generate per how many epochs?')
    parser.add_argument('--noplot', default=False, action='store_true', help='Do not plot training/loss graphs')
    args = parser.parse_args()

    print('=== Loading Model ===')
    print('GPU: {}'.format(torch.cuda.is_available()))
    generator = DeepJG()
    discriminator = DeepJD()

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()

    if args.path:
        generator.load_state_dict(torch.load(args.path))
        print('Restored model from checkpoint.')

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
    train(generator, discriminator, train_generator, val_generator, plot=not args.noplot, gen_rate=args.gen)

if __name__ == '__main__':
    main()
