import math
import os
import sys

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from constants import *
from util import create_env
from model import ActorCritic
from torch.autograd import Variable

def train(env, model):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # TODO: Verify location of this
    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0

    while True:
        episode_length += 1
        # Sync with the shared model
        model.load_state_dict(model.state_dict())
        if done:
            memory = None
        else:
            memory = tuple(Variable(x.data) for x in memory)
        # Pick a new noise vector (until next optimisation step)
        model.sample_noise()

        values = []
        log_probs = []
        rewards = []

        for step in range(args.num_steps):
            value, logit, memory = model((Variable(state.unsqueeze(0)), memory))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            # entropy = -(log_prob * prob).sum(1)
            # entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))


            state, reward, done, _ = env.step(action.numpy()[0])

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), memory))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)

        # Standardize rewards
        rewards = np.array(rewards, dtype=float)
        rewards -= np.mean(rewards)
        reward_std = np.std(rewards)
        rewards /= reward_std if reward_std != 0 else 1

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss += 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss -= log_probs[i] * Variable(gae)
            # TODO: Remove entropy?
        optimizer.zero_grad()

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(model, model)
        optimizer.step()