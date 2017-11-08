import math
import os
import sys
import random
import math
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

import torch
import torch.nn.functional as F
import torch.optim as optim

from dataset import *
from constants import *
from model import *
from torch.autograd import Variable
from generate import Generation

bce = nn.BCEWithLogitsLoss()
cel = nn.CrossEntropyLoss()

def g_rollout(model, num_steps, batch_size=BATCH_SIZE):
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

    # Perform sequence generation #
    for step in range(num_steps):
        # print(state)
        state_vec = torch.zeros(batch_size, NUM_ACTIONS) if state is None else one_hot_batch(state, NUM_ACTIONS)
        state_vec = state_vec.unsqueeze(1)
        value, policy, memory = model(var(state_vec), None, memory, no_d=True)
        value = value.squeeze(1)
        policy = policy.squeeze(1)

        prob = F.softmax(policy)
        log_prob = F.log_softmax(policy)

        action = prob.multinomial().data
        log_prob = log_prob.gather(1, var(action))

        # Action become's next state
        state = action.cpu()
        states.append(state)

        values.append(value)
        log_probs.append(log_prob)
    return states, values, log_probs

def compute_rl_loss(generator, values, log_probs, R, num_steps):
    """
    Computes the loss from reinforcement learning (G rollouts)
    """
    # TODO: Should we append the standardized R or after?
    # Standardize rewards
    mean_rewards = R.mean()
    std_rewards = R.std()
    R -= mean_rewards
    if std_rewards.data[0] != 0:
        R /= std_rewards

    values.append(R)

    policy_loss = 0
    value_loss = 0

    batch_size = R.size(0)
    gae = var(torch.zeros(batch_size, 1))

    for i in reversed(range(num_steps)):
        R = DISCOUNT * R
        advantage = R - values[i]
        value_loss += 0.5 * advantage.pow(2)

        # Generalized Advantage Estimataion
        delta_t = DISCOUNT * values[i + 1] - values[i]
        gae = gae * DISCOUNT * TAU + delta_t
        # TODO: Need entropy?
        policy_loss -= log_probs[i] * gae

    loss = torch.sum(policy_loss + 0.5 * value_loss)
    return mean_rewards.data[0], loss

def compute_d_loss(model, fake_seqs, real_seqs):
    """
    Computes loss from the discriminator
    """
    fake_seqs_hot = one_hot_seq(torch.cat(fake_seqs, dim=1), NUM_ACTIONS)
    real_seqs_hot = one_hot_seq(real_seqs, NUM_ACTIONS)
    # First have all the fake sequences, then have the real sequences
    input_batch = var(torch.cat((fake_seqs_hot, real_seqs_hot), dim=0))
    _, p_output, d_output, _ = model(input_batch, None)

    # Create classes the first half the batch are fake = 0. Second half are real = 1.
    d_targets = var(torch.cat((torch.zeros(BATCH_SIZE, 1), torch.ones(BATCH_SIZE, 1))))
    d_loss = bce(d_output, d_targets)

    real_p_output = p_output[-real_seqs.size(0):, :-1].contiguous().view(-1, NUM_ACTIONS)
    mle_targets = var(real_seqs[:, 1:].contiguous().view(-1))
    mle_loss = cel(real_p_output, mle_targets)

    probs = F.sigmoid(d_output)
    log_probs = torch.log(probs)
    entropy = torch.mean(-probs * log_probs - (1 - probs) * torch.log(1 - probs))
    accuracy = (probs.round() == d_targets).sum().data[0] / (BATCH_SIZE * 2)

    # TODO: Log prob or prob?
    reward = var(log_probs[:BATCH_SIZE].data)

    return accuracy, reward, entropy.data[0], d_loss, mle_loss

def compute_mle_loss(model, data, validate=False):
    """
    Computes the MLE loss of the generator
    """
    # Convert all tensors into variables
    note_seq, styles = data
    # styles = var(one_hot_batch(styles, NUM_STYLES), volatile=validate)
    styles = None # TODO

    # Feed it to the model
    inputs = var(one_hot_seq(note_seq[:, :-1], NUM_ACTIONS), volatile=validate)
    targets = var(note_seq[:, 1:], volatile=validate)
    _, policy, _ = model(inputs, styles, None, no_d=True)

    # Compute the loss.
    loss = cel(policy.contiguous().view(-1, NUM_ACTIONS), targets.view(-1))
    return loss

def train(model, train_generator, val_generator, plot=True, gen_rate=0):
    # Construct optimizer
    opt = optim.Adam(model.parameters(), lr=LR)
    train_gen = train_generator()

    with tqdm() as tq:
        epoch = 0
        running_reward = None
        running_acc = None
        running_entropy = None
        target_steps = MIN_SEQ_LEN

        mle_loss = 0
        mle_train_loss = 0
        cl_counter = 0
        avg_reward = 0

        mle_train_losses = []
        mle_losses = []
        all_rewards = []
        all_accs = []

        for data in train_gen:
            epoch += 1
            cl_counter += 1

            model.train()
            opt.zero_grad()
            total_loss = 0

            # Gradually increase the number of steps
            # if running_entropy is not None and running_entropy > CL_THRESHOLD and cl_counter > MIN_EPOCH_CL:
            if cl_counter > MIN_EPOCH_CL:
                # Only increase timestep if the discriminator
                # is uncertain about the generator's output (convergence)
                target_steps = min(target_steps + 1, SEQ_LEN)
                cl_counter = 0
                
            num_steps = random.randint(MIN_SEQ_LEN, target_steps)
            
            # Cut out the data
            # TODO: Use a function to do this in data generator
            real_seqs, styles = data
            real_seqs = real_seqs[:, :num_steps]
            
            # Perform a rollout #
            fake_seqs, values, log_probs = g_rollout(model, num_steps)

            # Train the discriminator (and compute rewards)
            accuracy, reward, entropy, d_loss, mle_train_loss = compute_d_loss(model, fake_seqs, real_seqs)
            running_entropy = accumulate_running(running_entropy, entropy)
            running_acc = accumulate_running(running_acc, accuracy)
            
            # TODO: Modulate the MLE loss over time using some function?
            modulation = min(epoch / 1e5, 1)
            total_loss += mle_train_loss * (1 - modulation)

            # We don't train the generator if it is too good. Let G catch up.
            if running_acc < D_OPT_MAX_ACC:
                total_loss += d_loss

            # Train the generator (if the discriminator is decent)
            if running_acc > G_OPT_MIN_ACC:
                avg_reward, rl_loss = compute_rl_loss(model, values, log_probs, reward, num_steps)
                running_reward = accumulate_running(running_reward, avg_reward)
                total_loss += rl_loss * modulation

            # Perform gradient updates
            total_loss.backward()
            # TODO: Tune gradient clipping parameter?
            torch.nn.utils.clip_grad_norm(model.parameters(), GRADIENT_CLIP)
            opt.step()

            ## Statistics ##
            mle_train_loss = mle_train_loss.data[0]

            tq.set_postfix(len=target_steps, reward=running_reward, d_acc=running_acc, val_loss=mle_loss, entropy=running_entropy, train_loss=mle_train_loss, m=modulation)
            tq.update(1)

            if epoch % 50 == 0:
                # Statistic
                mle_loss = sum(compute_mle_loss(model, data, validate=True).data[0] for data in  itertools.islice(val_generator(), VAL_STEPS)) / VAL_STEPS
                mle_train_losses.append(mle_train_loss)
                mle_losses.append(mle_loss)
                all_rewards.append(avg_reward)
                all_accs.append(accuracy)

                if plot:
                    plot_loss(mle_losses, 'mle_loss')
                    plot_loss(mle_train_losses, 'mle_train_loss')
                    plot_loss(all_rewards, 'reward')
                    plot_loss(all_accs, 'accuracy')

            if epoch % 1000 == 0:
                # Save model
                torch.save(model.state_dict(), OUT_DIR + '/model_' + str(epoch) + '.pt')
                # Generate an output sequence
                Generation(model).export(name='epoch_' + str(epoch))

def plot_loss(validation_loss, name):
    # Draw graph
    plt.clf()
    plt.plot(validation_loss)
    plt.savefig(OUT_DIR + '/' + name)

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
        # torch.backends.cudnn.enabled = False
        model.cuda()

    if args.path:
        model.load_state_dict(torch.load(args.path))
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
    train(model, train_generator, val_generator, plot=not args.noplot, gen_rate=args.gen)

if __name__ == '__main__':
    main()
