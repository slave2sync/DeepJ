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

def g_rollout(model, num_steps, override_seqs=None, batch_size=BATCH_SIZE):
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
        state_vec = torch.zeros(batch_size, NUM_ACTIONS) if state is None else one_hot_batch(state, NUM_ACTIONS)
        state_vec = state_vec.unsqueeze(1)
        value, logit, memory = model(var(state_vec), None, memory)
        value = value.squeeze(1)
        logit = logit.squeeze(1)

        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)

        # Sample actions
        if override_seqs is not None:
            # Forcefully sample the right sequence to get positive rewards
            action = override_seqs[:, step:step+1]
        else:
            action = prob.multinomial().data
        log_prob = log_prob.gather(1, var(action))

        # Action become's next state
        state = action.cpu()
        states.append(state)

        values.append(value)
        log_probs.append(log_prob)
    return states, values, log_probs

def g_train(generator, optimizer, values, log_probs, R, num_steps):
    generator.train()

    values.append(R)

    policy_loss = 0
    value_loss = 0

    batch_size = R.size(0)
    gae = var(torch.zeros(batch_size, 1))

    # Standardize rewards
    mean_rewards = R.mean()
    std_rewards = R.std()
    R -= mean_rewards
    if std_rewards.data[0] != 0:
        R /= std_rewards

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
    loss.backward()
    # TODO: Tune gradient clipping parameter?
    torch.nn.utils.clip_grad_norm(generator.parameters(), GRADIENT_CLIP)

    optimizer.step()
    optimizer.zero_grad()
    return mean_rewards.data[0]

def d_train(discriminator, optimizer, fake_seqs, real_seqs, optimize=True):
    """
    Trains the generator
    """
    discriminator.train()
    optimizer.zero_grad()
    criterion = nn.BCEWithLogitsLoss()

    fake_seqs = one_hot_seq(torch.cat(fake_seqs, dim=1), NUM_ACTIONS)
    real_seqs = one_hot_seq(real_seqs, NUM_ACTIONS)
    input_batch = var(torch.cat((fake_seqs, real_seqs), dim=0))
    outputs, _ = discriminator(input_batch, None)

    # Create classes the first half the batch are fake = 0. Second half are real = 1.
    targets = var(torch.cat((torch.zeros(BATCH_SIZE, 1), torch.ones(BATCH_SIZE, 1))))

    probs = F.sigmoid(outputs)
    log_probs = torch.log(probs)
    entropy = torch.mean(-probs * log_probs - (1 - probs) * torch.log(1 - probs))
    accuracy = (probs.round() == targets).sum().data[0] / (BATCH_SIZE * 2)

    # TODO: Log prob or prob?
    reward = var(log_probs[:BATCH_SIZE].data)

    if optimize:
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
    return accuracy, reward, entropy.data[0]

def train(generator, discriminator, train_generator, val_generator, plot=True, gen_rate=0):
    # Construct optimizer
    g_opt = optim.Adam(generator.parameters(), lr=G_LR)
    d_opt = optim.Adam(discriminator.parameters(), lr=D_LR)
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

        mle_losses = []
        all_rewards = []
        all_accs = []

        for data in train_gen:
            epoch += 1
            cl_counter += 1
            real_seqs, styles = data

            # Gradually increase the number of steps
            if running_entropy is not None and running_entropy > CL_THRESHOLD and cl_counter > MIN_EPOCH_CL:
                # Only increase timestep if the discriminator
                # is uncertain about the generator's output (convergence)
                target_steps = min(target_steps + 1, SEQ_LEN)
                cl_counter = 0
                
            num_steps = random.randint(MIN_SEQ_LEN, target_steps)
            
            if epoch % 50 == 0:
                mle_train_loss = compute_mle_loss(generator, data, g_opt, validate=False)

            # Perform a rollout #
            fake_seqs, values, log_probs = g_rollout(generator, num_steps)

            # Train the discriminator (and compute rewards)
            real_seqs = real_seqs[:, :num_steps]
            # We don't train the generator if it is too good. Let G catch up.
            optimize = running_acc is None or running_acc < D_OPT_MAX_ACC
            accuracy, reward, entropy = d_train(discriminator, d_opt, fake_seqs, real_seqs, optimize)
            running_entropy = accumulate_running(running_entropy, entropy)
            running_acc = accumulate_running(running_acc, accuracy)

            if running_acc is None or running_acc > G_OPT_MIN_ACC:
                # Train the generator (if the discriminator is decent)
                avg_reward = g_train(generator, g_opt, values, log_probs, reward, num_steps)
                running_reward = accumulate_running(running_reward, avg_reward)

            tq.set_postfix(len=target_steps, reward=running_reward, d_acc=running_acc, val_loss=mle_loss, entropy=running_entropy, train_loss=mle_train_loss)
            tq.update(1)

            if epoch % 200 == 0:
                # Statistic
                mle_loss = sum(compute_mle_loss(generator, data, g_opt, validate=True) for data in  itertools.islice(val_generator(), VAL_STEPS)) / VAL_STEPS
                mle_losses.append(mle_loss)
                all_rewards.append(avg_reward)
                all_accs.append(accuracy)

                if plot:
                    plot_loss(all_rewards, 'reward')
                    plot_loss(all_accs, 'accuracy')
                    plot_loss(mle_losses, 'mle_loss')

            if epoch % 2000 == 0:
                # Save model
                torch.save(generator.state_dict(), OUT_DIR + '/generator_' + str(epoch) + '.pt')
                torch.save(discriminator.state_dict(), OUT_DIR + '/discriminator_' + str(epoch) + '.pt')
                # Generate an output sequence
                Generation(generator).export(name='epoch_' + str(epoch))

def plot_loss(validation_loss, name):
    # Draw graph
    plt.clf()
    plt.plot(validation_loss)
    plt.savefig(OUT_DIR + '/' + name)

def compute_mle_loss(generator, data, optimizer, validate=False):
    """
    Computes the MLE loss of the generator
    """
    optimizer.zero_grad()
    # generator.eval()
    criterion = nn.CrossEntropyLoss()
    # Convert all tensors into variables
    note_seq, styles = data
    # styles = var(one_hot_batch(styles, NUM_STYLES), volatile=validate)
    styles = None # TODO

    # Feed it to the model
    inputs = var(one_hot_seq(note_seq[:, :-1], NUM_ACTIONS), volatile=validate)
    targets = var(note_seq[:, 1:], volatile=validate)
    _, output, _ = generator(inputs, styles, None)

    # Compute the loss.
    loss = criterion(output.contiguous().view(-1, NUM_ACTIONS), targets.view(-1))

    if not validate:
        loss.backward()
        optimizer.step()

    return loss.data[0]

def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--path', help='Load existing model?')
    parser.add_argument('--gen', default=0, type=int, help='Generate per how many epochs?')
    parser.add_argument('--noplot', default=False, action='store_true', help='Do not plot training/loss graphs')
    args = parser.parse_args()

    print('=== Loading Model ===')
    print('GPU: {}'.format(torch.cuda.is_available()))
    common = DeepJCommon()
    generator = DeepJG(common)
    discriminator = DeepJD(common)

    if torch.cuda.is_available():
        # TODO
        torch.backends.cudnn.enabled = False
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
