import numpy as np
import argparse
import heapq

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange

from midi_io import *
from dataset import *
from constants import *
from util import *
from model import DeepJ

class Generation():
    """
    Represents a music generation sequence
    """

    def __init__(self, model, style=None, default_temp=1, beam_size=1, adaptive=False):
        self.model = model

        self.beam_size = beam_size

        # Pick a random style
        self.style = style if style is not None else one_hot(np.random.randint(0, NUM_STYLES), NUM_STYLES)

        # Temperature of generation
        self.default_temp = default_temp
        self.temperature = self.default_temp

        # Progress of generated music between 0 and 1
        self.progress_scalar = 0
        self.levels = CATEGORY_LEVELS + 2
        # Culmulative amount of time in generated song
        self.time_passed = 0

        # Model parametrs
        self.beam = [
            (1, tuple(), None, self.progress_scalar, one_hot(0, CATEGORY_LEVELS))
        ]
        self.avg_seq_prob = 1
        self.step_count = 0
        self.adaptive_temp = adaptive

    def step(self, seq_len):
        """
        Generates the next set of beams
        """
        # Create variables
        style = var(to_torch(self.style), volatile=True).unsqueeze(0)

        new_beam = []
        sum_seq_prob = 0

        time_step = seq_len // self.levels

        # Iterate through the beam
        for prev_prob, evts, state, progress_scalar, progress_category in self.beam:
            if len(evts) > 0:
                prev_event = var(to_torch(one_hot(evts[-1], NUM_ACTIONS)), volatile=True).unsqueeze(0)
                if evts[-1] >= TIME_OFFSET and evts[-1] < TIME_OFFSET + TIME_QUANTIZATION:
                    time = TICK_BINS[evts[-1] - TIME_OFFSET] / TICKS_PER_SEC
                    self.time_passed += time
                    progress_scalar += time / seq_len
                    self.progress_scalar = progress_scalar
                    # Update categorical progress one hot vector
                    if self.time_passed > 0 and self.time_passed < time_step:
                        progress_category = one_hot(0, CATEGORY_LEVELS)
                    elif self.time_passed > 2 * time_step and self.time_passed < (2 * time_step) + time_step:
                        progress_category = one_hot(1, CATEGORY_LEVELS)
                    elif self.time_passed > 4 * time_step and self.time_passed < seq_len:
                        progress_category = one_hot(2, CATEGORY_LEVELS)
                    else:
                        # TODO: fix edge case where last progress vector is 0's instead of [0, 0, 1]
                        progress_category = np.zeros(CATEGORY_LEVELS)
            else:
                prev_event = var(torch.zeros((1, NUM_ACTIONS)), volatile=True)
            
            prev_event = prev_event.unsqueeze(1)
            progress_scalar_tensor = var(torch.FloatTensor([progress_scalar])).unsqueeze(1)
            progress_category_tensor = var(torch.FloatTensor([progress_category])).unsqueeze(0)
            probs, new_state = self.model.generate(prev_event, style, progress_scalar_tensor, progress_category_tensor, state, temperature=self.temperature)
            probs = probs.squeeze(1)

            for _ in range(self.beam_size):
                # Sample action
                output = probs.multinomial().data
                event = output[0, 0]
                
                # Create next beam
                seq_prob = prev_prob * probs.data[0, event]
                # Boost the sequence probability by the average
                new_beam.append((seq_prob / self.avg_seq_prob, evts + (event,), new_state, progress_scalar, progress_category))
                sum_seq_prob += seq_prob

        self.avg_seq_prob = sum_seq_prob / len(new_beam)
        # Find the top most probable sequences
        self.beam = heapq.nlargest(self.beam_size, new_beam, key=lambda x: x[0])

        if self.adaptive_temp and self.step_count > 50:
            r = repetitiveness(self.beam[0][1][-50:])
            if r < 0.1:
                self.temperature = self.default_temp
            else:
                self.temperature += 0.05
        
        self.step_count += 1

    def generate(self, seq_len=30):
        self.model.eval()
        while self.progress_scalar <= 1:
            self.step(seq_len)

        best = max(self.beam, key=lambda x: x[0])
        best_seq = best[1]
        return np.array(best_seq)

    def export(self, name='output', seq_len=30):
        """
        Export into a MIDI file.
        """
        seq = self.generate(seq_len)
        save_midi(name, seq)

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--path', help='Path to model file')
    parser.add_argument('--length', default=30, type=int, help='Length of generation in seconds')
    parser.add_argument('--style', default=None, type=int, nargs='+', help='Styles to mix together')
    parser.add_argument('--temperature', default=1, type=float, help='Temperature of generation')
    parser.add_argument('--beam', default=1, type=int, help='Beam size')
    parser.add_argument('--adaptive', default=False, action='store_true', help='Adaptive temperature')
    args = parser.parse_args()

    style = None

    if args.style:
        # Custom style
        style = np.mean([one_hot(i, NUM_STYLES) for i in args.style], axis=0)

    print('=== Loading Model ===')
    print('Path: {}'.format(args.path))
    print('Temperature: {}'.format(args.temperature))
    print('Adaptive Temperature: {}'.format(args.adaptive))
    print('GPU: {}'.format(torch.cuda.is_available()))
    settings['force_cpu'] = True
    
    model = DeepJ()

    if args.path:
        model.load_state_dict(torch.load(args.path))
    else:
        print('WARNING: No model loaded! Please specify model path.')

    print('=== Generating ===')
    Generation(model, style=style, default_temp=args.temperature, beam_size=args.beam, adaptive=args.adaptive).export(seq_len=args.length)

if __name__ == '__main__':
    main()
