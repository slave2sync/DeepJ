### MIDI Parameters ###
MIDI_VELOCITY = 128
# Number of possible notes
NUM_NOTES = 128
# Number of time shift quantizations
TIME_QUANTIZATION = 32
# Exponential representation of time shifts
TICK_EXP = 1.14
TICK_MUL = 1
# The number of ticks represented in each bin
TICK_BINS = [int(TICK_EXP ** x + TICK_MUL * x) for x in range(TIME_QUANTIZATION)]
# Ticks per second
TICKS_PER_SEC = 100
# Number of velocity buns
VEL_QUANTIZATION = 32

NOTE_ON_OFFSET = 0
TIME_OFFSET = NOTE_ON_OFFSET + NUM_NOTES
VEL_OFFSET = TIME_OFFSET + TIME_QUANTIZATION
NUM_ACTIONS = VEL_OFFSET + VEL_QUANTIZATION

# Trainin Parameters
BATCH_SIZE = 64
SEQ_LEN = 1024
GRADIENT_CLIP = 10
# The number of train generator cycles per sequence
TRAIN_CYCLES = 1000
VAL_CYCLES = int(TRAIN_CYCLES * 0.05)
LEARNING_RATE = 1e-3

# Style
GENRE = ['classical', 'jazz', 'ragtime', 'rock']
STYLES = [
    [
        'data/classical/bach',
        'data/classical/beethoven',
        'data/classical/chopin',
        'data/classical/debussy',
        'data/classical/mozart'
    ],
    [
        'data/jazz/ellington',
        'data/jazz/evans'
    ],
    [
        'data/ragtime/joplin',
        'data/ragtime/lamb',
        'data/ragtime/scott'
    ],
    [
        'data/rock/joel',
        'data/rock/john'
    ]
]
NUM_STYLES = sum(len(s) for s in STYLES)

# Paths
OUT_DIR = 'out'
CACHE_DIR = 'out/cache'
SAMPLES_DIR = 'out/samples'

settings = {
    'force_cpu': False
}
