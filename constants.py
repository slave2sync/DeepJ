### MIDI Parameters ###
MIDI_VELOCITY = 128
# Number of possible notes
NUM_NOTES = 128
# Number of time shift quantizations
TIME_QUANTIZATION = 100
# Exponential representation of time shifts
TICK_EXP = 1
TICK_MUL = 1
# The number of ticks represented in each bin
TICK_BINS = [int(TICK_EXP ** x + TICK_MUL * x) for x in range(TIME_QUANTIZATION)]
# Ticks per second
TICKS_PER_SEC = 100
# Number of velocity buns
VEL_QUANTIZATION = 32

NOTE_ON_OFFSET = 0
NOTE_OFF_OFFSET = NOTE_ON_OFFSET + NUM_NOTES
TIME_OFFSET = NOTE_OFF_OFFSET + NUM_NOTES
VEL_OFFSET = TIME_OFFSET + TIME_QUANTIZATION
NUM_ACTIONS = VEL_OFFSET + VEL_QUANTIZATION

# Trainin Parameters
BATCH_SIZE = 64
MIN_SEQ_LEN = 1
SEQ_LEN = 512 + 1
# Maximum silence time in seconds
SILENT_LENGTH = 3
GRADIENT_CLIP = 3
# Learning rates.
LR = 1e-3
# Minimum number of epochs before leveling up cirriculum
MIN_EPOCH_CL = 500
CL_THRESHOLD = 0.45
D_OPT_MAX_ACC = 0.9
G_OPT_MIN_ACC = 0.5
RUNNING_RATIO = 0.99
# Discount factor
DISCOUNT = 0.99
# GAE parameter
TAU = 1.00
# Number of batches to validate against
VAL_STEPS = 3


# Style
STYLES = ['data/baroque', 'data/classical', 'data/romantic', 'data/modern']
NUM_STYLES = len(STYLES)

# Paths
OUT_DIR = 'out'
CACHE_DIR = 'out/cache'
SAMPLES_DIR = 'out/samples'

settings = {
    'force_cpu': False
}