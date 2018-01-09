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
BATCH_SIZE = 32
SEQ_LEN = 1024
GRADIENT_CLIP = 10
# The number of train generator cycles per sequence
TRAIN_CYCLES = 1000
VAL_CYCLES = int(TRAIN_CYCLES * 0.05)
LEARNING_RATE = 1e-3

# Style

# DATA V4
# GENRE = ['baroque', 'classical', 'modern', 'romantic']
# STYLES = [
#     [
#         'data/baroque/bach',
#         'data/baroque/handel',
#         'data/baroque/pachelbel',
#         'data/baroque/rameau',
#         'data/baroque/scarlatti',
#         'data/baroque/soler'
#     ],
#     [
#         'data/classical/balakirev',
#         'data/classical/beethoven',
#         'data/classical/busoni',
#         'data/classical/clementi',
#         'data/classical/franck',
#         'data/classical/granados',
#         'data/classical/haydn',
#         'data/classical/mozart',
#         'data/classical/paganini',
#         'data/classical/scarlatti',
#         'data/classical/verdi'
#     ],
#     [
#         'data/modern/babajanian',
#         'data/modern/barber',
#         'data/modern/bartok',
#         'data/modern/berg',
#         'data/modern/berio',
#         'data/modern/carter',
#         'data/modern/corigliano',
#         'data/modern/coulthard',
#         'data/modern/danielpour',
#         'data/modern/dutilleux',
#         'data/modern/gao',
#         'data/modern/gershwin',
#         'data/modern/ginastera',
#         'data/modern/gubaidulina',
#         'data/modern/hindemith',
#         'data/modern/jalbert',
#         'data/modern/janacek',
#         'data/modern/kapustin',
#         'data/modern/kurtag',
#         'data/modern/kuzmenko',
#         'data/modern/liebermann',
#         'data/modern/ligeti',
#         'data/modern/lutoslawski',
#         'data/modern/martin',
#         'data/modern/mcintyre',
#         'data/modern/messiaen',
#         'data/modern/morel',
#         'data/modern/mozetich',
#         'data/modern/muczynski',
#         'data/modern/nancarrow',
#         'data/modern/poulenc',
#         'data/modern/prokofiev',
#         'data/modern/rodrigo',
#         'data/modern/rzewski',
#         'data/modern/schnittke',
#         'data/modern/scriabin',
#         'data/modern/sheng',
#         'data/modern/slonimsky',
#         'data/modern/stravinsky',
#         'data/modern/szymanowski',
#         'data/modern/takemitsu',
#         'data/modern/vine',
#         'data/modern/zaimont'
#     ],
#     [
#         'data/romantic/albeniz',
#         'data/romantic/balakirev',
#         'data/romantic/bowen',
#         'data/romantic/brahms',
#         'data/romantic/chopin',
#         'data/romantic/debussy',
#         'data/romantic/defalla',
#         'data/romantic/enescu',
#         'data/romantic/franck',
#         'data/romantic/gershwin',
#         'data/romantic/godowsky',
#         'data/romantic/gounod',
#         'data/romantic/korsakov',
#         'data/romantic/kreisler',
#         'data/romantic/liszt',
#         'data/romantic/medtner',
#         'data/romantic/mendelssohn',
#         'data/romantic/mussorgsky',
#         'data/romantic/nancarrow',
#         'data/romantic/prokofiev',
#         'data/romantic/rachmaninoff',
#         'data/romantic/ravel',
#         'data/romantic/saintsaens',
#         'data/romantic/schonberg',
#         'data/romantic/schubert',
#         'data/romantic/schumann',
#         'data/romantic/scriabin',
#         'data/romantic/shostakovich',
#         'data/romantic/taneyev',
#         'data/romantic/tchaikovsky',
#         'data/romantic/verdi',
#         'data/romantic/wagner',
#         'data/romantic/weber'
#     ]
# ]

# DATA V5
# GENRE = ['classical', 'jazz', 'ragtime', 'rock']
# """
# baroque => 1, 11, 20
# classical => 2, 5, 8, 10, 12, 15, 16
# modern => 3, 4, 17, 24
# romantic => 0, 6, 7, 9, 13, 14, 18, 19, 21, 22, 23, 25
# """
# STYLES = [
#     [
#         'data/classical/albeniz',
#         'data/classical/bach',
#         'data/classical/balakirev',
#         'data/classical/barber',
#         'data/classical/bartok',
#         'data/classical/beethoven',
#         'data/classical/brahms',
#         'data/classical/chopin',
#         'data/classical/clementi',
#         'data/classical/debussy',
#         'data/classical/granados',
#         'data/classical/handel',
#         'data/classical/haydn',
#         'data/classical/liszt',
#         'data/classical/mendelssohn',
#         'data/classical/mozart',
#         'data/classical/paganini',
#         'data/classical/prokofiev',
#         'data/classical/rachmaninoff',
#         'data/classical/ravel',
#         'data/classical/scarlatti',
#         'data/classical/schubert',
#         'data/classical/scriabin',
#         'data/classical/shostakovich',
#         'data/classical/stravinsky',
#         'data/classical/tchaikovsky'
#     ],
#     [
#         'data/jazz/ellington',
#         'data/jazz/evans',
#         'data/jazz/mckenzie',
#         'data/jazz/tatum'
#     ],
#     [
#         'data/ragtime/blake',
#         'data/ragtime/johnson',
#         'data/ragtime/joplin',
#         'data/ragtime/lamb',
#         'data/ragtime/morton',
#         'data/ragtime/paull',
#         'data/ragtime/scott',
#         'data/ragtime/turpin',
#         'data/ragtime/wenrich'
#     ],
#     [
#         'data/rock/joel',
#         'data/rock/john'
#     ]
# ]
# NUM_STYLES = sum(len(s) for s in STYLES)
# STYLES = ['data/classical', 'data/jazz', 'data/rock']
# NUM_STYLES = len(STYLES)

# DATA V6
GENRE = ['baroque', 'classical', 'jazz', 'ragtime', 'romantic']
STYLES = [
    [
        'data/baroque/bach',
        'data/baroque/handel',
        'data/baroque/scarlatti'
    ],
    [
        'data/classical/balakirev',
        'data/classical/beethoven',
        'data/classical/clementi',
        'data/classical/granados',
        'data/classical/haydn',
        'data/classical/mozart',
        'data/classical/paganini'
    ],
    [
        'data/jazz/ellington',
        'data/jazz/evans',
        'data/jazz/mckenzie',
        'data/jazz/tatum'
    ],
    [
        'data/ragtime/blake',
        'data/ragtime/johnson',
        'data/ragtime/joplin',
        'data/ragtime/lamb',
        'data/ragtime/morton',
        'data/ragtime/paull',
        'data/ragtime/scott',
        'data/ragtime/turpin',
        'data/ragtime/wenrich'
    ],
    [
        'data/romantic/albeniz',
        'data/romantic/brahms',
        'data/romantic/chopin',
        'data/romantic/debussy',
        'data/romantic/liszt',
        'data/romantic/mendelssohn',
        'data/romantic/rachmaninoff',
        'data/romantic/ravel',
        'data/romantic/schubert',
        'data/romantic/scriabin',
        'data/romantic/shostakovich',
        'data/romantic/tchaikovsky'
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
