import re

# Zone order must match training.
ZONE_ORDER = [
    "Z1_Header",
    "Z2_Preamble",
    "Z3_Fact",
    "Z4_Reasoning",
    "Z5_Judgment",
    "Z6_Footer",
    "Z7_SignOff",
]

# Sentence split whitelist: punctuation and newlines.
SENTENCE_SPLIT_REGEX = re.compile(r"[。！？!?；;：:、\n]+")

# Anchor constraints (used in inference when --anchor auto).
Z1_ANCHOR_CHAR = "号"
Z1_ANCHOR_MAX_CHARS = 100
Z4_ANCHOR_REGEX = re.compile(r"(判\s*决\s*如\s*下|如\s*下\s*判\s*决)\s*[:：]?")

# Inference-time preprocessing hyperparameters (must match training).
MAX_SENTENCES = 240
MAX_CHARS_PER_SENT = 80

# Model hyperparameters (must match training).
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2

