SAMPLE_RATE      = 16_000   # Hz  — Silero VAD requires exactly 16 kHz
CHANNELS         = 1        # Mono only (health features are mono)
CHUNK_DURATION   = 0.032    # seconds per raw chunk (~512 samples @ 16 kHz)
# Silero works on 512-sample windows at 16 kHz → 32 ms exactly
CHUNK_SAMPLES    = int(SAMPLE_RATE * CHUNK_DURATION)   # 512


VAD_THRESHOLD         = 0.5    # Silero speech probability cutoff [0, 1]
VAD_MIN_SPEECH_MS     = 100    # discard speech bursts shorter than this
VAD_MIN_SILENCE_MS    = 300    # silence gap that ends a speech segment
VAD_PADDING_MS        = 100    # extra ms added before/after each segment


SPEAKER_ENROLL_SECONDS   = 10    # record this many seconds during enrollment
SPEAKER_SIMILARITY_THRESH = 0.75 # cosine similarity cutoff (0=no match, 1=identical)
SPEAKER_EMBED_DIM        = 256   # Resemblyzer d-vector dimension


N_MFCC           = 40     # number of MFCC coefficients
N_MELS           = 128    # mel filterbank bins for spectral features
HOP_LENGTH       = 160    # STFT hop in samples (10 ms @ 16 kHz)
WIN_LENGTH       = 400    # STFT window in samples (25 ms @ 16 kHz)
EMBEDDING_DIM    = 768    # wav2vec2-base output dimension (if used)


# Set USE_DEEP_EMBEDDINGS = True  → wav2vec2 contextual embeddings (GPU recommended)
# Set USE_DEEP_EMBEDDINGS = False → classical acoustic feature vector (CPU friendly)
USE_DEEP_EMBEDDINGS = False
WAV2VEC_MODEL       = "facebook/wav2vec2-base"   # HuggingFace model id

SPEAKER_PROFILE_PATH = "speaker_profile.npy"   # saved enrollment embedding
