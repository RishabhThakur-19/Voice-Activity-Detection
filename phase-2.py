import numpy as np
import soundfile as sf
import torch

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


# ── CONFIG ─────────────────────────────────────────────

INPUT_WAV = "segments_train.wav"   # your long recording
OUTPUT_FILE = "speaker_centroid.npy"
NUM_CHUNKS = 20                   # split audio into parts


# ── LOAD MODEL ─────────────────────────────────────────

print("🔄 Loading SpeechBrain model...")

model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models",
    local_strategy=LocalStrategy.COPY   # 🔥 FIX for Windows (no symlink)
)

print("✅ Model loaded\n")


# ── LOAD AUDIO ─────────────────────────────────────────

audio, sr = sf.read(INPUT_WAV)

if sr != 16000:
    raise ValueError("❌ Audio must be 16kHz")

if len(audio.shape) > 1:
    audio = audio[:, 0]   # convert to mono


# ── SPLIT INTO CHUNKS ──────────────────────────────────

chunks = np.array_split(audio, NUM_CHUNKS)

embeddings = []

print("🎧 Generating embeddings...\n")

for i, chunk in enumerate(chunks):
    chunk = chunk.astype(np.float32)

    # normalize
    chunk = chunk - np.mean(chunk)
    chunk = chunk / (np.max(np.abs(chunk)) + 1e-6)

    tensor = torch.tensor(chunk).unsqueeze(0)

    emb = model.encode_batch(tensor).squeeze().numpy()

    embeddings.append(emb)

    print(f"✔ Processed chunk {i+1}/{NUM_CHUNKS}")


# ── CREATE CENTROID ────────────────────────────────────

centroid = np.mean(embeddings, axis=0)

np.save(OUTPUT_FILE, centroid)

print("\n✅ Speaker centroid saved → speaker_centroid.npy")
print(f"Embedding shape: {centroid.shape}")