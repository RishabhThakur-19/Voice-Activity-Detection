import numpy as np
import soundfile as sf
import torch
import time

from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from audio_stream import AudioStream
from vad_processor import VADProcessor


# ── CONFIG ─────────────────────────────────────────────

SAMPLE_RATE = 16000
BUFFER_SEC = 1.2
STEP_SEC = 0.6
THRESHOLD = 0.75   # tune this later


# ── PREPROCESS ─────────────────────────────────────────

def preprocess_audio(audio):
    audio = audio.astype(np.float32)

    if len(audio.shape) > 1:
        audio = audio[:, 0]

    audio = audio - np.mean(audio)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    return audio


# ── MAIN SYSTEM ────────────────────────────────────────

class RealTimeSpeakerVerifier:
    def __init__(self, centroid_file="speaker_centroid.npy"):
        print("🔄 Loading model...")

        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models",
            local_strategy=LocalStrategy.COPY
        )

        print("✅ Model ready\n")

        self.centroid = np.load(centroid_file)

        self.vad = VADProcessor()

        self.buffer_audio = []
        self.buffer_duration = 0.0

        self.matched_segments = []
        self.scores = []

    def get_embedding(self, audio):
        audio = preprocess_audio(audio)

        tensor = torch.tensor(audio).unsqueeze(0)

        emb = self.model.encode_batch(tensor).squeeze().numpy()
        return emb

    def compare(self, audio):
        emb = self.get_embedding(audio)
        score = 1 - cosine(emb, self.centroid)
        return score

    def run(self):
        print("🎤 Press ENTER to start recording")
        input()

        print("🔴 Recording... Press ENTER to stop")

        stop_flag = False

        import threading
        def stop_listener():
            input()
            nonlocal stop_flag
            stop_flag = True

        threading.Thread(target=stop_listener, daemon=True).start()

        start_time = time.time()

        with AudioStream() as stream:
            while not stop_flag:
                chunk = stream.read()

                segment = self.vad.process_chunk(chunk)

                if segment is not None:
                    self.buffer_audio.append(segment.audio)
                    self.buffer_duration += segment.duration_s

                if self.buffer_duration >= BUFFER_SEC:
                    combined = np.concatenate(self.buffer_audio)

                    score = self.compare(combined)
                    self.scores.append(score)

                    current_time = time.time() - start_time

                    print(f"⏱ {current_time:.2f}s | 🔍 Score: {score:.3f}")

                    if score > THRESHOLD:
                        print("✅ TARGET SPEAKER")
                        self.matched_segments.append(combined)
                    else:
                        print("❌ OTHER")

                    # sliding window
                    keep_samples = int(STEP_SEC * SAMPLE_RATE)
                    combined = combined[-keep_samples:]

                    self.buffer_audio = [combined]
                    self.buffer_duration = len(combined) / SAMPLE_RATE

        print("\n⏹ Stopping...")

        self.save_output()
        self.plot_scores()

    def save_output(self):
        if len(self.matched_segments) == 0:
            print("⚠️ No matched segments")
            return

        final_audio = np.concatenate(self.matched_segments)

        sf.write("target_speaker_live.wav", final_audio, SAMPLE_RATE)

        print("\n🎧 Saved → target_speaker_live.wav")
        print(f"Duration: {len(final_audio)/SAMPLE_RATE:.2f} sec")

    def plot_scores(self):
        try:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(self.scores)
            plt.axhline(y=THRESHOLD)
            plt.title("Confidence Scores Over Time")
            plt.xlabel("Segments")
            plt.ylabel("Score")
            plt.show()

        except:
            print("⚠️ Matplotlib not installed (skip graph)")


# ── RUN ───────────────────────────────────────────────

if __name__ == "__main__":
    verifier = RealTimeSpeakerVerifier()
    verifier.run()