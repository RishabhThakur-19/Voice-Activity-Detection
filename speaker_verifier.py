from __future__ import annotations

import numpy as np
import os

from resemblyzer import VoiceEncoder, preprocess_wav

from config import (
    SAMPLE_RATE,
    SPEAKER_SIMILARITY_THRESH,
    SPEAKER_EMBED_DIM,
    SPEAKER_PROFILE_PATH,
    SPEAKER_ENROLL_SECONDS,
)

class SpeakerVerifier: 
    def __init__(self):
        print("[Speaker] Loading Resemblyzer encoder …")
        self._encoder=VoiceEncoder()     # downloads weights on first use
        self._profile: np.ndarray | None = None    # (256,) unit vector
        print("[Speaker] Ready.")

    def enroll(self, audio: np.ndarray) -> np.ndarray:
        min_samples=SPEAKER_ENROLL_SECONDS * SAMPLE_RATE
        if len(audio) < min_samples:
            print(
                f"[Speaker] Warning: enrollment audio is only "
                f"{len(audio)/SAMPLE_RATE:.1f}s, recommended ≥ {SPEAKER_ENROLL_SECONDS}s. "
                f"Profile may be less accurate."
            )

        preprocessed=preprocess_wav(audio, source_sr=SAMPLE_RATE)
        self._profile=self._encoder.embed_utterance(preprocessed)
        print(f"[Speaker] Enrolled. Profile shape: {self._profile.shape}")
        return self._profile

    def enroll_from_segments(self, segments: list[np.ndarray]) -> np.ndarray:
        embeddings=[]
        for seg in segments:
            wav=preprocess_wav(seg, source_sr=SAMPLE_RATE)
            emb=self._encoder.embed_utterance(wav)
            embeddings.append(emb)

        stacked = np.stack(embeddings, axis=0)    # (N, 256)
        mean_emb = stacked.mean(axis=0)
        # Re-normalize to unit sphere (cosine similarity requires unit vectors)
        self._profile = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        print(f"[Speaker] Enrolled from {len(segments)} segments.")
        return self._profile

    def save_profile(self, path: str = SPEAKER_PROFILE_PATH) -> None:
        """Persist the profile to disk as a .npy file."""
        if self._profile is None:
            raise RuntimeError("No profile to save. Run enroll() first.")
        np.save(path, self._profile)
        print(f"[Speaker] Profile saved → {path}")

    def load_profile(self, path: str = SPEAKER_PROFILE_PATH) -> None:
        """Load a previously saved profile from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Speaker profile not found at '{path}'. "
                f"Run enrollment first and call save_profile()."
            )
        self._profile = np.load(path)
        assert self._profile.shape==(SPEAKER_EMBED_DIM,), (
            f"Corrupted profile: expected shape ({SPEAKER_EMBED_DIM},), "
            f"got {self._profile.shape}"
        )
        print(f"[Speaker] Profile loaded from {path}")

    #Verification 

    def is_target_speaker(self, audio: np.ndarray) -> tuple[bool, float]:
        if self._profile is None:
            raise RuntimeError("No speaker profile. Enroll or load_profile() first.")

        wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
        emb = self._encoder.embed_utterance(wav)          # (256,) unit vector
        similarity = float(np.dot(self._profile, emb))    # cosine similarity

        is_user=similarity >= SPEAKER_SIMILARITY_THRESH
        return is_user, similarity

    def embed(self, audio: np.ndarray) -> np.ndarray:
        wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
        return self._encoder.embed_utterance(wav)

    @property
    def has_profile(self) -> bool:
        return self._profile is not None
