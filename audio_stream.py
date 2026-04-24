import queue
import numpy as np
import sounddevice as sd

from config import SAMPLE_RATE, CHANNELS, CHUNK_SAMPLES


class AudioStream:
    def __init__(self, device: int | None = None):
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,   # exactly 512 samples per callback
            device=device,
            callback=self._callback,
        )
    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            # e.g. input overflow — log but don't crash
            print(f"[AudioStream] status: {status}")
        # indata shape is (CHUNK_SAMPLES, CHANNELS); flatten to 1-D
        self._queue.put(indata[:, 0].copy())

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def read(self, timeout: float = 5.0) -> np.ndarray:
        return self._queue.get(timeout=timeout)

    def drain(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
