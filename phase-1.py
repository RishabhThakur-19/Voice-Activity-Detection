import sounddevice as sd
import soundfile as sf
import numpy as np
import time

SAMPLE_RATE = 16000
#BEEP SOUND ADDED
def beep():
    try:
        import winsound
        winsound.Beep(1000, 400)
    except:
        print("\a")


SEGMENTS = [
    # Speech variation
    ("Speak normally", 60),
    ("Speak loudly", 60),
    ("Whisper softly", 60),
    ("Speak very fast", 60),
    ("Speak very slow", 60),
    ("Speak with pauses", 60),

    # Emotion
    ("Speak while smiling", 60),
    ("Speak angrily", 60),
    ("Speak excitedly", 60),
    ("Speak tired / low energy", 60),

    # Mic variation
    ("Very close to mic", 60),
    ("Normal distance", 60),
    ("Far from mic", 60),
    ("Turn head while speaking", 60),

    # Noise
    ("Fan noise background", 60),
    ("TV/music background", 60),
    ("Outdoor ambient noise", 60),
    ("Typing while speaking", 60),

    # Non-speech (important)
    ("Cough naturally", 40),
    ("Sneezing / throat clearing", 40),
    ("Breathing sounds", 40),
    ("Lip/mouth sounds", 40),

    # Real-world behavior
    ("Talk while eating", 60),
    ("Talk while walking", 60),
    ("Talk with interruptions", 60),
    ("Free conversation", 60),
]


#  Recorder PART

def record_guided_session(filename="full_session.wav"):

    print("\N Guided Recording Session...")
    print(" Follow instructions. Everything is automatic...\n")

    input("Press ENTER to start...\n")

    recording = []

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        recording.append(indata.copy())

    #  SINGLE STREAM (important fix)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):

        for idx, (text, duration) in enumerate(SEGMENTS):

            print(f"\n Segment {idx+1}/{len(SEGMENTS)}")
            beep()

            print(f" TASK: {text}")
            print(f"Duration: {duration} sec")

            for i in range(duration, 0, -1):
                print(f"   Remaining: {i}s", end="\r")
                time.sleep(1)

    print("\n\n Recording complete. Processing...")

    audio = np.concatenate(recording, axis=0)

    # ensure correct format
    audio = audio.astype(np.float32)

    sf.write(filename, audio, SAMPLE_RATE)

    print(f" Saved → {filename}")
    print(f" Total duration: {len(audio)/SAMPLE_RATE:.2f} sec")


#  Run 

if __name__ == "__main__":
    record_guided_session()