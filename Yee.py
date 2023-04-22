import pyaudio
import numpy as np
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
from yeelight import Bulb, LightType, transitions
import time
import random
import os
from dotenv import load_dotenv

load_dotenv()

# Yeelight configuration
bulb_ip = os.getenv('IP')
bulb_auto_on = True
bulb = Bulb(bulb_ip, auto_on=bulb_auto_on)
bulb.turn_on()
bulb.set_rgb(255, 255, 255)
time.sleep(1)
bulb.start_music()

# Audio configuration
RATE = 44100
BUFFER_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paFloat32
BPM_WINDOW_SIZE = 5
bpm_values = []

# Instantiate PyAudio and open the stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=BUFFER_SIZE)

# Madmom beat detection
bp = RNNBeatProcessor()
btp = BeatTrackingProcessor(fps=200)

# Buffer to store audio data for beat detection
audio_buffer = np.empty((0, 1), dtype=np.float32)

def beat_detection(audio_data):
    global audio_buffer
    audio_buffer = np.append(audio_buffer, audio_data)

    if audio_buffer.shape[0] >= RATE:
        act = bp(audio_buffer[:RATE])
        beats = btp(act)
        audio_buffer = audio_buffer[BUFFER_SIZE:]

        if len(beats) > 1:
            bpm = 60 / np.mean(np.diff(beats))
            return beats, bpm
        else:
            return [], 0
    else:
        return [], 0

# Change Yeelight color based on the beat
def change_color(beats):
    for i in range(len(beats) - 1):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        bulb.set_rgb(r, g, b)
        print(f"Color changed to ({r}, {g}, {b})")
        time_until_next_beat = beats[i + 1] - beats[i]
        time.sleep(time_until_next_beat)


stream.start_stream()

# Keep the script running
try:
    while True:
        in_data = stream.read(BUFFER_SIZE)
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        beats, bpm = beat_detection(audio_data)

        if bpm > 0:
            bpm_values.append(bpm)
            if len(bpm_values) > BPM_WINDOW_SIZE:
                bpm_values.pop(0)

            smoothed_bpm = np.mean(bpm_values)
            print(f"Beat detected, BPM: {smoothed_bpm:.2f}")

            if len(beats) > 0:
                change_color(beats)
except KeyboardInterrupt:
    pass


# Clean up
stream.stop_stream()
stream.close()
p.terminate()
bulb.stop_music(mode=1)
