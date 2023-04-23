import sounddevice as sd
import numpy as np
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from yeelight import Bulb, LightType, transitions
import time
import random
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

# Set your Spotify app credentials
sp_client_id = os.getenv('CLIENT_ID')
sp_client_secret = os.getenv('CLIENT_SECRET')
sp_redirect_uri = os.getenv('REDIRECT_URI')

# Initialize the Spotipy client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=sp_client_id, client_secret=sp_client_secret, redirect_uri=sp_redirect_uri, scope="user-read-playback-state"))

# Yeelight configuration
bulb_ip = os.getenv('IP')
bulb_auto_on = True
bulb = Bulb(bulb_ip, auto_on=bulb_auto_on)
bulb.turn_on()
bulb.set_rgb(255, 255, 255)
time.sleep(1)
bulb.start_music()

# Find the index of the audio output device you want to capture
device_index = None
devices = sd.query_devices()
for index, device in enumerate(devices):
    if device['name'] == 'VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)':
        device_index = index
        break

# Check if device_index is not None, and continue
if device_index is None:
    print("Audio output device not found.")
    exit()

# Madmom beat detection
bp = RNNBeatProcessor()
beat_processor = DBNBeatTrackingProcessor(fps=100)

current_beats = []


# Buffer to store audio data for beat detection
audio_buffer = np.empty((0, 1), dtype=np.float32)

# Audio configuration
RATE = 44100
BUFFER_SIZE = 2048
CHANNELS = 2
FORMAT = 'float32'
BPM_WINDOW_SIZE = 5
bpm_values = []

def get_current_song_bpm():
    try:
        current_playback = sp.current_playback()
        if current_playback and current_playback['item']:
            track_id = current_playback['item']['id']
            audio_features = sp.audio_features([track_id])[0]
            return audio_features['tempo']
    except Exception as e:
        print(f"Error getting current song BPM: {e}")

    return None

def beat_detection(audio_data):
    global audio_buffer
    audio_buffer = np.append(audio_buffer, audio_data)

    if audio_buffer.shape[0] >= RATE:
        act = bp(audio_buffer[:RATE])
        beats = beat_processor(act)
        audio_buffer = audio_buffer[BUFFER_SIZE:]

        if len(beats) > 1:
            bpm = 60 / np.mean(np.diff(beats))
            return beats, bpm
        else:
            return [], 0
    else:
        return [], 0

# Change Yeelight color based on the beat
def change_color(beats, reference_bpm, detected_bpm):
    global current_beats
    if len(beats) > 0:
        current_beats = beats
        scale_factor = reference_bpm / detected_bpm
        for i in range(len(beats) - 1):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            bulb.set_rgb(r, g, b)
            print(f"Color changed to ({r}, {g}, {b})")
            time_until_next_beat = (beats[i + 1] - beats[i]) * scale_factor
            print(f"Time difference between beats: {time_until_next_beat:.3f} seconds")
            time.sleep(time_until_next_beat)

# Callback function to process audio data
def audio_callback(indata, frames, time, status):
    audio_data = indata[:, 0]
    beats, detected_bpm = beat_detection(audio_data)

    if detected_bpm > 0:
        bpm_values.append(detected_bpm)
        if len(bpm_values) > BPM_WINDOW_SIZE:
            bpm_values.pop(0)

        smoothed_bpm = np.mean(bpm_values)

        # Get the current song BPM from Spotify
        reference_bpm = get_current_song_bpm()

        if reference_bpm > 0:
            print(f"Reference BPM: {reference_bpm:.2f}, Detected BPM: {smoothed_bpm:.2f}")
        else:
            print(f"Detected BPM: {smoothed_bpm:.2f}")

        if len(beats) > 0 and reference_bpm:
            change_color(beats, reference_bpm, smoothed_bpm)

# Start the stream with the audio output device
with sd.InputStream(device=device_index, channels=CHANNELS, samplerate=RATE, blocksize=BUFFER_SIZE, dtype=FORMAT, callback=audio_callback):
    print("Press Ctrl+C to stop the script.")
    try:
        while True:
            time.sleep(1)
            if len(current_beats) > 0:
                reference_bpm = get_current_song_bpm()
                detected_bpm = np.mean(bpm_values)
                change_color(current_beats, reference_bpm, detected_bpm)
    except KeyboardInterrupt:
        pass

# Clean up
bulb.stop_music(mode=1)

    