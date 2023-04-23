import sounddevice as sd
import numpy as np
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
from yeelight import Bulb, LightType, transitions
import time
import random
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import threading
import queue

# Set your Spotify app credentials
sp_client_id = "abf64ee82abd462c90d42b5ea8ae8b8e"
sp_client_secret = "4cb9acb47cdb437ba91384b819122e3b"
sp_redirect_uri = "http://localhost:8888/callback"

# Initialize the Spotipy client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=sp_client_id, client_secret=sp_client_secret, redirect_uri=sp_redirect_uri, scope="user-read-playback-state"))

load_dotenv()

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
btp = BeatTrackingProcessor(fps=100)

# Buffer to store audio data for beat detection
audio_buffer = np.empty((0, 1), dtype=np.float32)

# Audio configuration
RATE = 44100
BUFFER_SIZE = 1024
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
def change_color(beat_queue):
    while True:
        try:
            reference_bpm, detected_bpm = beat_queue.get(timeout=1)
            beat_interval = 60 / reference_bpm
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            bulb.set_rgb(r, g, b)
            print(f"Color changed to ({r}, {g}, {b})")
            time.sleep(beat_interval)
        except queue.Empty:
            continue

# Create a beat_sync_event and start the change_color thread outside the callback
beat_queue = queue.Queue()
color_change_thread = threading.Thread(target=change_color, args=(beat_queue,))
color_change_thread.start()


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

        if reference_bpm:
            print(f"Reference BPM: {reference_bpm:.2f}, Detected BPM: {smoothed_bpm:.2f}")
        else:
            print(f"Detected BPM: {smoothed_bpm:.2f}")

        if len(beats) > 0 and reference_bpm:
            beat_queue.put((reference_bpm, smoothed_bpm))


# Start the stream with the audio output device
with sd.InputStream(device=device_index, channels=CHANNELS, samplerate=RATE, blocksize=BUFFER_SIZE, dtype=FORMAT, callback=audio_callback):
    print("Press Ctrl+C to stop the script.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

# Clean up
bulb.stop_music(mode=1)

    