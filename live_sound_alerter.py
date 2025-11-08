import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import time
import threading
from twilio.rest import Client
from queue import Queue

# --- 1. CONFIGURATION ---

# --- CRITICAL: EDIT THIS MAP ---
# Use the *real names* you found with the `find_my_class_names.py` script.
ALERT_SOUNDS_MAP = {
    # "real_class_name_from_finder": "Your custom SMS message",
    "Baby Crying Sounds": "Alert: The baby is crying!", # UPDATED from "16000" to match your log
    "Dog Barking": "Alert: The dog is barking!", # UPDATED from "dog_bark" to match your log
    "Siren": "Alert: A siren is detected nearby.",
    "glass_break": "Alert: A glass window just broke!",
    "fire_alarm": "Alert: FIRE ALARM DETECTED!"
}

# --- Twilio Configuration ---
TWILIO_ACCOUNT_SID = 'AC452ca9b882dcec93371999a28fd0a00c'
TWILIO_AUTH_TOKEN = '1b46062cdd194551e19003eb5921f3f9'
TWILIO_FROM_NUMBER = '+13392290261'
YOUR_TO_NUMBER = '+917720907937'

# --- Model & Audio Configuration ---
MODEL_PATH = 'environmental_sound_classifier.h5'
CLASSES_PATH = 'classes.npy'
TARGET_SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
IMG_HEIGHT = 224
IMG_WIDTH = 224
RECORD_DURATION_SEC = 5.0
CONFIDENCE_THRESHOLD = 60.0 # Alert if confidence is over this %
COOLDOWN_PERIOD_SEC = 10.0  # Wait this long before sending another SMS

# --- 2. GLOBAL VARIABLES ---
MODEL = None
CLASS_LABELS = None
TWILIO_CLIENT = None
LAST_ALERT_TIME = 0
AUDIO_BUFFER = np.array([], dtype=np.float32)
SAMPLES_PER_CHUNK = int(RECORD_DURATION_SEC * TARGET_SAMPLE_RATE)
audio_queue = Queue()

def load_model_and_labels():
    global MODEL, CLASS_LABELS
    try:
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        CLASS_LABELS = np.load(CLASSES_PATH, allow_pickle=True)
        print(f"Model and {len(CLASS_LABELS)} labels loaded.")
        print(f"Model expects input: {MODEL.input_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

def initialize_twilio():
    global TWILIO_CLIENT
    try:
        TWILIO_CLIENT = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        print(f"Twilio client initialized. Ready to send SMS to: {YOUR_TO_NUMBER}")
    except Exception as e:
        print(f"Error initializing Twilio: {e}")
        exit()

def preprocess_audio_stream(signal_chunk):
    try:
        mel_spec = librosa.feature.melspectrogram(y=signal_chunk,
                                                  sr=TARGET_SAMPLE_RATE,
                                                  n_mels=IMG_HEIGHT,
                                                  n_fft=N_FFT,
                                                  hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if np.max(mel_spec_db) != np.min(mel_spec_db):
            mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))
        else:
            mel_spec_db = np.zeros_like(mel_spec_db)

        mel_spec_db_batch = mel_spec_db[np.newaxis, ..., np.newaxis]
        resized_spec = tf.image.resize(mel_spec_db_batch, [IMG_HEIGHT, IMG_WIDTH])
        resized_spec_rgb = tf.image.grayscale_to_rgb(resized_spec)
        return resized_spec_rgb
    except Exception as e:
        print(f"Pre-processing error: {e}")
        return None

def predict_sound(processed_audio):
    try:
        prediction = MODEL.predict(processed_audio, verbose=0)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_class = CLASS_LABELS[predicted_index]
        confidence = prediction[0][predicted_index] * 100
        return predicted_class, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0

def send_sms_alert(message_body):
    global LAST_ALERT_TIME
    current_time = time.time()

    if (current_time - LAST_ALERT_TIME) < COOLDOWN_PERIOD_SEC:
        print(f"(In cooldown, alert for '{message_body}' suppressed)")
        return

    print(f"\n>>> SENDING SMS: {message_body} <<<\n")
    try:
        message = TWILIO_CLIENT.messages.create(
            body=message_body,
            from_=TWILIO_FROM_NUMBER,
            to=YOUR_TO_NUMBER
        )
        print(f"SMS sent successfully! SID: {message.sid}")
        LAST_ALERT_TIME = current_time
    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! FATAL ERROR SENDING SMS !!!")
        print(f"!!! Twilio Error: {e}")
        print("!!! 1. Is your 'To' number verified in your Twilio trial account?")
        print("!!! 2. Are you trying to send to India? (Requires DLT registration)")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

def audio_callback(indata, frames, time_info, status):
    global AUDIO_BUFFER
    if status:
        print(f"Audio warning: {status}")
    AUDIO_BUFFER = np.append(AUDIO_BUFFER, indata.flatten())
    while len(AUDIO_BUFFER) >= SAMPLES_PER_CHUNK:
        signal_chunk = AUDIO_BUFFER[:SAMPLES_PER_CHUNK]
        AUDIO_BUFFER = AUDIO_BUFFER[SAMPLES_PER_CHUNK:]
        try:
            audio_queue.put_nowait(signal_chunk)
        except Queue.Full:
            pass

def processing_thread_func():
    while True:
        try:
            signal_chunk = audio_queue.get()
            processed_audio = preprocess_audio_stream(signal_chunk)

            if processed_audio is not None:
                predicted_class, confidence = predict_sound(processed_audio)

                if predicted_class is not None and confidence >= CONFIDENCE_THRESHOLD:
                    print(f"Heard: '{predicted_class}' (Confidence: {confidence:.2f}%)")

                    # Check if this sound is one we need to alert for
                    if predicted_class in ALERT_SOUNDS_MAP:
                        message_to_send = ALERT_SOUNDS_MAP[predicted_class]
                        send_sms_alert(message_to_send)
                    else:
                        # --- THIS IS THE UPDATE ---
                        print(f"--> INFO: Detected '{predicted_class}', but it is not in your ALERT_SOUNDS_MAP. No SMS sent.")
                        print(f"--> Your map only alerts for: {list(ALERT_SOUNDS_MAP.keys())}")
                        # --- END OF UPDATE ---

        except Exception as e:
            print(f"Processing thread error: {e}")

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    load_model_and_labels()
    initialize_twilio()

    proc_thread = threading.Thread(target=processing_thread_func, daemon=True)
    proc_thread.start()

    print("\nStarting live audio monitor...")
    print(f"Listening for: {list(ALERT_SOUNDS_MAP.keys())}")
    print(f"Alerts will be sent if confidence is > {CONFIDENCE_THRESHOLD}%")
    print(f"SMS will be sent to: {YOUR_TO_NUMBER}")
    print("Press Ctrl+C to stop.")

    try:
        with sd.InputStream(callback=audio_callback,
                            samplerate=TARGET_SAMPLE_RATE,
                            channels=1,
                            dtype='float32'):
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor.")