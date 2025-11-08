import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time

# --- 1. CONFIGURATION (MUST MATCH YOUR PREDICTION SCRIPTS) ---
DATASET_PATH = r"D:\1LPUcollegesem\semister 7\int422 Deep Learning\homeassistant project\MODEL" # The folder with your 'Baby crying sound', etc. subfolders
MODEL_OUTPUT_NAME = 'environmental_sound_classifier.h5'
CLASSES_OUTPUT_NAME = 'classes.npy'

# Audio processing settings
TARGET_SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
IMG_HEIGHT = 224
IMG_WIDTH = 224
RECORD_DURATION_SEC = 2.0 # Process audio in 2-second chunks
SAMPLES_PER_CHUNK = int(RECORD_DURATION_SEC * TARGET_SAMPLE_RATE)

# Training settings
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2 # Use 20% of data for validation

# --- 2. PREPROCESSING FUNCTION (Identical to your other scripts) ---

def preprocess_audio_chunk(signal_chunk):
    """
    Converts a raw audio chunk into a 224x224x3 spectrogram image.
    This MUST be identical to the function in your live alerter/visualizer.
    """
    try:
        mel_spec = librosa.feature.melspectrogram(y=signal_chunk,
                                                  sr=TARGET_SAMPLE_RATE,
                                                  n_mels=IMG_HEIGHT,
                                                  n_fft=N_FFT,
                                                  hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        if np.max(mel_spec_db) != np.min(mel_spec_db):
            mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))
        else:
            mel_spec_db = np.zeros_like(mel_spec_db)

        # Resize and convert to 3-channel (RGB)
        # This logic ensures the chunk is processed correctly
        if mel_spec_db.shape[1] < IMG_WIDTH:
             # Pad if too short (common for the last chunk of a file)
             pad_width = IMG_WIDTH - mel_spec_db.shape[1]
             mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        elif mel_spec_db.shape[1] > IMG_WIDTH:
             # Truncate if too long (shouldn't happen with 2s chunks, but safe)
             mel_spec_db = mel_spec_db[:, :IMG_WIDTH]

        mel_spec_db_batch = mel_spec_db[np.newaxis, ..., np.newaxis]
        resized_spec = tf.image.resize(mel_spec_db_batch, [IMG_HEIGHT, IMG_WIDTH])
        resized_spec_rgb = tf.image.grayscale_to_rgb(resized_spec)

        # Squeeze out the batch dimension
        return np.squeeze(resized_spec_rgb, axis=0)

    except Exception as e:
        print(f"Pre-processing error: {e}")
        return None

# --- 3. DATA LOADING FUNCTION ---

def load_dataset(dataset_path):
    """
    Loads all audio files, processes them into chunks, and returns
    spectrogram images (X) and labels (y).
    """
    print(f"Starting dataset load from: {dataset_path}")
    X = [] # To store spectrogram "images"
    y = [] # To store labels (as strings first)

    class_labels = []

    # Find all subdirectories (these are the class names)
    try:
        for class_name in os.listdir(dataset_path):
            class_dir = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_dir):
                print(f"Found class: '{class_name}'")
                class_labels.append(class_name)

                # Go through each audio file in this class folder
                for file_name in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file_name)

                    try:
                        # Load the entire audio file
                        signal, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)

                        # Break the signal into 2-second chunks
                        num_chunks = int(np.ceil(len(signal) / SAMPLES_PER_CHUNK))
                        if num_chunks == 0:
                            continue

                        for i in range(num_chunks):
                            start = i * SAMPLES_PER_CHUNK
                            end = start + SAMPLES_PER_CHUNK
                            signal_chunk = signal[start:end]

                            # Pad the last chunk if it's too short
                            if len(signal_chunk) < SAMPLES_PER_CHUNK:
                                signal_chunk = np.pad(signal_chunk, (0, SAMPLES_PER_CHUNK - len(signal_chunk)), 'constant')

                            # Preprocess the chunk into an image
                            spectrogram_image = preprocess_audio_chunk(signal_chunk)

                            if spectrogram_image is not None:
                                X.append(spectrogram_image)
                                y.append(class_name)

                    except Exception as e:
                        print(f"Warning: Could not load/process file {file_path}: {e}")

    except Exception as e:
        print(f"Fatal Error: Could not read dataset directory {dataset_path}: {e}")
        return None, None, None

    print(f"\nDataset loading complete. Found {len(X)} audio chunks.")

    # Convert labels from strings to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # The `le.classes_` array now holds the mapping from index to string name
    # This is *exactly* what we need to save for `classes.npy`
    print(f"Encoded {len(le.classes_)} classes: {le.classes_}")

    return np.array(X), np.array(y_encoded), le.classes_

# --- 4. MODEL BUILDING FUNCTION ---

def build_model(input_shape, num_classes):
    """
    Builds a simple CNN model.
    """
    print("Building CNN model...")
    model = models.Sequential()

    # Use a pre-built model (MobileNetV2) for better accuracy
    # This is called "Transfer Learning"
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False, # Don't include the final ImageNet classifier layer
        weights='imagenet' # Use weights pre-trained on images
    )
    base_model.trainable = False # Freeze the base model

    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

# --- 5. PLOTTING FUNCTION ---

def plot_history(history):
    """Plots the training and validation accuracy/loss."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.set_title('Model Accuracy')

    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.set_title('Model Loss')

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nSaved training history plot to 'training_history.png'")

# --- 6. MAIN EXECUTION ---

if __name__ == "__main__":
    start_time = time.time()

    # 1. Load data
    X, y, class_labels = load_dataset(DATASET_PATH)

    if X is None or len(X) == 0:
        print("Failed to load dataset. Exiting.")
        exit()

    print(f"\nTotal data shape: {X.shape}")
    print(f"Total labels shape: {y.shape}")

    # 2. Split the data
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=y # Ensure classes are balanced in splits
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # 3. Build the model
    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
    NUM_CLASSES = len(class_labels)
    model = build_model(INPUT_SHAPE, NUM_CLASSES)

    # 4. Train the model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
    )
    print("--- Model Training Complete ---")

    # 5. Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"\nFinal Validation Accuracy: {accuracy*100:.2f}%")
    print(f"Final Validation Loss: {loss:.4f}")

    # 6. Save the results
    print(f"Saving model to {MODEL_OUTPUT_NAME}...")
    model.save(MODEL_OUTPUT_NAME)

    print(f"Saving class labels to {CLASSES_OUTPUT_NAME}...")
    np.save(CLASSES_OUTPUT_NAME, class_labels)

    # 7. Plot results
    plot_history(history)

    end_time = time.time()
    print(f"\nAll done! Total time: {(end_time - start_time) / 60:.2f} minutes.")
    print(f"You can now run your other scripts with the new '{MODEL_OUTPUT_NAME}' and '{CLASSES_OUTPUT_NAME}' files.")