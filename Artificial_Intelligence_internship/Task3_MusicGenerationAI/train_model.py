import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "midi_dataset"
MODEL_PATH = "music_model.keras"
NOTES_PATH = "notes.pkl"
SEQUENCE_LENGTH = 50


def extract_notes():
    notes = []

    midi_files = [
        file for file in os.listdir(DATASET_PATH)
        if file.endswith(".mid") or file.endswith(".midi")
    ]

    if len(midi_files) == 0:
        print("No MIDI files found in midi_dataset folder.")
        return notes

    for file in midi_files:
        file_path = os.path.join(DATASET_PATH, file)
        print(f"Reading {file_path}")

        try:
            midi = converter.parse(file_path)
            parts = instrument.partitionByInstrument(midi)

            if parts:
                elements = parts.parts[0].recurse()
            else:
                elements = midi.flat.notes

            for element in elements:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))

        except Exception as e:
            print(f"Could not read {file}: {e}")

    with open(NOTES_PATH, "wb") as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes):
    unique_notes = sorted(set(notes))
    note_to_int = dict((note_value, number) for number, note_value in enumerate(unique_notes))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]

        network_input.append([note_to_int[note_value] for note_value in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    n_vocab = len(unique_notes)

    network_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output, num_classes=n_vocab)

    return network_input, network_output, n_vocab


def create_model(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(128, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def train():
    notes = extract_notes()

    if len(notes) < SEQUENCE_LENGTH + 1:
        print("Not enough notes found. Please add more MIDI files.")
        return

    network_input, network_output, n_vocab = prepare_sequences(notes)

    model = create_model(network_input, n_vocab)

    print("Training model...")
    model.fit(network_input, network_output, epochs=20, batch_size=64)

    model.save(MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")


if __name__ == "__main__":
    train()