import pickle
import random
import numpy as np
from music21 import instrument, note, stream, chord
from tensorflow.keras.models import load_model

MODEL_PATH = "music_model.keras"
NOTES_PATH = "notes.pkl"
OUTPUT_PATH = "generated_music/generated_music.mid"
SEQUENCE_LENGTH = 50


def generate_notes():
    with open(NOTES_PATH, "rb") as filepath:
        notes = pickle.load(filepath)

    unique_notes = sorted(set(notes))
    n_vocab = len(unique_notes)

    note_to_int = dict((note_value, number) for number, note_value in enumerate(unique_notes))
    int_to_note = dict((number, note_value) for number, note_value in enumerate(unique_notes))

    network_input = []

    for i in range(0, len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[note_value] for note_value in sequence_in])

    start = random.randint(0, len(network_input) - 1)
    pattern = network_input[start]

    model = load_model(MODEL_PATH)

    prediction_output = []

    for _ in range(100):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)

        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output


def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if "." in pattern or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            chord_notes = []

            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)

            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=OUTPUT_PATH)

    return OUTPUT_PATH


def generate_music():
    prediction_output = generate_notes()
    output_file = create_midi(prediction_output)
    print(f"Generated MIDI saved at {output_file}")


if __name__ == "__main__":
    generate_music()