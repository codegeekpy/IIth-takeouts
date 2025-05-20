import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
#need to download the tensor flow 

# Generate a synthetic dataset of notes for simplicity
notes = [60, 62, 64, 65, 67, 69, 71, 72]  # MIDI note numbers for a C major scale
sequence_length = 4

# Prepare the dataset
X, y = [], []
for i in range(len(notes) - sequence_length):
    X.append(notes[i:i+sequence_length])
    y.append(notes[i+sequence_length])

X = np.array(X)
y = to_categorical(y, num_classes=128)  # MIDI has 128 possible notes

# Reshape X for LSTM input
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build the RNN model
model = Sequential([
    LSTM(128, activation='relu', input_shape=(sequence_length, 1)),
    Dense(128, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=200, batch_size=8, verbose=0)
print("Model trained!")

# Generate a music sequence
def generate_music(seed, length=50):
    generated = []
    input_seq = np.array(seed).reshape((1, sequence_length, 1))

    for _ in range(length):
        prediction = model.predict(input_seq, verbose=0)
        next_note = np.argmax(prediction) #argmax is everywhere
        generated.append(next_note)
        input_seq = np.append(input_seq[:, 1:, :], [[[next_note]]], axis=1)

    return generated

# Seed for music generation
seed = [60, 62, 64, 65]  # Starting notes
music_sequence = generate_music(seed, length=50)
print("Generated music sequence (MIDI notes):", music_sequence)

# Convert MIDI to audio (requires a MIDI library like pretty_midi)
import pretty_midi

def create_midi_from_sequence(sequence, output_file="output.mid"):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    start_time = 0.0
    duration = 0.5  # Each note lasts 0.5 seconds

    for note in sequence:
        midi_note = pretty_midi.Note(velocity=100, pitch=note, start=start_time, end=start_time + duration)
        piano.notes.append(midi_note)
        start_time += duration

    midi.instruments.append(piano)
    midi.write(output_file)
    print(f"MIDI file saved to {output_file}")

# Save generated sequence to a MIDI file
create_midi_from_sequence(music_sequence)