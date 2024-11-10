from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import numpy as np
import nltk

# Download 'punkt' tokenizer for sentence splitting
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Initialize the processor and models
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move models to the device
model = model.to(device)
vocoder = vocoder.to(device)

# Function to split text into manageable chunks
def split_text(text, max_length=500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Define the synthesis function
def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    # Handle long input sequences by truncating
    if input_ids.size(1) > 600:
        input_ids = input_ids[:, :600]
    speech = model.generate_speech(
        input_ids, speaker_embeddings.to(device), vocoder=vocoder
    )
    return speech.cpu().numpy()

# Your long text
long_text = """Your very long text goes here. It can be multiple paragraphs long, and we'll split it into chunks to process."""

# Split the text into chunks
text_chunks = split_text(long_text, max_length=500)

# Generate speech for each chunk and concatenate
audio_outputs = []
for chunk in text_chunks:
    speech = synthesise(chunk)
    audio_outputs.append(speech)

# Concatenate all audio chunks
full_speech = np.concatenate(audio_outputs)

# Save the audio to a file
sf.write('output.wav', full_speech, 16000)
