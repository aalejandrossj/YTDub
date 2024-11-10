import torch
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

# Carga el modelo y el procesador
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True
)

TRANSCRIPTS_DIR = 'transcripts'

def transcribe_audio(audio_path):
    # Verifica si el archivo de audio existe
    if not os.path.exists(audio_path):
        print(f"El archivo de audio en {audio_path} no existe.")
        return None

    # Crea la carpeta transcripts si no existe
    if not os.path.exists(TRANSCRIPTS_DIR):
        os.makedirs(TRANSCRIPTS_DIR)

    # Procesa el audio y realiza la transcripción
    print(f"Transcribiendo el audio en {audio_path}...")
    result = pipe(audio_path)
    
    # Obtiene el texto transcrito
    transcription_text = result["text"]
    
    # Define la ruta de la transcripción en la carpeta transcripts
    transcription_path = os.path.join(TRANSCRIPTS_DIR, os.path.splitext(os.path.basename(audio_path))[0] + "_transcription.txt")
    with open(transcription_path, "w") as f:
        f.write(transcription_text)

    print(f"Transcripción completada y guardada en {transcription_path}")
    
    return transcription_path  # Devuelve la ruta del archivo de transcripción

if __name__ == "__main__":
    audio_path = ""
    transcribe_audio(audio_path)
