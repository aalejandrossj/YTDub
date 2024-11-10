from transformers import MarianMTModel, MarianTokenizer
import os


modelo_nombre = 'Helsinki-NLP/opus-mt-en-es' 
tokenizador = MarianTokenizer.from_pretrained(modelo_nombre)
modelo = MarianMTModel.from_pretrained(modelo_nombre)
TRANSLATES_DIR = "translates"

def translate_transcript(transcript_path):

    if not os.path.exists(TRANSLATES_DIR):
        os.makedirs(TRANSLATES_DIR)

    transcribed_text = leer_archivo(transcript_path)
    texto_spliteado = splitear_texto(transcribed_text)
    texto_traducido = traducir_texto_completo(texto_spliteado, modelo, tokenizador)

    # Guardar el texto traducido en un archivo .txt
    output_path = os.path.join(TRANSLATES_DIR, 'texto_traducido.txt')
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(texto_traducido)
    return output_path

def leer_archivo(transcript_path):
    with open(transcript_path, "r", encoding='utf-8') as f:
        transcribed_text = f.read()
    return transcribed_text

def splitear_texto(transcribed_text, max_words_per_segment=100):
    # Dividir el texto en palabras
    words = transcribed_text.split()
    
    # Crear los segmentos
    segments = []
    for i in range(0, len(words), max_words_per_segment):
        segment = " ".join(words[i:i + max_words_per_segment])
        segments.append(segment)
    
    return segments

def traducir_texto_completo(segments, modelo, tokenizador):
    traducciones = []
    for segment in segments:
        # Tokenizar el segmento
        tokens = tokenizador([segment], return_tensors='pt', padding=True)
        # Generar la traducción
        traduccion = modelo.generate(**tokens)
        # Decodificar la traducción
        texto_traducido = tokenizador.batch_decode(traduccion, skip_special_tokens=True)
        traducciones.append(texto_traducido[0])
    # Unir todas las traducciones en un solo texto
    texto_traducido_completo = ' '.join(traducciones)
    return texto_traducido_completo

    
def main():   
    transcript_path = "transcripts\Back to Basics： Understanding Retrieval Augmented Generation (RAG)_transcription.txt"  # Reemplaza esto con la ruta a tu archivo
    transcribed_text = leer_archivo(transcript_path)
    segments = splitear_texto(transcribed_text)
    
    # Traducir el texto completo
    texto_traducido = traducir_texto_completo(segments, modelo, tokenizador)
    
    if not os.path.exists(TRANSLATES_DIR):
        os.makedirs(TRANSLATES_DIR)
    output_path = os.path.join(TRANSLATES_DIR, 'texto_traducido.txt')
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(texto_traducido)
    return output_path

if __name__ == '__main__':
    main()
