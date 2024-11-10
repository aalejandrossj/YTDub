from download import download_audio
from transcript import transcribe_audio
from translate import translate_transcript

print("Bienvenido a YTDub, elige qué quieres hacer")
print("\n1. Comenzar el proceso de doblar un video al español")
print("2. Salir")

opcion = input("Elige una opción: ")

if opcion == "1":
    URLVid = input("Pon la URL del video a traducir: ")
    audio_path = download_audio(URLVid)  # Esta función debe ser implementada.
    transcript_path = transcribe_audio(audio_path)
    txt_translated_path = translate_transcript(transcript_path)

else:
    print("Programa terminado.")
