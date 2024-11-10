import yt_dlp
import os
from transcript import transcribe_audio

AUDIO_DIR = 'audios'

## Función para descargar el audio
def download_audio(URLVid):
    # Si no existe la carpeta audios, la crea
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR)
    
    # Opciones para descargar el audio
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(AUDIO_DIR, '%(title)s.%(ext)s'),  # Plantilla para el nombre de archivo
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    # Intenta descargar el audio
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(URLVid, download=True)  # Cambiado download=False a True
            title = info_dict.get('title', 'audio')
            final_output_path = os.path.join(AUDIO_DIR, f"{title}.mp3")
            
            # Verifica si el audio se descargó correctamente
            downloaded_files = os.listdir(AUDIO_DIR)
            downloaded_file = next(
                (file for file in downloaded_files if file.endswith(".mp3")), None
            )

            if downloaded_file:
                final_output_path = os.path.join(AUDIO_DIR, downloaded_file)
                print(f"Audio descargado correctamente en {final_output_path}")
                return final_output_path  # Retorna la ruta del archivo descargado
            else:
                print(f"Fallo al guardar el audio en {final_output_path}")
                return None
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        return None


def main():
    URLVid = input("Pon la URL del video a traducir: ")
    audio_path = download_audio(URLVid)
    
    if audio_path:
        transcribe_audio(audio_path)  # Llama a la función de transcripción con la ruta del archivo descargado
    else:
        print("No se pudo descargar el audio, no se puede proceder con la transcripción.")


if __name__ == '__main__':
    main()
