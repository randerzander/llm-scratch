import os


#def piper(text, out_fn, model_path="/home/dev/projects/models/en_US-lessac-medium.onnx"):
def piper(text, out_fn, model_path="/home/dev/projects/models/en_US-ryan-medium.onnx"):
    from piper.voice import PiperVoice
    from pydub import AudioSegment
    import wave
    import numpy as np
    import sounddevice as sd

    voice = PiperVoice.load(model_path)
    
    # Create a temporary WAV file
    temp_wav = "temp_output.wav"
    
    with wave.open(temp_wav, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        voice.synthesize(text, wav_file)
    
    # Convert WAV to MP3 using pydub
    audio = AudioSegment.from_wav(temp_wav)
    audio.export(out_fn, format="mp3")
    
    # Remove the temporary WAV file
    os.remove(temp_wav)


def piper_wav(text, out_fn):
    model_path = "/home/dev/projects/models/en_US-lessac-medium.onnx"
    voice = PiperVoice.load(model_path)
    with wave.open(out_fn, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        voice.synthesize(text, wav_file)


def text_to_voice(text, out_fn):
    from elevenlabs.client import ElevenLabs

    api_key = os.getenv("ELEVENLABS_API_KEY")
    client = ElevenLabs(
      api_key=api_key,
    )

    audio = client.generate(
      text=text,
      voice="Brian",
      model="eleven_multilingual_v2"
    )
    
    audio_data = b''.join(chunk for chunk in audio)
    with open(out_fn, "wb") as file:
        file.write(audio_data)
    print(f"Saved to {out_fn}")
