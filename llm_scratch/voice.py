from elevenlabs import play
from elevenlabs.client import ElevenLabs
import os

def text_to_voice(text, out_fn):
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
