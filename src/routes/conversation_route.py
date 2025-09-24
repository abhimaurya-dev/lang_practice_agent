import io
import webrtcvad
from fastapi import APIRouter, WebSocket # type: ignore
from soundfile import sf
import numpy as np

from ..pipelines.conversation_pipeline import Conversation_Pipeline

router = APIRouter()
vad = webrtcvad.Vad(2)  # Set aggressiveness mode (0-3)
@router.websocket("/conversation")
async def conversation_endpoint(websocket: WebSocket):
    conversation_pipeline = Conversation_Pipeline()
    await websocket.accept()
    audio_buffer = []
    silence_counter_ms = 0
    max_silence_ms = 800

    while True:
        try:
            data = await websocket.receive_bytes()
            audio_buf = io.BytesIO(data)
            audio, sr = sf.read(audio_buf, dtype='float32')
            if sample_rate is None:
                sample_rate = sr
            
            audio_int16 = (audio * 32768).astype(np.int16)
            if audio_int16.ndim > 1:
                audio_int16 = audio_int16.mean(axis=1).astype(np.int16)
                frame_bytes = audio_int16.tobytes()
                is_speech = vad.is_speech(frame_bytes, sample_rate)
                if is_speech:
                    audio_buffer.append(audio)
                    silence_counter_ms = 0
                else:
                    silence_counter_ms += len(audio) / sample_rate * 1000

                if silence_counter_ms > max_silence_ms and audio_buffer:
                    merged_audio = np.concatenate(audio_buffer, axis=0)
                    text = conversation_pipeline.transcribe(merged_audio)
                    response_text = conversation_pipeline.generate_response(text)
                    response_audio = conversation_pipeline.synthesize(response_text)
                    output_buf = io.BytesIO()
                    sf.write(output_buf, response_audio, sample_rate, format='WAV')
                    await websocket.send_bytes(output_buf.getvalue())
                    audio_buffer = []
                    silence_counter_ms = 0
        except Exception as e:
            await websocket.close()
            print(f"Error: {e}")
            break