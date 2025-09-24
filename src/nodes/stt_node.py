import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class stt:
    def __init__(self):
        self.model_id = "/home/abhishek-maurya/learn-lang/model/wishper-large-v3-turbo/whisper-large-v3-turbo"
        self.device = "cuda"if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
    def _tts_model(self, audio):
        self.model.to(self.device)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
        )
        pipe.model.to(self.device)
        result = pipe(audio, return_timestamps=True, chunk_length_s=30, stride_length_s=5)
        pipe.model.to("cpu")
        return result
    def convert(self, audio_path):
        text = self._tts_model(audio_path)
        return text['text']