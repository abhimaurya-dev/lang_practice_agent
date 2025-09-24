import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
import soundfile as sf

class Tts:
    def __init__(self):
        self.model_id = "/home/abhishek-maurya/learn-lang/model/marvis-tts-250m-v0.1-transformers"
        self.device = "cuda"if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = CsmForConditionalGeneration.from_pretrained(self.model_id)
    def _tts_model(self, text):
        self.model.to(self.device)
        inputs = self.processor(text, add_special_tokens=True, return_tensors="pt").to(self.device)
        audio = self.model.generate(input_ids=inputs['input_ids'], output_audio=True)
        self.model.to("cpu")
        return audio
    def generate_audio(self, text):
        gen_audio = self._tts_model(text)
        return gen_audio