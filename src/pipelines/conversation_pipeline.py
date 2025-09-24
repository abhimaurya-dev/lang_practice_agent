from src.nodes.stt_node import Stt
from src.nodes.tts_node import Tts
from .llm_pipeline import ChatLlmPipeline

class TemporaryMemory:
    def __init__(self):
        self.memory = []
    def add(self, text):
        if len(self.memory) > 10:
            self.memory.pop(0)
        self.memory.append(text)
    def get_history(self):
        return " ".join(self.memory)

class ConversationPipeline:
    def __init__(self):
        self.stt_node = Stt()
        self.tts_node = Tts()
        self.temporary_memory = TemporaryMemory()
        self.chat_llm = ChatLlmPipeline(memory=self.temporary_memory)
    def generate_response(self, text):
        chat_response = self.chat_llm.chat(
            system_prompt="You are a helpful assistant.",
            user_prompt=text
        )
        return chat_response
    def transcribe(self, audio):
        return self.stt_node.get_text(audio)
    def synthesize(self, text):
        return self.tts_node.generate_audio(text)