from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import ChatPromptTemplate

class ChatLLMPipeline:
    def __init__(self, memory=None):
        self.model_name = ""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.memory = memory
        self.llm_pipeline = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
        self.hf_pipeline = HuggingFacePipeline(pipeline=self.llm_pipeline)
    
    def chat(self, system_prompt, user_prompt):
        
        if self.memory is None:
            raise ValueError("Memory is not set for the pipeline.")
        
        history = self.memory.get_history()
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", f"history: {history} \n new_prompt: {user_prompt}")
        ])

        
        llm_chain = chat_prompt | self.hf_pipeline
        respone = llm_chain.invoke()
        return respone['text']