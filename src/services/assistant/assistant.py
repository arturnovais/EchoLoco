import os
from utils.load_config import load_config
from services.assistant.llm import GPT, GEMINI

class Assistant:
    def __init__(self, config_path=None, prompt_path=None):
        config = load_config(config_path) if config_path else load_config()
        self.name = config['assistant'].get('name', 'Assistente')
        self.greeting = config['assistant'].get('greeting', 'Olá!')
        self.farewell = config['assistant'].get('farewell', 'Até logo!')

        llm_config = config['llm']
        provider = llm_config.get('provider', 'gpt').lower()


        if provider == "gpt":
            model = llm_config.get('gpt_model', 'gpt-4o-mini')
            self.llm = GPT(model=model)
        elif provider == "gemini":
            model = llm_config.get('gemini_model', 'gemini-2.5-flash-preview-04-17')
            self.llm = GEMINI(model=model)
        else:
            raise ValueError(f"Provider não suportado: {provider}")


        prompt_path = prompt_path or os.path.join(os.path.dirname(__file__), "prompt.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.prompt_base = f.read().strip()
        else:
            self.prompt_base = ""


    def reply(self, user_text):
        messages = [
            {"role": "system", "content": self.prompt_base},
            {"role": "user", "content": user_text}
        ]
        resposta = self.llm.invoke(messages)
        return resposta
    
    def reply_api(self, messages):
        resposta = self.llm.invoke(messages)
        return resposta

    def welcome(self):
        return self.greeting

    def goodbye(self):
        return self.farewell
