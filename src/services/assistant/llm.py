import os
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class GPT():
    def __init__(self, api_key=None, model='gpt-4.1-mini'):
        
        self.api_key = api_key
        
        if self.api_key is None:
            self.api_key = self.get_api_key()
            
        self.model = model
        
        try:
            self.client = OpenAI(api_key=self.api_key)

        except Exception as e:
            print(f"Erro ao conectar com a API do OpenAI: {e}")
            raise e
        
        
    def get_api_key(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key.strip()
        
        
    def invoke(self, messages):
        # Se receber um dicionário único, converte para lista
        if isinstance(messages, dict):
            messages = [messages]
        
        response = self.client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
        )
        
        return response.choices[0].message.content.strip()
    
    
    
class GEMINI():
    def __init__(self, api_key=None, model='gemini-2.5-flash-preview-04-17'):
        
        self.api_key = api_key
        
        if self.api_key is None:
            self.api_key = self.get_api_key()
            
        self.model = model
        
        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(model_name=self.model)
            
        except Exception as e:
            print(f"Erro ao conectar com a API do Gemini: {e}")
            raise e
        
    def get_api_key(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            return api_key.strip()
        
    def invoke(self, prompt):
        try:
            response = self.client.generate_content(prompt)
            return response.text.strip()
        
        except Exception as e:
            print(f"Erro ao gerar resposta: {e}")
            return None