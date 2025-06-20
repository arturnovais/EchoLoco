"""
Script para converter arquivos OPUS para WAV.
Execute o script com o comando:
python -m src.utils.convert_wav

O script irá converter todos os arquivos OPUS em uma pasta específica para WAV.

Especialmente útil para áudis que vem do WhatsApp.
"""

import os
import subprocess

def converter_opus_para_wav(pasta):
    for arquivo in os.listdir(pasta):
        if arquivo.lower().endswith('.opus'):
            caminho_opus = os.path.join(pasta, arquivo)
            caminho_wav = os.path.join(pasta, os.path.splitext(arquivo)[0] + '.wav')

            print(f'Convertendo {arquivo} para WAV...')
            
            comando = [
                'ffmpeg', '-y',  # -y para sobrescrever arquivos sem perguntar
                '-i', caminho_opus,
                caminho_wav
            ]
            
            try:
                subprocess.run(comando, check=True)
                os.remove(caminho_opus)
                print(f'Arquivo {arquivo} convertido e original removido.')
            except subprocess.CalledProcessError as e:
                print(f'Erro ao converter {arquivo}: {e}')

if __name__ == '__main__':
    pasta_alvo = 'audios/Gustavo'  # Substitua pelo caminho da sua pasta
    converter_opus_para_wav(pasta_alvo)
