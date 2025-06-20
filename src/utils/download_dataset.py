"""
Script para baixar o dataset do Google Drive e descompactar o arquivo ZIP.
Execute o script com o comando:
python -m src.utils.download_dataset

O arquivo ZIP contém os seguintes arquivos:
- audios: pasta com os áudios
- audios_metadata.csv: arquivo CSV com os metadados dos áudios

O arquivo CSV contém as seguintes colunas:
- path: caminho do áudio
- speaker: nome do locutor
"""

import requests
import zipfile
import os

def download(file_id, destination):
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0&confirm=t&uuid=7cfea6d1-7d10-4d72-8816-8d38b1c36254&at=AIrpjvO7ioPI-EsZnqKt6qMpMuHU%3A1738285656505"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Arquivo baixado com sucesso: {destination}")
        return True
    else:
        print(f"Falha no download. Código de status: {response.status_code}")
        return False

def descompactar_zip(arquivo_zip, destino_pasta):
    try:
        with zipfile.ZipFile(arquivo_zip, 'r') as zip_ref:
            zip_ref.extractall(destino_pasta)
        print(f"Arquivo descompactado com sucesso em: {destino_pasta}")
    except zipfile.BadZipFile:
        print("Erro: o arquivo baixado não é um arquivo ZIP válido.")

if __name__ == "__main__":
    zip_path = "dataset.zip"
    extract_path = "audios"

    if download("14zZVEFXlzN91anDWGyavXtA1ZvG69oxK", zip_path):
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        descompactar_zip(zip_path, extract_path)
