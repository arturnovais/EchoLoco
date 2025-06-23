from typing import ByteString
import os, mimetypes
from infra.storage import gcs_client

def _upload_to_gcs(local_path: str, dest_prefix: str = "tts") -> str:
    """
    Faz o upload de um arquivo para o Google Cloud Storage (GCS) e retorna o URI do GCS.

    Args:
        local_path (str): O caminho local do arquivo a ser enviado.
        dest_prefix (str): O prefixo de destino no GCS. Padrão é "tts".

    Returns:
        str: O URI do GCS do arquivo enviado.
    """
    filename = os.path.basename(local_path)
    dest_path = f"{dest_prefix}/{filename}"

    # Determina o Content-Type (por exemplo, audio/wav)
    ctype, _ = mimetypes.guess_type(filename)
    ctype = ctype or "application/octet-stream"

    gs_uri = gcs_client.upload_file(local_path, dest_path, content_type=ctype)
    os.remove(local_path)
    return gs_uri


def _download_from_gcs(gs_uri: str) -> ByteString:
    """
    Baixa um objeto no Google Cloud Storage e devolve os bytes.

    Args:
        gs_uri (str): URI no formato gs://bucket/path/file.ext

    Returns:
        bytes: Conteúdo bruto do arquivo.
    """
    return gcs_client.download_bytes(gs_uri)