"""
Cliente para o Google Cloud Storage (Bucket de armazenamento dos audios).
"""
from __future__ import annotations

import json
import os
import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Iterable, Union

from google.cloud import storage
from google.oauth2.service_account import Credentials

# --------------------- Config -------------------------------------------------
BUCKET_NAME = os.getenv("GCS_BUCKET", "audios-entrada")
DEFAULT_EXPIRATION = int(os.getenv("GCS_URL_TTL", "3600"))          # segundos

_SCOPES = ("https://www.googleapis.com/auth/devstorage.read_write",)

# --------------------- Helpers internos --------------------------------------
@lru_cache(maxsize=1)
def _client() -> storage.Client:
    """
    Cria (uma vez) e devolve o storage.Client.
    Usa:
      • GOOGLE_APPLICATION_CREDENTIALS (arquivo)
      • ou GOOGLE_APPLICATION_CREDENTIALS_JSON (json inline)
      • ou ADC padrão (gcloud auth application-default login)
    """
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
        creds = Credentials.from_service_account_info(
            json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]),
            scopes=_SCOPES,
        )
        return storage.Client(credentials=creds)
    return storage.Client()          # ADC  

def _blob(path: str, bucket_name: str | None = None) -> storage.Blob:
    bucket = _client().bucket(bucket_name or BUCKET_NAME)
    return bucket.blob(path)

def _split_gs_uri(uri: str) -> tuple[str, str]:
    """
    Divide gs://bucket/obj -> (bucket, obj)
    """
    if not uri.startswith("gs://"):
        raise ValueError("URI precisa começar com gs://")
    _, rest = uri.split("gs://", 1)
    bucket, _, blob = rest.partition("/")
    if not bucket or not blob:
        raise ValueError("URI deve ser gs://bucket/obj")
    return bucket, blob

# --------------------- API pública -------------------------------------------
def upload_file(
    local: Union[str, Path, BinaryIO],
    dest_path: str,
    bucket_name: str | None = None,
    overwrite: bool = False,
    content_type: str | None = None,
) -> str:
    """Sobe um arquivo/disco *ou* file-like para o GCS e devolve o gs://…"""
    blob = _blob(dest_path, bucket_name)
    if blob.exists() and not overwrite:
        raise FileExistsError(f"{blob.name} já existe em {blob.bucket.name}")
    if isinstance(local, (str, Path)):
        blob.upload_from_filename(str(local), content_type=content_type)
    else:
        blob.upload_from_file(local, content_type=content_type)
    return f"gs://{blob.bucket.name}/{blob.name}"

def download_bytes(
    gcs_path: str,
) -> bytes:
    """Baixa o objeto inteiro como bytes."""
    bucket_name, blob_name = _split_gs_uri(gcs_path)
    blob = _blob(blob_name, bucket_name)
    return blob.download_as_bytes()

def generate_signed_url(
    gcs_path: str,
    expires: int = DEFAULT_EXPIRATION,
    method: str = "GET",
    content_type: str | None = None,
) -> str:
    """Gera URL V4 assinada."""
    bucket_name, blob_name = _split_gs_uri(gcs_path)
    blob = _blob(blob_name, bucket_name)
    return blob.generate_signed_url(
        version="v4",
        expiration=dt.timedelta(seconds=expires),
        method=method.upper(),
        content_type=content_type,
    )

def list_objects(
    prefix: str = "",
    bucket_name: str | None = None,
) -> Iterable[str]:
    """Lista caminhos (gs://…) começando por prefix."""
    bucket = _client().bucket(bucket_name or BUCKET_NAME)
    for obj in bucket.list_blobs(prefix=prefix):
        yield f"gs://{bucket.name}/{obj.name}"

def delete_object(gcs_path: str) -> None:
    """Exclui um objeto."""
    bucket_name, blob_name = _split_gs_uri(gcs_path)
    _blob(blob_name, bucket_name).delete()