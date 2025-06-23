import os, mimetypes
from infra.storage import gcs_client

def _upload_to_gcs(local_path: str, dest_prefix: str = "tts") -> str:
    filename = os.path.basename(local_path)
    dest_path = f"{dest_prefix}/{filename}"

    # Determina Content-Type (ex.: audio/wav)
    ctype, _ = mimetypes.guess_type(filename)
    ctype = ctype or "application/octet-stream"

    gs_uri = gcs_client.upload_file(local_path, dest_path, content_type=ctype)
    os.remove(local_path)
    return gs_uri