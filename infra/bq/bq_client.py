"""
Cliente utilitário para Google BigQuery.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence, Any

from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT_ID   = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
DATASET_ID   = os.getenv("BQ_DATASET", "speech_chatbot")
_SCOPES      = ("https://www.googleapis.com/auth/bigquery",)

@lru_cache(maxsize=1)
def _client() -> bigquery.Client:
    """
    Cria (uma única vez) e devolve um bigquery.Client autenticado.

    Ordem de procura das credenciais:
      1. GOOGLE_APPLICATION_CREDENTIALS_JSON  (string JSON inline)
      2. GOOGLE_APPLICATION_CREDENTIALS       (caminho para o arquivo)
      3. service_account.json na raiz do projeto
      4. Application Default Credentials      (gcloud auth application-default login)
    """
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
        creds_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
        creds = service_account.Credentials.from_service_account_info(
            creds_info, scopes=_SCOPES
        )
        return bigquery.Client(credentials=creds, project=creds.project_id)

    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        return bigquery.Client()

    root = Path(__file__).parent.parent.parent
    sa_path = root / "service_account.json"
    if sa_path.exists():
        creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=_SCOPES
        )
        return bigquery.Client(credentials=creds, project=creds.project_id)

    # Fallback: ADC padrão
    return bigquery.Client()

def _table_ref(
    table: str,
    dataset: str | None = None,
    project: str | None = None,
) -> bigquery.TableReference:
    """Devolve TableReference completo no formato project.dataset.table."""
    project = project or PROJECT_ID or _client().project
    dataset = dataset or DATASET_ID
    return bigquery.DatasetReference(project, dataset).table(table)

def insert_rows(
    table: str,
    rows: Sequence[Mapping[str, Any]],
    skip_invalid_rows: bool = False,
    ignore_unknown_values: bool = False,
) -> Sequence[Mapping[str, Any]]:
    """
    Insere linhas via streaming no BigQuery.

    Retorna lista de erros. Lista vazia == sucesso.
    """
    ref = _table_ref(table)
    errors = _client().insert_rows_json(
        ref,
        rows,
        skip_invalid_rows=skip_invalid_rows,
        ignore_unknown_values=ignore_unknown_values,
    )
    return errors  # Se [], nenhuma falha

def query(
    sql: str,
    job_config: bigquery.QueryJobConfig | None = None,
) -> bigquery.table.RowIterator:
    """
    Executa uma consulta SQL arbitrária.

    Use list(result) ou result.to_dataframe() para materializar.
    """
    job = _client().query(sql, job_config=job_config)
    return job.result()

def list_rows(
    table: str,
    selected_fields: Iterable[bigquery.SchemaField] | None = None,
    max_results: int | None = None,
) -> bigquery.table.RowIterator:
    """
    Itera sobre as linhas de uma tabela (leitura simples, sem SQL).
    """
    ref = _table_ref(table)
    return _client().list_rows(ref, selected_fields=selected_fields, max_results=max_results)
