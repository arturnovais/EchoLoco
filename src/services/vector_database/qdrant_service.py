import os
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from src.utils.load_config import load_config
import uuid


def is_valid_uuid(val):
        try:
            uuid.UUID(str(val))
            return True
        except ValueError:
            return False


class QdrantService:
    """
    Service layer for interaction with a Qdrant vector database.
    Manages collection lifecycle and vector operations for voice embeddings.
    """

    def __init__(self, config_path: Optional[str] = None, host: str = "localhost", port: int = 6334):
        """
        Initialize the Qdrant service using application configuration.

        Args:
            config_path (str, optional): Path to application config YAML.
            host (str): Qdrant service host.
            port (int): Qdrant service port.
        """
        config = load_config(config_path) if config_path else load_config()
        self.collection_name = config["qdrant"]["collection"]
        self.distance = getattr(Distance, config["qdrant"].get("distance", "COSINE"))
        self.vector_size = config["embedding_model"]["vector_size"]
        self.client = QdrantClient(host, port=port)

    def create_collection(self, force_recreate: bool = False) -> None:
        """
        Create or recreate the voice embedding collection with proper vector size.

        Args:
            force_recreate (bool): If True, drops and recreates the collection.
        """
        if force_recreate:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                )
            )
        else:
            if self.collection_name not in [c.name for c in self.client.get_collections().collections]:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance,
                    )
                )


    def insert_embedding(
        self,
        embedding: list[float],
        record_id: str = None,
        payload: dict = None
    ) -> str:
        """
        Insert or upsert a new embedding vector with associated metadata.
        
        Args:
            embedding (List[float]): Vector representation of the speaker.
            record_id (str, optional): Unique identifier (UUID v4). If None, one is generated.
            payload (Dict[str, Any], optional): Additional speaker metadata.

        Returns:
            str: The UUID used for this record.
        """
        payload = payload or {}
        if record_id is None:
            record_id = str(uuid.uuid4())
        elif not is_valid_uuid(record_id):
            payload["external_id"] = record_id
            record_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[{
                "id": record_id,
                "vector": embedding,
                "payload": payload
            }]
        )
        return record_id

     

    def search_similar(
        self,
        embedding: List[float],
        top_k: int = 3
    ) -> List[Any]:
        """
        Search for the most similar voice embeddings.

        Args:
            embedding (List[float]): Query embedding vector.
            top_k (int): Number of top matches to retrieve.

        Returns:
            List: List of matching points with scores and payloads.
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k
        )
        return results

    def delete_collection(self) -> None:
        """
        Delete the current embedding collection from Qdrant.
        """
        self.client.delete_collection(self.collection_name)

    def list_collections(self) -> List[str]:
        """
        List all Qdrant collections available.

        Returns:
            List[str]: Names of all collections.
        """
        return [c.name for c in self.client.get_collections().collections]

