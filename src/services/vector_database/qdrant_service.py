import os
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
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

    def __init__(self, config_path: Optional[str] = None, host: str = "localhost", port: int = 6333):
        """
        Initialize the Qdrant service using application configuration.

        Args:
            config_path (str, optional): Path to application config YAML.
            host (str): Qdrant service host.
            port (int): Qdrant service port.
        """
        config = load_config(config_path) if config_path else load_config()
        self.default_collection_name = config["qdrant"]["collection"]
        self.distance = getattr(Distance, config["qdrant"].get("distance", "COSINE"))
        self.vector_size = config["embedding_model"]["vector_size"]
        self.client = QdrantClient(host, port=port)

    def create_collection(self, collection_name: Optional[str] = None, force_recreate: bool = False) -> None:
        """
        Create or recreate the voice embedding collection with proper vector size.

        Args:
            collection_name (str, optional): Collection name. Uses default if None.
            force_recreate (bool): If True, drops and recreates the collection.
        """
        collection_name = collection_name or self.default_collection_name
        
        if force_recreate:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                )
            )
        else:
            if collection_name not in [c.name for c in self.client.get_collections().collections]:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance,
                    )
                )


    def insert_embedding(
        self,
        embedding: list[float],
        record_id: str = None,
        payload: dict = None,
        collection_name: Optional[str] = None
    ) -> str:
        """
        Insert or upsert a new embedding vector with associated metadata.
        Creates the collection automatically if it doesn't exist.
        
        Args:
            embedding (List[float]): Vector representation of the speaker.
            record_id (str, optional): Unique identifier (UUID v4). If None, one is generated.
            payload (Dict[str, Any], optional): Additional speaker metadata.
            collection_name (str, optional): Collection name. Uses default if None.

        Returns:
            str: The UUID used for this record.
        """
        collection_name = collection_name or self.default_collection_name
        payload = payload or {}
        
        # Create collection if it doesn't exist
        if collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.create_collection(collection_name)
        
        if record_id is None:
            record_id = str(uuid.uuid4())
        elif not is_valid_uuid(record_id):
            payload["external_id"] = record_id
            record_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=collection_name,
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
        top_k: int = 3,
        collection_name: Optional[str] = None,
        with_vectors: bool = False
    ) -> List[Any]:
        """
        Search for the most similar voice embeddings.

        Args:
            embedding (List[float]): Query embedding vector.
            top_k (int): Number of top matches to retrieve.
            collection_name (str, optional): Collection name. Uses default if None.
            with_vectors (bool): Whether to include vectors in the response.

        Returns:
            List: List of matching points with scores and payloads.
        """
        collection_name = collection_name or self.default_collection_name
        results = self.client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=top_k,
            with_vectors=with_vectors
        )
        return results
    
    def query_by_payload(
        self,
        key: str,
        value: Any,
        limit: int = 20,
        return_vectors: bool = True,
        collection_name: Optional[str] = None
    ):
        """
        Query points by payload field.

        Args:
            key (str): Payload field key to filter by.
            value (Any): Value to match for the specified key.
            limit (int): Maximum number of points to return.
            return_vectors (bool): Whether to include vectors in the response.
            collection_name (str, optional): Collection name. Uses default if None.

        Returns:
            List: List of matching points.
        """
        collection_name = collection_name or self.default_collection_name
        filtro = Filter(
            must=[FieldCondition(key=key, match=MatchValue(value=value))]
        )

        pontos, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=filtro,
            with_payload=True,
            with_vectors=return_vectors, 
            limit=limit
        )
        return pontos

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        """
        Delete the embedding collection from Qdrant.
        
        Args:
            collection_name (str, optional): Collection name. Uses default if None.
        """
        collection_name = collection_name or self.default_collection_name
        self.client.delete_collection(collection_name)

    def list_collections(self) -> List[str]:
        """
        List all Qdrant collections available.

        Returns:
            List[str]: Names of all collections.
        """
        return [c.name for c in self.client.get_collections().collections]

    def set_default_collection(self, collection_name: str) -> None:
        """
        Set the default collection name for subsequent operations.
        
        Args:
            collection_name (str): New default collection name.
        """
        self.default_collection_name = collection_name

    def get_default_collection(self) -> str:
        """
        Get the current default collection name.
        
        Returns:
            str: Current default collection name.
        """
        return self.default_collection_name

