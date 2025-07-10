import logging
from chromadb import Client

logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

class Chroma_DB:
    """
        Initialize the Chroma_DB wrapper.

        Args:
            client (chromadb.Client): ChromaDB client instance.
            collection_name (str): Name of the collection to use or create.
    """
    def __init__(self, client: Client, collection_name: str):
        self.client = client
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            )
   
        
    def insert_into_vectordb(self, embedding: list, item_id: str, meta: dict, document: str = None, suppress_logs: bool = False) -> None:
        """
        Insert a new vector entry into the collection if it doesn't exist by unique_key.

        Args:
            embedding (list): The embedding vector to insert.
            item_id (str): Unique identifier for the entry.
            meta (dict): Metadata.
            document (str): Documents.
            supperss_logs (bool): suppress_logs (bool, optional): If True, disables logging output
            
        """
        existing = self.collection.get(ids=[item_id])
        if item_id not in existing.get("ids", []):
            try:
                add_kwargs = {
                    "embeddings": [embedding],
                    "ids": [item_id],
                }

                if meta is not None:
                    add_kwargs["metadatas"] = [meta]

                if document is not None:
                    add_kwargs["documents"] = [document]

                self.collection.add(**add_kwargs)
                   
                if not suppress_logs:
                   logger.info(f"Entity with ID {item_id} inserted successfully.")
            except Exception as e:
                if not suppress_logs:
                    logger.exception(e)
                raise RuntimeError(f"Failed to insert embedding for ID '{item_id}'") from e
                
        else:
            if not suppress_logs:
               logger.warning(f"Entity with ID {item_id} already exist.")
    
    def size(self) -> int:
        """
        Get the number of items currently in the collection.

        Returns:
            int: Number of items if retrievable, else None.
        """
        try:
            count = self.collection.count()
            return count
        except Exception as e:
            logger.exception(e)
            raise RuntimeError("Failed to retrieve collection size") from e
           
            
    def delete_from_vectordb(self, item_id: str, suppress_logs: bool = False) -> None:
        """
        Delete entries from the collection matching the given filename metadata.

        Args:
            item_id (str): Unique identifier 
            supperss_logs (bool): To control logs while using 'self.update_vectordb()'
        """
        
        existing = self.collection.get(ids=[item_id])
        if existing:
            try:
                self.collection.delete(ids=[item_id])
                if not suppress_logs:
                   logger.info(f"Embedding associated with ID '{item_id}' deleted successfully.")
            except Exception as e:
                if not suppress_logs:
                    logger.exception(e)
                raise RuntimeError(f"Failed to delete embedding for ID '{item_id}'") from e
        else:
            if not suppress_logs:
               logger.warning(f"Embedding associated with ID '{item_id}' does not exist")
        
        
    def update_vectordb(self, embedding: list, item_id: str, meta: dict) -> None:
        """
        Update an existing entry by deleting any existing entries with the filename
        and inserting the new data.

        Args:
            embedding (list): New embedding vector.
            item_id (str): Unique identifier.
            meta (dict): Metadata.
            
        """
        try:
            self.delete_from_vectordb(item_id,suppress_logs=True)
            self.insert_into_vectordb(embedding, item_id, meta, suppress_logs=True) 
            logger.info(f"Embedding associated with ID '{item_id}' updated successfully.")
        except Exception as e:
            logger.exception(e)
            raise RuntimeError(f"Failed to update embedding for ID '{item_id}'") from e

        
    def retrieve_top_n_from_vectordb(self, embedding: list, meta_filter: dict, k: int) -> dict[str]:
        """
        Retrieve the top-k most similar entries to the given embedding, optionally filtered by metadata.

        Args:
            embedding (list[float]): Query embedding vector.
            meta_filter (dict, optional): Metadata filter. Defaults to empty dict (no filter).
            k (int, optional): Number of results to retrieve. Defaults to 3.

        Returns:
            dict[str, list]: Query results containing 'ids', 'distances', 'documents', and 'metadatas'.
        """
        try:
            query_kwargs = {
                "query_embeddings": [embedding],
                "n_results": k,
            }

            if meta_filter is not None:
                query_kwargs["where"] = meta_filter

            results = self.collection.query(**query_kwargs)
            return results
        except Exception as e:
            logger.exception(e)
            raise RuntimeError("Failed to retrieve top-k embeddings from ChromaDB") from e
            

