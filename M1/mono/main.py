from .processing import Audio_Embedding,Image_Embedding,Document_Embedding
from .vectordb import Chroma_DB
import logging
import os

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

logger = logging.getLogger(__name__)


class mono_audio:

    """
    Manages audio embeddings insertion, storage in vector DB, fine-tuning, and retrieval.

    Args:
        client: Chroma DB client instance.
        collection_name (str): Name of the vector database collection.
    """

    def __init__(self,client,collection_name):
        self.client = client
        self.collection_name = collection_name
        self.vector_db_default = Chroma_DB(client, f"{collection_name}_audio_default")
        self.vector_db_custom = Chroma_DB(client, f"{collection_name}_audio_custom")
        self.audio_class = Audio_Embedding()

    def mount_audio(self, audio_path: str, meta: dict = None, model_path: str = None) -> None:

        """
        Embeds and mounts a single audio file into vector DB.

        Args:
            audio_path (str): Path to audio file.
            meta (dict, optional): Metadata to store.
            model_path (str, optional): Path to custom model for embedding.
        """

        if model_path is None:
            embedding = self.audio_class.create_embedding(audio_path)
            self.vector_db_default.insert_into_vectordb(embedding,audio_path,meta)
        else:
            embedding = self.audio_class.create_custom_model_audio_embedding(audio_path,model_path)
            self.vector_db_custom.insert_into_vectordb(embedding[0],audio_path,meta)

    def mount_audio_batch(self, audio_dir: str, meta: dict = None, model_path: str = None) -> None:

        """
        Embeds and mounts all audio files from a directory.

        Args:
            audio_dir (str): Directory of audio files.
            meta (dict, optional): Metadata for each file.
            model_path (str, optional): Path to custom model.
        """

        for root, _, files in os.walk(audio_dir):
            for file in files:
                file_path = os.path.join(root, file)
                meta = {
                    "file_name": file
                   }
                try:
                    self.mount_audio(file_path, meta, model_path)
                except Exception as e:
                    logger.warning(f"Failed to mount audio '{file_path}': {e}")
    
    def remove_audio(self, audio_path: str, default: bool = True) -> None:

        """
        Removes an audio entry from the vector DB.

        Args:
            audio_path (str): Identifier for the audio.
            default (bool): If True, use default model DB; else custom.
        """

        if default:
            self.vector_db_default.delete_from_vectordb(audio_path)
        else:
            self.vector_db_custom.delete_from_vectordb(audio_path)

    def create_audio_model(self,directory_x: str, directory_y: str, epochs: int = 10, patience :int = 3) -> None:

        """
        Fine-tunes an audio embedding model on paired datasets.

        Args:
            directory_x (str): Directory with reference audios.
            directory_y (str): Directory with target audios.
            epochs (int): Number of training epochs.
            patience (int): Early stopping patience.
        """

        self.audio_class.fine_tune_custom_audio_model(directory_x,directory_y,epochs,patience)
       
    def find_similar_audio(self, audio_path: str, k: int = 3, meta_filter: dict = None, model_path: str = None) ->  dict[str]:

        """
        Finds top-K similar audios from the vector DB.

        Args:
            audio_path (str): Path to query audio.
            k (int): Number of similar results to return.
            meta_filter (dict, optional): Metadata filters.
            model_path (str, optional): Use custom model if provided.

        Returns:
            dict[str]: Similar audio entries.
        """

        if model_path is None:
            embedding = self.audio_class.create_embedding(audio_path)
            similar_k = self.vector_db_default.retrieve_top_n_from_vectordb(embedding,meta_filter,k)
            return similar_k 
        else:
            embedding = self.audio_class.create_custom_model_audio_embedding(audio_path,model_path)
            similar_k = self.vector_db_custom.retrieve_top_n_from_vectordb(embedding[0],meta_filter,k)
            return similar_k 

    

class mono_image:

    """
    Manages image embeddings, storage in vector DB, fine-tuning, and retrieval.
    
    Args:
        client: Chroma DB client instance.
        collection_name (str): Name of the vector database collection.
    """

    def __init__(self,client,collection_name):
        self.client = client
        self.collection_name = collection_name
        self.vector_db_default = Chroma_DB(client, f"{collection_name}_image_default")
        self.vector_db_custom = Chroma_DB(client, f"{collection_name}_image_custom")
        self.image_class = Image_Embedding()
    
    
    def mount_image(self, image_path: str, meta: dict = None, model_path: str = None) -> None:

        """
        Embeds and mounts a single image.

        Args:
            image_path (str): Path to image file.
            meta (dict, optional): Metadata.
            model_path (str, optional): Custom model for embedding.
        """

        if model_path is None:
            embedding = self.image_class.create_embedding(image_path)
            self.vector_db_default.insert_into_vectordb(embedding,image_path,meta)
        else:
            embedding = self.image_class.create_custom_model_image_embedding(image_path,model_path)
            self.vector_db_custom.insert_into_vectordb(embedding,image_path,meta)

    def mount_image_batch(self, image_dir: str, meta: dict = None, model_path: str = None) -> None:
        """
        Mounts all images in a directory.

        Args:
            image_dir (str): Directory of images.
            meta (dict, optional): Metadata.
            model_path (str, optional): Custom model.
        """
        for root, _, files in os.walk(image_dir):
            for file in files:
                file_path = os.path.join(root, file)
                meta = {
                    "file_name": file
                   }
                try:
                    self.mount_image(file_path, meta=meta, model_path=model_path)
                except Exception as e:
                    logger.warning(f"Failed to mount image '{file_path}': {e}")
    
  
    def create_image_model(self, directory_x: str, directory_y: str, epochs: int = 10, patience: int = 3, learning_rate: float = 0.001) -> None:
        """
        Fine-tunes an image embedding model.

        Args:
            directory_x (str): Input image directory.
            directory_y (str): Target image directory.
            epochs (int): Training epochs.
            patience (int): Early stopping.
            learning_rate (float): Learning rate.
        """
        self.image_class.fine_tune_custom_image_model(directory_x,directory_y,epochs,patience,learning_rate)

    def remove_image(self, img_path: str,default: bool = True) -> None:
        """
        Removes image from vector DB.

        Args:
            img_path (str): Image ID.
            default (bool): Use default model if True.
        """
        if default:
            self.vector_db_default.delete_from_vectordb(img_path)
        else:
            self.vector_db_custom.delete_from_vectordb(img_path)
    
    def find_similar_image(self, img_path: str, k: int = 3, meta_filter: dict = None, model_path: str = None) -> dict[str]:
        """
        Finds similar images.

        Args:
            img_path (str): Query image.
            k (int): Top-K results.
            meta_filter (dict, optional): Metadata filter.
            model_path (str, optional): Custom model.

        Returns:
            dict[str]: Similar image entries.
        """
        if model_path is None:
            embedding = self.image_class.create_embedding(img_path)
            similar_k = self.vector_db_default.retrieve_top_n_from_vectordb(embedding,meta_filter,k)
            return similar_k 
        else:
            embedding = self.image_class.create_custom_model_image_embedding(img_path,model_path)
            similar_k = self.vector_db_custom.retrieve_top_n_from_vectordb(embedding,meta_filter,k)
            return similar_k 


class mono_document:
    """
    Manages document embeddings using text chunking, storage, and similarity search.
    
    Args:
        client: Chroma DB client instance.
        collection_name (str): Name of the vector database collection.
    """
    def __init__(self,client,collection_name):
        self.vector_db_default = Chroma_DB(client, f"{collection_name}_document_default")
        self.doc_class = Document_Embedding()

    def text_splitter(self, text: str, capacity: tuple[int,int] = (150,200), overlap: int = 20, trim: bool = True) -> list[str]:
        """
        Splits text into chunks.

        Args:
            text (str): Full text.
            capacity (tuple[int, int]): Min/max chunk size.
            overlap (int): Overlap tokens.
            trim (bool): Trim whitespace.

        Returns:
            list[str]: Text chunks.
        """
        return self.doc_class.split_text_to_chunks(text,capacity,overlap,trim)

    def mount_document(self, chunk: str, item_id: str, metadata: dict=None) -> None:

        """
        Mounts a document chunk into vector DB.

        Args:
            chunk (str): Text chunk.
            item_id (str): Unique ID.
            metadata (dict, optional): Extra metadata.
        """

        embedding,document = self.doc_class.create_embedding(chunk)
        self.vector_db_default.insert_into_vectordb(embedding[0],item_id,metadata,document)

    def find_similar_documents(self, query: str, metadata: dict = None, k: int = 3) -> dict[str]:

        """
        Finds documents similar to a text query.

        Args:
            query (str): Search text.
            metadata (dict, optional): Metadata filter.
            k (int): Number of results.

        Returns:
            dict[str]: Top-K similar documents.
        """
        query_embedding, _ = self.doc_class.create_embedding(query)
        results = self.vector_db_default.retrieve_top_n_from_vectordb(query_embedding[0],metadata,k)
        return results 


