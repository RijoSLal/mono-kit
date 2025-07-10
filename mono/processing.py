import os
import tensorflow_hub as hub
import librosa
import logging
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.utils import load_img, img_to_array # type: ignore
from tensorflow.keras.saving import save_model # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from semantic_text_splitter import TextSplitter
from .finetuning import Audio_Embedding_Finetuning,Image_Embedding_Finetuning,cosine_contrastive_loss
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logger = logging.getLogger(__name__)

#-------------Audio------------#

class Audio_Embedding(Audio_Embedding_Finetuning):

    """
    Handles audio file preprocessing, embedding generation using a pretrained model,
    and fine-tuning a custom audio embedding model
    """

    def __init__(self):
        super().__init__()
        self.__model = hub.load('https://kaggle.com/models/google/vggish/frameworks/TensorFlow2/variations/vggish/versions/1')
        self.__audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        self.__save_dir = "custom_trained_audio_embedding_model"

    def _is_valid_audio_file(self, audio_file_path: str) -> bool:

        """
        Checks if a file has a valid audio extension.

        Args:
            audio_file_path (str): Path to the audio file.

        Returns:
            bool: True if the file has a valid extension, False otherwise.
        """

        _, ext = os.path.splitext(audio_file_path)
        return ext.lower() in self.__audio_extensions    
    
    def _preprocess_audio(self, file: str) -> np.ndarray:

        """
        Loads and processes an audio file to a fixed duration and sample rate.

        Args:
            file (str): Path to the audio file.

        Returns:
            np.ndarray: Preprocessed audio waveform.
        """

        wave, _ = librosa.load(file, sr=self._max_sample_rate, duration=self._max_duration)
        target_length = int(self._max_sample_rate * self._max_duration)
        wave = librosa.util.fix_length(wave, size=target_length)
        return wave
    
    
    def create_dataset(self, dir_x: str , dir_y: str) -> tuple[int,int]:

        """
        Creates audio datasets (X and Y) from reference and target directories.

        Args:
            dir_x (str): Directory for reference audio files.
            dir_y (str): Directory for target audio files.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of preprocessed audio arrays (X, Y).

        Raises:
            FileNotFoundError: If either directory does not exist.
            RuntimeError: If dataset creation fails.
        """

        if not (os.path.isdir(dir_x) and os.path.isdir(dir_y)):
           raise FileNotFoundError(f"Input directories not found: '{dir_x}' or '{dir_y}' does not exist.")
        try:
            files = sorted([
                f for f in os.listdir(dir_x)
                if self._is_valid_audio_file(f) and os.path.exists(os.path.join(dir_y, f))
            ])

            X = np.array([self._preprocess_audio(os.path.join(dir_x, f)) for f in files])
            Y = np.array([self._preprocess_audio(os.path.join(dir_y, f)) for f in files])

            return X, Y 
        except Exception:
            raise RuntimeError(f"Failed to create audio dataset from '{dir_x}' and '{dir_y}'")
    
    def create_embedding(self, audio_path:str = None) -> np.ndarray:

        """
        Creates an embedding for a given audio file using the pretrained VGGish model.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.ndarray: Embedding vector.

        Raises:
            ValueError: If path is None or file is invalid.
            RuntimeError: If embedding fails.
        """

        if audio_path is None:
            logger.error("Cannot create embedding: missing audio file path.")
            raise ValueError("Missing required argument: 'file_path' cannot be None.")
        
        try:
        
            if self._is_valid_audio_file(audio_path):

                wave_form = self._preprocess_audio(audio_path)  # Preprocess the audio file to get the waveform
                
                embedding_tensor = self.__model(wave_form)
                embedding = embedding_tensor.numpy()
                embedding = np.mean(embedding, axis=0)
                return embedding 
            else:
                logger.warning("Audio file supported only.")
                raise ValueError(f"Invalid file: '{audio_path}'")
        except Exception as e:
            logger.exception(f"Failed to create embedding from audio file: '{audio_path}'")
            raise RuntimeError(f"Embedding creation failed for file: '{audio_path}'") from e

        
    def create_custom_model_audio_embedding(self, audio_path: str, model_path: str) -> np.ndarray:

        """
        Creates an embedding using a custom fine-tuned model.

        Args:
            audio_path (str): Path to the audio file.
            model_path (str): Path to the trained model.

        Returns:
            np.ndarray: Embedding vector from the custom model.

        Raises:
            ValueError: If file is missing or invalid.
            RuntimeError: If embedding fails.
        """

        if audio_path is None:
            logger.error("Cannot create custom model embedding: missing audio file path.")
            raise ValueError("Missing required argument: 'audio_path' cannot be None.")
        try:
            if self._is_valid_audio_file(audio_path):
                wave_form = self._preprocess_audio(audio_path)
                model = load_model(model_path,custom_objects={'contrastive_loss': cosine_contrastive_loss})
                wave_form = np.expand_dims(wave_form, axis=(0, -1))
                embedding = model.predict(wave_form)
                return embedding 
            else:
                logger.warning("Audio file supported only.")
                raise ValueError(f"Invalid file: '{audio_path}'")
            
        except Exception as e:
            logger.exception(f"Failed to create custom model embedding from audio file: '{audio_path}'")
            raise RuntimeError(f"Embedding creation failed for file: '{audio_path}'") from e
           
    def fine_tune_custom_audio_model(self, reference_dir: str, target_dir: str, epochs: int, patience: int) -> None:

        """
        Fine-tunes the custom audio model using paired reference and target datasets.

        Args:
            reference_dir (str): Directory of input audio files.
            target_dir (str): Directory of corresponding target audio files.
            epochs (int): Number of training epochs.
            patience (int): Patience for early stopping.

        Raises:
            ValueError: If directories are invalid.
            RuntimeError: If training or saving fails.
        """

        if not os.path.isdir(reference_dir) or not os.path.isdir(target_dir):
            logger.error(f"Invalid input directories: '{reference_dir}', '{target_dir}'")
            raise ValueError(f"Both reference_dir and target_dir must be valid directories.")
        
        try:
            logger.info("Creating audio dataset for fine-tuning.")
            X,Y = self.create_dataset(reference_dir,target_dir)

            logger.info("Starting model training.")
            trained_model = self.custom_model_training(X,Y,epochs,patience)
            os.makedirs(self.__save_dir, exist_ok=True)
            model_path = os.path.join(self.__save_dir, "audio_model.keras")

            logger.info(f"Saving fine-tuned model to: {model_path}")
            save_model(trained_model,model_path)
            
            logger.info("Audio model fine-tuning and saving completed successfully.")
        except Exception as e:
            logger.exception("Failed to fine-tune or save the audio model.")
            raise RuntimeError("Audio model fine-tuning failed.") from e




#-------------Image------------#



class Image_Embedding(Image_Embedding_Finetuning):

    """
    Provides image preprocessing, embedding generation using ResNet50,
    and functionality to fine-tune and save custom image embedding models.
    """

    def __init__(self):
        super().__init__()
        self.__base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.__image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.ppm'}
        self.__save_dir = "custom_trained_image_embedding_model"
        
    def _is_valid_image_file(self, img_file_path: str) -> bool:

        """
        Checks if a file has a valid image extension.

        Args:
            img_file_path (str): Path to the image file.

        Returns:
            bool: True if valid, False otherwise.
        """

        _, ext = os.path.splitext(img_file_path)
        return ext.lower() in self.__image_extensions 

    def image_preprocessing(self, img_path: str, expand_dims: bool = True) -> np.ndarray:

        """
        Loads and preprocesses an image for embedding.

        Args:
            img_path (str): Path to the image file.
            expand_dims (bool): Whether to add batch dimension.

        Returns:
            np.ndarray: Preprocessed image array.

        Raises:
            ValueError: If path is missing or invalid.
        """
        if img_path is None:
            logger.error("Cannot create embedding: missing image file path.")
            raise ValueError("Missing required argument: 'img_path' cannot be None.")
        
        if not self._is_valid_image_file(img_path):
            logger.warning("Image file supported only.")
            raise ValueError(f"Invalid file: '{img_path}'")
        
        img = load_img(img_path, target_size=(224, 224)) 
        img_array = img_to_array(img)

        if expand_dims:
           img_array = np.expand_dims(img_array, axis=0)

        img_array = preprocess_input(img_array) 
        return img_array
    
    def create_dataset(self, dir_x: str, dir_y: str)  -> tuple[np.ndarray, np.ndarray]:

        """
        Creates a dataset of image embeddings for fine-tuning.

        Args:
            dir_x (str): Directory of input images.
            dir_y (str): Directory of target images.

        Returns:
            tuple[np.ndarray, np.ndarray]: Input images and target embeddings.

        Raises:
            FileNotFoundError: If either directory does not exist.
            RuntimeError: If dataset creation fails.
        """
        if not (os.path.isdir(dir_x) and os.path.isdir(dir_y)):
           raise FileNotFoundError(f"Input directories not found: '{dir_x}' or '{dir_y}' does not exist.")
        
        try:
            files = sorted([
                f for f in os.listdir(dir_x)
                if self._is_valid_image_file(f) and os.path.exists(os.path.join(dir_y, f))
            ])
            

            load = lambda p: self.image_preprocessing(p,expand_dims=False)
            X = np.array([load(os.path.join(dir_x, f)) for f in files])
            Y = np.array([load(os.path.join(dir_y, f)) for f in files])
            Y_embeddings = self.__base_model.predict(Y)
            return X,Y_embeddings 
        
        except Exception:
            raise RuntimeError(f"Failed to create image dataset from '{dir_x}' and '{dir_y}'")

    
    def create_embedding(self, img_path: str = None) -> np.ndarray:
        """
        Creates an embedding from a given image using ResNet50.

        Args:
            img_path (str): Path to the image file.

        Returns:
            np.ndarray: Normalized image embedding.

        Raises:
            ValueError: If image path is missing or invalid.
            RuntimeError: If embedding fails.
        """
        if img_path is None:
            logger.error("Cannot create embedding: missing image file path.")
            raise ValueError("Missing required argument: 'img_path' cannot be None.")
        try:
            img_array = self.image_preprocessing(img_path)
            embedding = self.__base_model.predict(img_array)
            embedding = embedding[0] / np.linalg.norm(embedding[0], ord=2)
            return embedding 
        
        except Exception as e:
            logger.exception(f"Failed to create custom model embedding from image file: '{img_path}'")
            raise RuntimeError(f"Embedding creation failed for file: '{img_path}'") from e
    
    def create_custom_model_image_embedding(self, img_path: str, model_path: str) -> np.ndarray:
        """
        Generates image embedding using a fine-tuned model.

        Args:
            img_path (str): Path to image file.
            model_path (str): Path to trained Keras model.

        Returns:
            np.ndarray: Embedding vector from custom model.

        Raises:
            ValueError: If input is invalid.
            RuntimeError: If embedding fails.
        """
        if img_path is None:
            logger.error("Cannot create embedding: missing image file path.")
            raise ValueError("Missing required argument: 'img_path' cannot be None.")
        
        try:
            img_array = self.image_preprocessing(img_path)
            model = load_model(model_path)
            embedding = model.predict(img_array)[0]
            return embedding  
        except Exception as e:
            logger.exception(f"Failed to create custom model embedding from image file: '{img_path}'")
            raise RuntimeError(f"Embedding creation failed for file: '{img_path}'") from e
    
    
    
    def fine_tune_custom_image_model(self, reference_dir: str, target_dir: str, epochs: int,patience: int,learning_rate: float) -> None:
        """
        Fine-tunes a custom image model and saves it.

        Args:
            reference_dir (str): Directory of input images.
            target_dir (str): Directory of target images.
            epochs (int): Number of training epochs.
            patience (int): Early stopping patience.
            learning_rate (float): Learning rate for optimizer.

        Raises:
            ValueError: If input directories are invalid.
            RuntimeError: If training or saving fails.
        """
        if not os.path.isdir(reference_dir) or not os.path.isdir(target_dir):
            logger.error(f"Invalid input directories: '{reference_dir}', '{target_dir}'")
            raise ValueError(f"Both {reference_dir} and {target_dir} must be valid directories.")
        try:
            X,Y = self.create_dataset(reference_dir,target_dir) 
            self.custom_model_training(X,Y,epochs,patience,learning_rate)
            os.makedirs(self.__save_dir, exist_ok=True)
            model_path = os.path.join(self.__save_dir, "image_model.keras")

            logger.info(f"Saving fine-tuned model to: {model_path}")
            save_model(self._custom_model,model_path)

            logger.info("Image model fine-tuning and saving completed successfully.")

        except Exception as e:
            logger.exception("Failed to fine-tune or save the image model.")
            raise RuntimeError("Image model fine-tuning failed.") from e


#-----------------Documents-----------------------------------------------

class Document_Embedding:

    """
    Supports splitting text into semantic chunks and generating embeddings
    using a default embedding function.
    """

    def __init__(self):

        """
        Initializes the default embedding function for text.
        
        Raises:
            RuntimeError: If initialization fails.
        """

        try:
           self.default_ef = DefaultEmbeddingFunction()
        except Exception as e:
            logger.exception("Failed to initialize DefaultEmbeddingFunction.")
            raise RuntimeError("Document embedding initialization failed.") from e

    def split_text_to_chunks(self, text: str, capacity: tuple[int, int], overlap: int, trim: bool) -> list[str]:

        """
        Splits a text string into smaller semantic chunks.

        Args:
            text (str): Input text.
            capacity (tuple[int, int]): Minimum and maximum chunk size.
            overlap (int): Number of overlapping tokens.
            trim (bool): Whether to trim leading/trailing whitespace.

        Returns:
            list[str]: List of text chunks.

        Raises:
            RuntimeError: If splitting fails.
        """
        try:
            splitter = TextSplitter(capacity,overlap,trim)
            chunks = splitter.chunks(text)
            return chunks 
        except Exception as e:
            logger.exception("Failed to split text into chunks.")
            raise RuntimeError("Text splitting failed.") from e
    
    def create_embedding(self, chunk: str) -> tuple[np.ndarray,str]:
    
        """
        Creates an embedding for a given text chunk.

        Args:
            chunk (str): A text snippet.

        Returns:
            tuple[np.ndarray, str]: Embedding and the original chunk.

        Raises:
            RuntimeError: If embedding fails.
        """
        try:
            doc_embedding = self.default_ef([chunk])
            return doc_embedding, chunk
        except Exception as e:
            logger.exception(f"Failed to create embedding for chunk: '{chunk}...'")
            raise RuntimeError("Document embedding generation failed.") from e



