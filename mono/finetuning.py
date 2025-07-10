import tensorflow as tf
import numpy as np 
from tensorflow.keras.applications.resnet50 import ResNet50 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import CosineSimilarity # type: ignore
from tensorflow.keras import config # type: ignore

from tensorflow.keras.utils import register_keras_serializable # type: ignore

config.enable_unsafe_deserialization() # Allow deserialization of custom loss functions


# Custom contrastive loss based on cosine similarity
@tf.keras.utils.register_keras_serializable()
def cosine_contrastive_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    emb1, emb2 = tf.split(y_pred, num_or_size_splits=2, axis=1) # Split embeddings into two halves
    cosine_sim = tf.reduce_sum(emb1 * emb2, axis=1)  # Dot product of unit vectors

    # y_true = 1 (same): maximize similarity (minimize 1 - sim)
    # y_true = 0 (different): minimize similarity (minimize sim)
    loss = y_true * (1 - cosine_sim) + (1 - y_true) * tf.maximum(cosine_sim, 0)  # Compute loss based on similarity
    return tf.reduce_mean(loss)



# L2 normalize the embeddings
@tf.keras.utils.register_keras_serializable()
def l2_normalize(x: tf.Tensor) -> tf.Tensor:
    return tf.math.l2_normalize(x, axis=1)



# --------- Audio Embedding Fine-tuning ---------

class Audio_Embedding_Finetuning:
    def __init__(self):
        self._max_duration = 15 # Max duration (in seconds) for input audio
        self._max_sample_rate = 16000 # Sample rate for audio preprocessing
        self._embedding_dims = 128 # Output embedding dimension
    
    # Build the encoder network used for embedding audio
    def create_base_network(self,input_dim: int):
        
        # Convolutional layers to extract features
        inputs = tf.keras.Input(shape=(input_dim, 1))  # Input dimension = 240000
        x = tf.keras.layers.Conv1D(8, kernel_size=9, strides=2, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

        x = tf.keras.layers.Conv1D(16, kernel_size=5, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

        x = tf.keras.layers.Conv1D(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Dense layers + final normalization
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(self._embedding_dims, activation='linear')(x) 
        x = tf.keras.layers.Lambda(
                l2_normalize,
                output_shape=(self._embedding_dims,)
            )(x)
        return tf.keras.Model(inputs, x, name="encoder")

    # def contrastive_loss(self,y_true, y_pred):
    #     margin = 1.0
    #     emb1, emb2 = tf.split(y_pred, num_or_size_splits=2, axis=1)
    #     d = tf.reduce_sum(tf.square(emb1 - emb2), axis=1)
    #     loss = y_true * d + (1 - y_true) * tf.square(tf.maximum(margin - tf.sqrt(d + 1e-9), 0))
    #     return tf.reduce_mean(loss)


    # Train a Siamese model using cosine contrastive loss
    def custom_model_training(self, x1: np.ndarray, x2: np.ndarray, epochs: int, patience: int) -> tf.keras.Model:
        x1 = np.expand_dims(x1, axis=-1)  
        
        x2 = np.expand_dims(x2, axis=-1)
        
        labels = np.zeros((len(x1),), dtype=np.float32) # All pairs are treated as dissimilar (y=0)
        input_dim = self._max_duration * self._max_sample_rate 

        # Define inputs
        input_a = tf.keras.Input(shape=(input_dim,1))
        input_b = tf.keras.Input(shape=(input_dim,1))
    
        base_network = self.create_base_network(input_dim)
        embedding_a = base_network(input_a)
        embedding_b = base_network(input_b)
        
        # Concatenate both embeddings for loss
        merged_output = tf.keras.layers.Concatenate()([embedding_a, embedding_b])
        
        # Define Siamese model
        siamese_model = tf.keras.Model(inputs=[input_a, input_b], outputs=merged_output)
        
        # Early stopping to avoid overfitting
        early_stop = EarlyStopping(
            monitor='loss',
            mode='min', 
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        # Compile and train the model
        siamese_model.compile(optimizer='adam', loss=cosine_contrastive_loss)
        siamese_model.fit([x1, x2], labels, epochs=epochs, callbacks=[early_stop])
        
        # Return only the encoder
        return base_network
         
    


#-------------- Image Embedding Fine-tuning ---------------------



         
class Image_Embedding_Finetuning:
    def __init__(self):
        # Use pretrained ResNet50 model for feature extraction
        self._custom_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
    # Fine-tune the ResNet50 model on custom image embeddings with 'CosineSimilarity' loss
    def custom_model_training(self, x: np.ndarray, y: np.ndarray, epochs: int, patience: int, learning_rate: float) -> None:

        # Early stopping to avoid overfitting
        early_stop = EarlyStopping(
            monitor='cosine_similarity',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        # Unfreeze last two layers for fine tuning
        for layer in self._custom_model.layers[-2:]:
            layer.trainable = True

        # Compile and train the model
        self._custom_model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=CosineSimilarity(axis=-1),
                metrics=['cosine_similarity']
        )
        self._custom_model.fit(
            x,y,
            epochs=epochs,
            callbacks=[early_stop],
            
        )

     
    
