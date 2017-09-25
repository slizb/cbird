
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
from keras.models import Model, load_model
from functools import reduce


def load_encoder():
    """Loads the pretrained encoder model."""
    autoencoder = load_model('autoencoder.h5')
    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer('encoder').output)
    return encoder


def compute_image_embeddings(encoder, arrays):
    """Computes embeddings for a batch of image arrays using the given encoder model."""
    db_vecs = encoder.predict(arrays)
    return db_vecs


def show_n_closest_images(query_img, n, encoder, historical_embeddings, historical_images):
    """Retrieves the most similar n images to a query image, and displays them."""
    ir = _ImageRetriever(encoder, historical_embeddings, historical_images)
    ir.retrieve_closest_images(query_img, n)


class _ImageRetriever:

    def __init__(self, encoder, historical_embeddings, historical_images):
        self.encoder = encoder
        self.historical_embeddings = historical_embeddings
        self.historical_images = historical_images

    def retrieve_closest_images(self, query_img, n_samples=10):
        db_vecs = self.historical_embeddings
        flat_tail_size = self.multiply_all(db_vecs.shape[1:])
        db_vecs = db_vecs.reshape(db_vecs.shape[0], flat_tail_size)

        new_vec = self.encoder.predict(np.array([query_img]))
        new_vec = new_vec.flatten()

        distances, learned_code_index = self.compute_distance(db_vecs, new_vec)
        kept_indexes = self.extract_closest_indexes(distances, learned_code_index, n_samples)
        self.plot_imgs(kept_indexes, n_samples)

    def plot_imgs(self, kept_indexes, n_samples):
        retrieved_images = self.historical_images[int(kept_indexes[0]), :]
        for i in range(1, n_samples):
            retrieved_images = np.hstack((retrieved_images,
                                          self.historical_images[int(kept_indexes[i]), :]))
        retrieved_images = retrieved_images.reshape(28, 28 * n_samples)
        plt.imshow(retrieved_images)
        plt.show()

    @staticmethod
    def extract_closest_indexes(distances, learned_code_index, n_samples):
        df = pd.DataFrame(distances, learned_code_index, ['distances'])
        df = df.sort_values(by='distances')
        n_closest_ix = df.head(n_samples).index
        return n_closest_ix

    @staticmethod
    def compute_distance(db_vecs, new_vec):
        distances = np.linalg.norm(db_vecs - new_vec, axis=1)
        nb_elements = db_vecs.shape[0]
        learned_code_index = np.arange(nb_elements)
        return distances, learned_code_index

    @staticmethod
    def multiply_all(x):
        product = reduce(operator.mul, x, 1)
        return product