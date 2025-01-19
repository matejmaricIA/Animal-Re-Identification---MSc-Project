from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from constants import *
import pandas as pd

def classify_test_images(test_fisher_vectors, train_fisher_vectors, train_labels, top_n = 5):

    predictions = {}

    # Stack train Fisher Vectors and labels for efficient comparison
    train_vectors = np.stack(list(train_fisher_vectors.values()))
    train_image_ids = list(train_fisher_vectors.keys())
    train_class_labels = np.array([train_labels[image_id] for image_id in train_image_ids])

    for test_image_id, test_fisher_vector in test_fisher_vectors.items():
        # Normalize Fisher Vectors for cosine similarity
        test_fisher_vector = test_fisher_vector / np.linalg.norm(test_fisher_vector)
        train_vectors_normalized = train_vectors / np.linalg.norm(train_vectors, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(train_vectors_normalized, test_fisher_vector)

        # Sort similarities in descending order
        sorted_indices = np.argsort(similarities)[::-1]
        top_indices = sorted_indices[:top_n]

        # Get top-N class labels and similarities
        top_n_matches = [(similarities[i], train_class_labels[i]) for i in top_indices]
        print(top_n_matches)

        # Predicted class is the class of the most similar train image
        predicted_class = top_n_matches[0][1]

        # Save the prediction and top-N matches
        predictions[test_image_id] = {
            'predicted_class': predicted_class,
            'top_n': top_n_matches
        }

    return predictions

def predict(pred_fisher_vectors, db_fisher_vectors, threshold = 0.4):
    df = pd.read_csv(DATAFRAME_PATH)
    class_labels = dict(zip(df['image_id'], df['identity']))
    db_vectors = np.stack(list(db_fisher_vectors.values()))
    train_image_ids = list(db_fisher_vectors.keys())
    db_class_labels = np.array([class_labels[image_id] for image_id in train_image_ids])

    for test_image_id, test_fisher_vector in pred_fisher_vectors.items():
        test_fisher_vector = test_fisher_vector / np.linalg.norm(test_fisher_vector)
        train_vectors_normalized = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)

        similarities = np.dot(train_vectors_normalized, test_fisher_vector)
        sorted_indices = np.argsort(similarities)[::-1]
        #print(max(similarities))
        top_indices = sorted_indices[:1]
        top_match = [(similarities[i], db_class_labels[i]) for i in top_indices][0]

        print(f"Top match is {top_match[1]} with similarity score of: {top_match[0]}")

        if top_match[1] >= threshold:
            print(f"Determine class is: {db_class_labels[top_indices]}")
        else:
            print("Unknown class.")