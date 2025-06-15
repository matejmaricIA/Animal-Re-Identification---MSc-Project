from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from constants import *
import pandas as pd
from geometric_verification import load_keypoints, compute_geometric_similarity
import time
from tqdm import tqdm

def classify_test_images_with_geometric_verification(
    test_fisher_vectors, train_fisher_vectors, 
    test_keypoints, train_keypoints,
    test_descriptors, train_descriptors,
    train_labels, top_n=5, geometric_candidates=20):
    """Efficient geometric verification with two-stage filtering"""
    
    predictions = {}
    train_vectors = np.stack(list(train_fisher_vectors.values()))
    train_image_ids = list(train_fisher_vectors.keys())
    train_class_labels = np.array([train_labels[image_id] for image_id in train_image_ids])
    
    total_test_images = len(test_fisher_vectors)
    print(f"\n=== Starting Geometric Verification ===")
    print(f"Total test images: {total_test_images}")
    print(f"Total training images: {len(train_fisher_vectors)}")
    print(f"Geometric candidates per query: {geometric_candidates}")
    print(f"=" * 50)
    
    # Track overall statistics
    total_geometric_verifications = 0
    total_inliers_found = 0
    images_with_keypoints = 0
    images_without_keypoints = 0
    
    start_time = time.time()
    
    # Use tqdm for progress bar
    for i, (test_image_id, test_fisher_vector) in enumerate(tqdm(
        test_fisher_vectors.items(), 
        desc="Processing test images", 
        total=total_test_images,
        unit="img"
    )):
        
        print(f"\n[{i+1}/{total_test_images}] Processing: {test_image_id}")
        
        # Stage 1: Fisher Vector similarity (fast)
        stage1_start = time.time()
        test_fisher_vector = test_fisher_vector / np.linalg.norm(test_fisher_vector)
        train_vectors_normalized = train_vectors / np.linalg.norm(train_vectors, axis=1, keepdims=True)
        similarities = np.dot(train_vectors_normalized, test_fisher_vector)
        
        # Get top candidates based on Fisher similarity
        top_indices = np.argsort(similarities)[::-1][:geometric_candidates]
        stage1_time = time.time() - stage1_start
        
        print(f"  ✓ Stage 1 (Fisher similarity): {stage1_time:.3f}s")
        print(f"    Best Fisher similarity: {similarities[top_indices[0]]:.4f}")
        
        if test_image_id in test_keypoints and test_image_id in test_descriptors:
            images_with_keypoints += 1
            
            # Stage 2: Geometric verification (slow, but only on top candidates)
            stage2_start = time.time()
            query_kp = test_keypoints.get(test_image_id, np.array([]))
            query_desc = test_descriptors.get(test_image_id, np.array([]))
            
            print(f"  → Stage 2 (Geometric verification): {len(query_kp)} query keypoints")
            
            final_scores = []
            successful_verifications = 0
            
            # Only verify geometric consistency for top candidates
            for j, idx in enumerate(top_indices):
                train_image_id = train_image_ids[idx]
                fisher_distance = 1.0 - similarities[idx]
                
                train_kp = train_keypoints.get(train_image_id, np.array([]))
                train_desc = train_descriptors.get(train_image_id, np.array([]))
                
                # Show progress for geometric verification
                if j % 5 == 0 or j == len(top_indices) - 1:
                    print(f"    Verifying candidate {j+1}/{len(top_indices)}: {train_image_id}")
                
                final_distance, n_inliers = compute_geometric_similarity(
                    query_desc, query_kp, train_desc, train_kp, fisher_distance
                )
                
                total_geometric_verifications += 1
                if n_inliers > 0:
                    successful_verifications += 1
                    total_inliers_found += n_inliers
                
                final_scores.append({
                    'distance': final_distance,
                    'fisher_distance': fisher_distance,
                    'n_inliers': n_inliers,
                    'class_label': train_class_labels[idx],
                    'train_image_id': train_image_id
                })
            
            # Sort by final distance and get top matches
            final_scores.sort(key=lambda x: x['distance'])
            top_n_matches = [(1.0 - score['distance'], score['class_label']) for score in final_scores[:top_n]]
            predicted_class = final_scores[0]['class_label']
            
            stage2_time = time.time() - stage2_start
            
            print(f"  ✓ Stage 2 completed: {stage2_time:.3f}s")
            print(f"    Successful verifications: {successful_verifications}/{geometric_candidates}")
            print(f"    Best match: {predicted_class} (inliers: {final_scores[0]['n_inliers']}, final_dist: {final_scores[0]['distance']:.4f})")
            
        else:
            images_without_keypoints += 1
            # Fallback to Fisher Vector only
            top_n_matches = [(similarities[i], train_class_labels[i]) for i in top_indices[:top_n]]
            predicted_class = top_n_matches[0][1]
            
            print(f"  ⚠ No keypoints available - using Fisher Vector only")
            print(f"    Predicted class: {predicted_class}")
        
        # Save the prediction and top-N matches
        predictions[test_image_id] = {
            'predicted_class': predicted_class,
            'top_n': top_n_matches
        }
        
        # Show running statistics every 10 images
        if (i + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_image = elapsed_time / (i + 1)
            estimated_remaining = avg_time_per_image * (total_test_images - i - 1)
            
            print(f"\n--- Progress Update ---")
            print(f"Processed: {i+1}/{total_test_images} images")
            print(f"Average time per image: {avg_time_per_image:.2f}s")
            print(f"Estimated remaining time: {estimated_remaining/60:.1f} minutes")
            print(f"Images with keypoints: {images_with_keypoints}")
            print(f"Images without keypoints: {images_without_keypoints}")
            if total_geometric_verifications > 0:
                print(f"Average inliers per verification: {total_inliers_found/total_geometric_verifications:.1f}")
            print("-" * 22)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n=== Geometric Verification Complete ===")
    print(f"Total processing time: {total_time/60:.2f} minutes")
    print(f"Average time per image: {total_time/total_test_images:.2f}s")
    print(f"Images processed with keypoints: {images_with_keypoints}/{total_test_images}")
    print(f"Images processed without keypoints: {images_without_keypoints}/{total_test_images}")
    print(f"Total geometric verifications performed: {total_geometric_verifications}")
    if total_geometric_verifications > 0:
        print(f"Average inliers found: {total_inliers_found/total_geometric_verifications:.2f}")
        print(f"Success rate (verifications with inliers): {(total_geometric_verifications - total_inliers_found == 0)/total_geometric_verifications*100:.1f}%")
    print(f"=" * 40)
    
    return predictions



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
        #print(top_n_matches)

        # Predicted class is the class of the most similar train image
        predicted_class = top_n_matches[0][1]

        # Save the prediction and top-N matches
        predictions[test_image_id] = {
            'predicted_class': predicted_class,
            'top_n': top_n_matches
        }

    return predictions

def predict(pred_fisher_vectors, db_fisher_vectors, dataset_name, threshold = 0.4):
    df = pd.read_csv(DATAFRAME_PATH.format(dataset_name))
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
        