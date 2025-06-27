import optuna
from sklearn.metrics import accuracy_score
from feature_aggregation import train_pca, train_gmm, compute_fisher_vectors
from predict import classify_test_images_with_geometric_verification, evaluate_predictions

def objective(trial):
    # Suggest hyperparameters
    n_pca = trial.suggest_int('n_pca', 32, 128, step=8)
    n_gmm = trial.suggest_int('n_gmm', 16, 512, step=16)
    min_inliers = trial.suggest_int('min_inliers', 4, 16, step=2)
    alpha = trial.suggest_float('alpha', 0.1, 0.9, step=0.05)  # weight between Fisher and geometric verification
    geometric_candidates = trial.suggest_int('geometric_candidates', 5, 50, step = 5)
    inlier_threshold = trial.suggest_float('inlier_threshold', 0.1, 0.9, step = 0.1)

    # PCA and GMM training
    pca_model = train_pca(stacked_descriptors_train, n_components=n_pca)
    reduced_descriptors_train = pca_model.transform(stacked_descriptors_train)
    gmm_model = train_gmm(reduced_descriptors_train, n_components=n_gmm)

    fisher_vectors_train = compute_fisher_vectors(train_dict, pca_model, gmm_model)
    fisher_vectors_test = compute_fisher_vectors(test_dict, pca_model, gmm_model)

    # Prediction with GV
    predictions = classify_test_images_with_geometric_verification(
        fisher_vectors_test, fisher_vectors_train,
        test_keypoints, train_keypoints,
        test_descriptors, train_descriptors,
        train_labels,
        geometric_candidates=geometric_candidates,
        use_lightglue=True,
        alpha = alpha,
        min_inliers = min_inliers,
        
    )

    # Evaluate predictions
    accuracy = evaluate_predictions(predictions, true_labels_test)

    return accuracy  # Optuna maximizes by default
