import optuna
from sklearn.metrics import accuracy_score
from feature_aggregation import train_pca, train_gmm, compute_fisher_vectors, stack_all_descriptors
from predict import classify_test_images_with_geometric_verification, evaluate_predictions

def objective(trial):
    # Suggest hyperparameters
    n_pca = trial.suggest_int('n_pca', 32, 128, step=8)
    n_gmm = trial.suggest_int('n_gmm', 16, 512, step=16)
    min_inliers = trial.suggest_int('min_inliers', 4, 16, step=2)
    alpha = trial.suggest_float('alpha', 0.1, 0.9, step=0.05)  # weight between Fisher and geometric verification
    geometric_candidates = trial.suggest_int('geometric_candidates', 5, 50, step = 5)
    inlier_threshold = trial.suggest_float('inlier_threshold', 0.1, 0.9, step = 0.1)

    method = 'disk'

    train_dict = load_descriptors(f"{base_dir}/feature_descriptors_train_{method}/descriptors.h5")
    test_dict = load_descriptors(f"{base_dir}/feature_descriptors_test_{method}/descriptors.h5")
    
    train_keypoints = load_keypoints(f"{base_dir}/feature_descriptors_train_{method}/keypoints.h5")
    test_keypoints = load_keypoints(f"{base_dir}/feature_descriptors_test_{method}/keypoints.h5")
    
    stacked_descriptors_train = stack_all_descriptors(train_dict, max_samples = MAX_GMM_DESCRIPTORS)
    desc_te = stack_all_descriptors(test_dict, max_samples = MAX_GMM_DESCRIPTORS)

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
        inlier_threshold = inlier_threshold

    )
    #test_fisher_vectors, train_fisher_vectors, 
    #test_keypoints, train_keypoints,
    #test_descriptors, train_descriptors,
    #train_labels, top_n=5, geometric_candidates=GEOMETRIC_CANDIDATES, use_lightglue=False, method = 'disk', alpha = ALPHA, min_inliers = MIN_INLIERS, inlier_threshold = inlier_threshold

    # Evaluate predictions
    accuracy = evaluate_predictions(predictions, true_labels_test)

    return accuracy  # Optuna maximizes by default

if __name__ == '__main__':
    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials = 100)

    print("Best parameters: ", study.best_params)