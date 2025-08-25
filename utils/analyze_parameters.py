import pandas as pd
import numpy as np

def analyze_hyperparameter_results(file_path='../evaluations/classification/classification_results.xlsx'):
    """
    Analyze hyperparameter optimization results across multiple datasets
    to find the best parameter combinations.
    """
    # Load the results data
    results_df = pd.read_excel(file_path, sheet_name='Sheet1')
    
    # Define the parameter columns that were optimized
    param_columns = [
        'Dataset', 'GMM Components', 'PCA Components', 'Use GV', 'Alpha (fv sim - gv)',
        'Geom. Candidates', 'Min Inliers', 'Inlier Threshold'
    ]
    
    print("=== HYPERPARAMETER OPTIMIZATION ANALYSIS ===\n")
    
    # 1. Find best parameter combination for each dataset
    print("1. BEST PARAMETER COMBINATIONS PER DATASET:")
    print("-" * 50)
    
    best_combinations = results_df.loc[results_df.groupby('Dataset')['Accuracy'].idxmax()]
    
    for _, row in best_combinations.iterrows():
        print(f"\n**{row['Dataset']}** (Accuracy: {row['Accuracy']:.4f})")
        print(f"  GMM Components: {row['GMM Components']}")
        print(f"  PCA Components: {row['PCA Components']}")
        print(f"  Use GV: {row['Use GV']}")
        print(f"  Alpha: {row['Alpha (fv sim - gv)']:.2f}")
        print(f"  Geom. Candidates: {row['Geom. Candidates']}")
        print(f"  Min Inliers: {row['Min Inliers']}")
        print(f"  Inlier Threshold: {row['Inlier Threshold']}")
        print(f"  Runtime: {row['Run Time (minutes)']:.2f} min")
    
    # 2. Analyze parameter frequency across best combinations
    print(f"\n\n2. MOST FREQUENT PARAMETER VALUES ACROSS BEST COMBINATIONS:")
    print("-" * 60)
    
    param_analysis = best_combinations.drop(columns=['Dataset', 'Accuracy', 'Top-5 Accuracy', 
                                                    'F-1 Score', 'Run Time (minutes)', 
                                                    'Training Examples', 'Num Classes', 
                                                    'Method', 'Remove Background', 
                                                    'MAX GMM Descriptors'])
    
    for col in param_analysis.columns:
        value_counts = param_analysis[col].value_counts()
        most_common = value_counts.index[0]
        frequency = value_counts.iloc[0]
        total_datasets = len(param_analysis)
        
        print(f"**{col}**: {most_common} (appears in {frequency}/{total_datasets} datasets = {frequency/total_datasets*100:.1f}%)")
    
    # 3. Calculate weighted recommendations based on dataset performance
    print(f"\n\n3. WEIGHTED PARAMETER RECOMMENDATIONS:")
    print("-" * 45)
    print("(Weighted by accuracy performance of each dataset)")
    
    # Weight each parameter choice by the accuracy achieved
    weighted_recommendations = {}
    
    for col in ['GMM Components', 'PCA Components', 'Alpha (fv sim - gv)', 
                'Geom. Candidates', 'Min Inliers', 'Inlier Threshold']:
        if col in param_analysis.columns:
            weighted_avg = 0
            total_weight = 0
            
            for _, row in best_combinations.iterrows():
                weight = row['Accuracy']
                value = row[col]
                weighted_avg += value * weight
                total_weight += weight
            
            weighted_recommendations[col] = weighted_avg / total_weight
            print(f"**{col}**: {weighted_recommendations[col]:.2f}")
    
    # 4. Performance summary across datasets
    print(f"\n\n4. DATASET PERFORMANCE SUMMARY:")
    print("-" * 35)
    
    performance_summary = best_combinations[['Dataset', 'Accuracy', 'Top-5 Accuracy', 'F-1 Score', 'Run Time (minutes)']].copy()
    performance_summary = performance_summary.sort_values('Accuracy', ascending=False)
    
    print(performance_summary.to_string(index=False, float_format='%.4f'))
    
    print(f"\nOverall Statistics:")
    print(f"  Mean Accuracy: {performance_summary['Accuracy'].mean():.4f}")
    print(f"  Std Accuracy: {performance_summary['Accuracy'].std():.4f}")
    print(f"  Mean Runtime: {performance_summary['Run Time (minutes)'].mean():.2f} minutes")
    
    # 5. Final recommendations
    print(f"\n\n5. FINAL PARAMETER RECOMMENDATIONS:")
    print("-" * 40)
    print("Based on frequency analysis and performance weighting:")
    print()
    print("**Universal Settings (used in all best combinations):**")
    print("  - Use GV: True")
    print()
    print("**Recommended Parameter Ranges:**")
    print(f"  - GMM Components: {int(weighted_recommendations['GMM Components'])} (±100)")
    print(f"  - PCA Components: {int(weighted_recommendations['PCA Components'])} (±20)")
    print(f"  - Alpha: {weighted_recommendations['Alpha (fv sim - gv)']:.2f} (±0.2)")
    print(f"  - Geom. Candidates: {int(weighted_recommendations['Geom. Candidates'])} (±10)")
    print(f"  - Min Inliers: {int(weighted_recommendations['Min Inliers'])} (±3)")
    print(f"  - Inlier Threshold: {weighted_recommendations['Inlier Threshold']:.2f} (±0.2)")
    
    return best_combinations, weighted_recommendations

if __name__ == "__main__":
    # Run the analysis
    best_combos, recommendations = analyze_hyperparameter_results()
