import csv
import sys
from collections import defaultdict

def analyze_csv(file_path):
    results = defaultdict(lambda: defaultdict(list))
    
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['status'] != 'ok':
                    continue
                
                dataset = row['dataset']
                method = row['method']
                pca_dim = int(row['pca_dim'])
                accuracy = float(row['accuracy'])
                
                if method not in results[dataset]:
                    results[dataset][method] = []
                results[dataset][method].append((pca_dim, accuracy))
                
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    print(f"Analysis of {file_path}")
    print("=" * 60)

    for dataset in sorted(results.keys()):
        print(f"\nDataset: {dataset}")
        print("-" * 40)
        
        methods = results[dataset]
        for method in sorted(methods.keys()):
            data = methods[method]
            # Sort by pca_dim
            data.sort(key=lambda x: x[0])
            
            print(f"  Method: {method}")
            best_acc = -1.0
            best_dim = -1
            
            for dim, acc in data:
                print(f"    PCA Dim: {dim:3d} -> Accuracy: {acc:.4f}")
                if acc > best_acc:
                    best_acc = acc
                    best_dim = dim
            
            if best_dim != -1:
                print(f"    => Best PCA Dim for {method}: {best_dim} (Acc: {best_acc:.4f})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_csv(sys.argv[1])
    else:
        print("Please provide a CSV file path.")
