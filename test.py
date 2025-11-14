# test_with_predict.py

import torch
import os
import re
import csv
from tqdm import tqdm
from cnnClassifier import (
    CatDogCNN,
    predict_test_folder,
    IMG_SIZE,
    NUM_CLASSES
)

def natural_sort_key(filename):
    """
    Split filename into strings and numbers for natural sorting
    Example: 'cat_10.jpg' -> ['cat_', 10, '.jpg']
    This ensures correct ordering: 1, 2, 3, ..., 9, 10, 11, ... instead of 1, 10, 100, 101, ...
    """
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', filename)]


def test_model_simple():
    """
    Test model using predict_test_folder function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Check if model file exists
    if not os.path.exists('best_model.pth'):
        print('\n❌ Error: Model file "best_model.pth" not found')
        print('Please run training program first to generate the model file')
        return
    
    # Load model
    model = CatDogCNN(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f'✅ Model loaded successfully (Epoch {checkpoint["epoch"]+1}, Val Acc: {checkpoint["val_acc"]:.2f}%)')
    
    # Set class names (modify according to your dataset)
    class_names = ['cat', 'dog']  # Change to ['dog', 'cat'] if order is wrong
    
    # Set test folder path
    test_folder = 'dataset/test'  # Modify to your test set path
    
    # Check if test folder exists
    if not os.path.exists(test_folder):
        print(f'\n❌ Error: Test folder not found: {test_folder}')
        print('Please ensure test images are in the correct path')
        return
    
    # Use predict_test_folder function for prediction
    print(f'\nTesting on folder: {test_folder}')
    print('=' * 70)
    
    results = predict_test_folder(test_folder, model, device, class_names)
    
    # Statistics
    print('\n' + '=' * 70)
    print('Prediction Summary:')
    print(f'Total images: {len(results)}')
    
    # Count by class
    cat_count = sum(1 for r in results if r['predicted_class'] == 'cat')
    dog_count = sum(1 for r in results if r['predicted_class'] == 'dog')
    
    print(f'Predicted as cat: {cat_count}')
    print(f'Predicted as dog: {dog_count}')
    
    # Calculate average confidence
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    print(f'Average confidence: {avg_confidence:.2f}%')
    
    # Find highest and lowest confidence predictions
    if results:
        max_conf_result = max(results, key=lambda x: x['confidence'])
        min_conf_result = min(results, key=lambda x: x['confidence'])
        
        print(f'\nHighest confidence: {max_conf_result["filename"]} -> '
              f'{max_conf_result["predicted_class"]} ({max_conf_result["confidence"]:.2f}%)')
        print(f'Lowest confidence: {min_conf_result["filename"]} -> '
              f'{min_conf_result["predicted_class"]} ({min_conf_result["confidence"]:.2f}%)')
    
    # Save results to CSV file
    save_results_to_csv(results, 'prediction_results.csv')
    
    return results


def save_results_to_csv(results, filename='prediction_results.csv'):
    """
    Save prediction results to CSV file (sorted by natural filename order)
    Format: Index, Prediction (0 for cat, 1 for dog)
    """
    # Use natural sort instead of regular string sort
    sorted_results = sorted(results, key=lambda x: natural_sort_key(x['filename']))
    
    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        
        # Write predictions (0 for cat, 1 for dog)
        for i, result in enumerate(sorted_results, 1):
            prediction = 0 if result['predicted_class'] == 'cat' else 1
            writer.writerow([i, prediction])
    
    print(f'✅ Results saved to {filename}')


if __name__ == '__main__':
    test_model_simple()