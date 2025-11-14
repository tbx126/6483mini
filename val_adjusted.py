# evaluate_val.py

import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from cnnClassifier import CatDogCNN, IMG_SIZE, NUM_CLASSES


def load_val_images(val_folder, class_names):
    """
    Load images and their true labels from validation folder
    
    Args:
        val_folder: Validation set root directory (dataset/val)
        class_names: List of class names ['cat', 'dog']
    
    Returns:
        images: List of image paths
        labels: List of true labels
    """
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(val_folder, class_name)
        
        if not os.path.exists(class_folder):
            print(f'⚠️  Warning: Folder not found {class_folder}')
            continue
        
        # Get all images in this class folder
        image_files = [f for f in os.listdir(class_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f'Found {len(image_files)} images for class "{class_name}"')
        
        for img_file in image_files:
            img_path = os.path.join(class_folder, img_file)
            images.append(img_path)
            labels.append(class_idx)
    
    return images, labels


def predict_images(model, image_paths, device):
    """
    Batch prediction on image list
    
    Args:
        model: Trained model
        image_paths: List of image paths
        device: Computing device
    
    Returns:
        predictions: List of predicted labels
        probabilities: List of prediction probabilities
    """
    model.eval()
    
    # Define image preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    predictions = []
    probabilities = []
    
    print('\nRunning predictions...')
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc='Predicting', unit='img'):
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Predict
                output = model(image_tensor)
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_class].item()
                
                predictions.append(pred_class)
                probabilities.append(confidence)
                
            except Exception as e:
                print(f'\n⚠️  Error processing image {img_path}: {e}')
                predictions.append(-1)  # Error marker
                probabilities.append(0.0)
    
    return predictions, probabilities


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(10, 8))
    
    # Use seaborn to plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentage in each cell
    total = np.sum(cm)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n✅ Confusion matrix saved to: {save_path}')
    plt.close()


def visualize_misclassified_images(image_paths, true_labels, predictions, 
                                   class_names, images_per_grid=16):
    """
    Visualize misclassified images in grid format
    
    Args:
        image_paths: List of image paths
        true_labels: List of true labels (indices)
        predictions: List of predicted labels (indices)
        class_names: List of class names
        images_per_grid: Number of images per grid (default: 16)
    """
    # Find misclassified images
    misclassified_indices = [i for i in range(len(predictions)) 
                            if predictions[i] != true_labels[i] and predictions[i] != -1]
    
    if len(misclassified_indices) == 0:
        print('\n✅ No misclassified images found!')
        return
    
    print(f'\nFound {len(misclassified_indices)} misclassified images')
    print(f'Generating visualization grids...')
    
    # Calculate number of grids needed
    num_grids = (len(misclassified_indices) + images_per_grid - 1) // images_per_grid
    
    for grid_idx in range(num_grids):
        start_idx = grid_idx * images_per_grid
        end_idx = min(start_idx + images_per_grid, len(misclassified_indices))
        grid_indices = misclassified_indices[start_idx:end_idx]
        
        # Create grid
        grid_size = int(np.ceil(np.sqrt(images_per_grid)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        axes = axes.flatten()
        
        for idx, ax in enumerate(axes):
            if idx < len(grid_indices):
                img_idx = grid_indices[idx]
                img_path = image_paths[img_idx]
                true_label = class_names[true_labels[img_idx]]
                pred_label = class_names[predictions[img_idx]]
                
                try:
                    # Load image and resize to 256x256
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((256, 256), Image.LANCZOS)
                    ax.imshow(img)
                    
                    # Set title with true and predicted labels
                    title = f'True: {true_label}\nPred: {pred_label}'
                    ax.set_title(title, fontsize=12, pad=10)
                    
                    # Color code: red for wrong prediction
                    ax.title.set_color('red')
                    ax.title.set_weight('bold')
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error loading\n{os.path.basename(img_path)}', 
                           ha='center', va='center', fontsize=10)
            
            ax.axis('off')
        
        # Add overall title
        fig.suptitle(f'Misclassified Images - Grid {grid_idx + 1}/{num_grids}', 
                    fontsize=16, y=0.995, weight='bold')
        
        plt.tight_layout()
        
        # Save grid
        save_path = f'misclassified_grid_{grid_idx + 1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'  ✅ Saved: {save_path} ({len(grid_indices)} images)')
    
    print(f'\n✅ Generated {num_grids} misclassification visualization grid(s)')



def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate detailed performance metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics for each class
    metrics = {}
    
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]  # True Positive
        fp = cm[:, i].sum() - tp  # False Positive
        fn = cm[i, :].sum() - tp  # False Negative
        tn = cm.sum() - tp - fp - fn  # True Negative
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'support': tp + fn
        }
    
    # Calculate overall accuracy
    accuracy = np.trace(cm) / cm.sum()
    
    return cm, metrics, accuracy


def save_detailed_report(metrics, accuracy, cm, class_names, 
                         y_true, y_pred, probabilities, 
                         save_path='validation_report.txt'):
    """
    Save detailed evaluation report
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('Validation Set Evaluation Report\n')
        f.write('=' * 80 + '\n\n')
        
        # Overall accuracy
        f.write(f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n')
        
        # Confusion matrix
        f.write('Confusion Matrix:\n')
        f.write('-' * 50 + '\n')
        f.write(f'{"":15s}' + ''.join([f'{name:>15s}' for name in class_names]) + '\n')
        for i, class_name in enumerate(class_names):
            f.write(f'{class_name:15s}' + ''.join([f'{cm[i,j]:>15d}' for j in range(len(class_names))]) + '\n')
        f.write('\n')
        
        # Detailed metrics for each class
        f.write('Detailed Metrics by Class:\n')
        f.write('=' * 80 + '\n')
        for class_name, metric in metrics.items():
            f.write(f'\n【{class_name.upper()}】\n')
            f.write(f'  Precision:    {metric["precision"]:.4f} ({metric["precision"]*100:.2f}%)\n')
            f.write(f'  Recall:       {metric["recall"]:.4f} ({metric["recall"]*100:.2f}%)\n')
            f.write(f'  F1-Score:     {metric["f1_score"]:.4f} ({metric["f1_score"]*100:.2f}%)\n')
            f.write(f'  Specificity:  {metric["specificity"]:.4f} ({metric["specificity"]*100:.2f}%)\n')
            f.write(f'  Support:      {metric["support"]}\n')
        
        # Macro and weighted averages
        f.write('\n' + '=' * 80 + '\n')
        f.write('Average Metrics:\n')
        f.write('-' * 50 + '\n')
        
        # Macro Average
        macro_precision = np.mean([m['precision'] for m in metrics.values()])
        macro_recall = np.mean([m['recall'] for m in metrics.values()])
        macro_f1 = np.mean([m['f1_score'] for m in metrics.values()])
        
        f.write(f'Macro Average:\n')
        f.write(f'  Precision: {macro_precision:.4f}\n')
        f.write(f'  Recall:    {macro_recall:.4f}\n')
        f.write(f'  F1-Score:  {macro_f1:.4f}\n\n')
        
        # Weighted Average
        total_samples = sum([m['support'] for m in metrics.values()])
        weighted_precision = sum([m['precision'] * m['support'] for m in metrics.values()]) / total_samples
        weighted_recall = sum([m['recall'] * m['support'] for m in metrics.values()]) / total_samples
        weighted_f1 = sum([m['f1_score'] * m['support'] for m in metrics.values()]) / total_samples
        
        f.write(f'Weighted Average:\n')
        f.write(f'  Precision: {weighted_precision:.4f}\n')
        f.write(f'  Recall:    {weighted_recall:.4f}\n')
        f.write(f'  F1-Score:  {weighted_f1:.4f}\n\n')
        
        # Confidence statistics
        f.write('=' * 80 + '\n')
        f.write('Prediction Confidence Statistics:\n')
        f.write('-' * 50 + '\n')
        f.write(f'Mean Confidence: {np.mean(probabilities):.4f} ({np.mean(probabilities)*100:.2f}%)\n')
        f.write(f'Max Confidence:  {np.max(probabilities):.4f} ({np.max(probabilities)*100:.2f}%)\n')
        f.write(f'Min Confidence:  {np.min(probabilities):.4f} ({np.min(probabilities)*100:.2f}%)\n')
        f.write(f'Std Deviation:   {np.std(probabilities):.4f}\n\n')
        
        # Sklearn classification report
        f.write('=' * 80 + '\n')
        f.write('Sklearn Classification Report:\n')
        f.write('-' * 50 + '\n')
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    print(f'\n✅ Detailed report saved to: {save_path}')


def evaluate_validation_set():
    """
    Main function: Evaluate validation set
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Check model file
    if not os.path.exists('best_model.pth'):
        print('\n❌ Error: Model file "best_model.pth" not found')
        return
    
    # Load model
    print('\nLoading model...')
    model = CatDogCNN(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'✅ Model loaded successfully (Epoch {checkpoint["epoch"]+1}, Val Acc: {checkpoint["val_acc"]:.2f}%)')
    
    # Set classes and paths
    class_names = ['cat', 'dog']
    val_folder = 'dataset/val'
    
    # Check validation folder
    if not os.path.exists(val_folder):
        print(f'\n❌ Error: Validation folder not found: {val_folder}')
        return
    
    # Load validation images
    print(f'\nLoading validation images...')
    print(f'Validation path: {val_folder}')
    image_paths, true_labels = load_val_images(val_folder, class_names)
    
    if len(image_paths) == 0:
        print('\n❌ Error: No images found in validation set')
        return
    
    print(f'\nTotal images loaded: {len(image_paths)}')
    
    # Make predictions
    predictions, probabilities = predict_images(model, image_paths, device)
    
    # Calculate metrics
    print('\nCalculating evaluation metrics...')
    cm, metrics, accuracy = calculate_metrics(true_labels, predictions, class_names)
    
    # Print results to console
    print('\n' + '=' * 80)
    print('Evaluation Results:')
    print('=' * 80)
    print(f'\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'\nConfusion Matrix:')
    print(cm)
    print(f'\nMetrics by Class:')
    for class_name, metric in metrics.items():
        print(f'\n{class_name}:')
        print(f'  Precision: {metric["precision"]:.4f}')
        print(f'  Recall:    {metric["recall"]:.4f}')
        print(f'  F1-Score:  {metric["f1_score"]:.4f}')
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    
    # Save detailed report
    save_detailed_report(metrics, accuracy, cm, class_names, 
                        true_labels, predictions, probabilities)
    
    # Visualize misclassified images
    visualize_misclassified_images(image_paths, true_labels, predictions, 
                                   class_names, images_per_grid=16)
    
    print('\n' + '=' * 80)
    print('✅ Evaluation completed!')
    print('=' * 80)


if __name__ == '__main__':
    evaluate_validation_set()