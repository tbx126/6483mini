import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
from tqdm import tqdm
from config import MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG, CLASS_CONFIG

IMG_SIZE = MODEL_CONFIG['img_size']
BATCH_SIZE = MODEL_CONFIG['batch_size']
NUM_EPOCHS = MODEL_CONFIG['num_epochs']
LEARNING_RATE = MODEL_CONFIG['learning_rate']
WEIGHT_DECAY = MODEL_CONFIG['weight_decay']
NUM_CLASSES = MODEL_CONFIG['num_classes']

TRAIN_DIR = DATA_CONFIG['train_dir']
VAL_DIR = DATA_CONFIG['val_dir']
MODEL_SAVE_PATH = DATA_CONFIG['model_save_path']

NUM_WORKERS = TRAINING_CONFIG['num_workers']
PIN_MEMORY = TRAINING_CONFIG['pin_memory']
LR_SCHEDULER_PATIENCE = TRAINING_CONFIG['lr_scheduler_patience']
LR_SCHEDULER_FACTOR = TRAINING_CONFIG['lr_scheduler_factor']

CLASS_NAMES = CLASS_CONFIG['class_names']


class CatDogCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CatDogCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def train_model(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS} [Train]',
                ncols=100, ascii=True)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        current_loss = running_loss / total
        current_acc = 100 * correct / total
        pbar.set_postfix({'loss': f'{current_loss:.4f}',
                          'acc': f'{current_acc:.2f}%'})
    return running_loss / total, 100 * correct / total


def validate_model(model, val_loader, criterion, device, epoch):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS} [Validation]',
                ncols=100, ascii=True)

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            current_loss = running_loss / total
            current_acc = 100 * correct / total
            pbar.set_postfix({'loss': f'{current_loss:.4f}',
                              'acc': f'{current_acc:.2f}%'})
    return running_loss / total, 100 * correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f'Training set size: {len(train_dataset)}')
    print(f'Validation set size: {len(val_dataset)}')
    print(f'Class mapping: {train_dataset.class_to_idx}')

    model = CatDogCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=LR_SCHEDULER_FACTOR,
                                                     patience=LR_SCHEDULER_PATIENCE)

    best_val_acc = 0.0

    print('\nTraining started...')
    print('=' * 60)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate_model(model, val_loader, criterion, device, epoch)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'  Current learning rate: {current_lr:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f'  ✓ Best model saved (Validation accuracy: {val_acc:.2f}%)')

        print('-' * 60)

    print('\nTraining finished!')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')


def predict_image(image_path, model, device, class_names):
    """Predict a single image"""
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score


def predict_test_folder(test_folder, model, device, class_names):
    """Batch predict images in test folder"""
    model.eval()
    results = []

    image_files = [f for f in os.listdir(test_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f'\nStart predicting {len(image_files)} images...')

    for filename in tqdm(image_files, desc='Prediction', ncols=100, ascii=True):
        image_path = os.path.join(test_folder, filename)
        predicted_class, confidence = predict_image(image_path, model, device, class_names)
        results.append({
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence
        })
        print(f'{filename}: {predicted_class} ({confidence:.2f}%)')

    return results


if __name__ == '__main__':
    main()
