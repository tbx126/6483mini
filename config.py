
# ========================================
# 配置文件: 超参数和路径设置
# ========================================

# 数据路径配置
DATA_CONFIG = {
    'train_dir': 'dataset/train',
    'val_dir': 'dataset/val',
    'test_dir': 'dataset/test',
    'model_save_path': 'best_model.pth',
    'predictions_save_path': 'predictions.csv'
}

# 模型超参数
MODEL_CONFIG = {
    'img_size': 224,
    'num_classes': 2,
    'batch_size': 32,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'dropout_rate': 0.5
}

# 训练配置
TRAINING_CONFIG = {
    'num_workers': 4,
    'pin_memory': True,
    'save_best_only': True,
    'early_stopping_patience': 5,
    'lr_scheduler_patience': 3,
    'lr_scheduler_factor': 0.5
}

# 设备配置
DEVICE_CONFIG = {
    'use_cuda': True,
    'cuda_device': 0
}

# 类别配置
CLASS_CONFIG = {
    'class_names': ['cat', 'dog'],
    'class_to_idx': {'cat': 0, 'dog': 1}
}
