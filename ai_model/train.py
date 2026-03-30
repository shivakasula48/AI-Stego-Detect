import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Fixed seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

logger = logging.getLogger(__name__)

def create_model(input_shape=(128, 128, 3)):
    """
    Creates a 'Sane Stegna-Net' architecture.
    Balances high-resolution early features with manageable parameter counts.
    """
    model = Sequential([
        # Block 1: High-res entry (NO POOL)
        Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        
        # Block 2: Initial downsampling
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Block 3: Further downsampling
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Block 4: Final downsampling
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Classifier head
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_split_dataset(base_dir="dataset", target_size=(128, 128)):
    """
    Loads train/val split from directories.
    Skips resizing if the image is already the correct target size.
    """
    from PIL import Image
    import cv2

    def load_dir(path, label):
        X, y = [], []
        if not os.path.exists(path):
            return X, y
        for fname in os.listdir(path):
            # Ignore debug difference images
            if not fname.lower().endswith('.png') or '_diff.png' in fname:
                continue
            img = cv2.imread(os.path.join(path, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Skip redundant resizing to prevent interpolation noise
            if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
                img = cv2.resize(img, target_size)
                
            X.append(img)
            y.append(label)
        return X, y

    logger.info("Using train/val split directories.")
    tc_X, tc_y = load_dir(os.path.join(base_dir, "train", "clean"), 0)
    ts_X, ts_y = load_dir(os.path.join(base_dir, "train", "stego"), 1)
    vc_X, vc_y = load_dir(os.path.join(base_dir, "val",   "clean"), 0)
    vs_X, vs_y = load_dir(os.path.join(base_dir, "val",   "stego"), 1)

    X_train = np.array(tc_X + ts_X)
    y_train = np.array(tc_y + ts_y)
    X_val   = np.array(vc_X + vs_X)
    y_val   = np.array(vc_y + vs_y)

    return X_train, y_train, X_val, y_val


def train_model(model_save_path="ai_model/stego_detector.h5", base_dir="dataset"):
    """
    Trains the CNN model. Returns best validation accuracy (0-100 float).
    """
    logger.info("Loading dataset...")
    X_train, y_train, X_val, y_val = load_split_dataset(base_dir=base_dir)

    if len(X_train) == 0:
        logger.error("No training data found.")
        return None

    # Normalization
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0

    # Dynamic Class Weighting
    from sklearn.utils import class_weight as cw
    target_weights = cw.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(target_weights))
    logger.info(f"Using class weights: {class_weight_dict}")

    model = create_model()
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    batch_size = 16
    
    logger.info(f"Starting training — {len(X_train)} train, {len(X_val)} val, batch={batch_size}...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        shuffle=True,
        verbose=2
    )

    best_val_acc = max(history.history['val_accuracy']) * 100
    
    # Classification report + confusion matrix
    y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    logger.info("\nClassification Report:\n" + 
                classification_report(y_val, y_pred, target_names=["Clean", "Stego"]))
    
    cm = confusion_matrix(y_val, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Stegna-Net Accuracy')
    plt.legend()
    plt.savefig('ai_model/accuracy_plot.png')
    
    return best_val_acc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    train_model()
