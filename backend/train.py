import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image
import json

def load_data_from_folder(data_dir):
    images = []
    labels = []
    
    # Iterate through files in directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            # Format: {label}_{scene_id}_{longitude}_{latitude}.png
            # label is 1 for ship, 0 for no-ship
            label = int(filename.split('_')[0])
            img_path = os.path.join(data_dir, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((80, 80)) # Ensure it's 80x80
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    return np.array(images), np.array(labels)

def build_model():
    # CNN Model for 80x80 RGB images
    model = models.Sequential([
        layers.InputLayer(input_shape=(80, 80, 3)),
        # Normalization layer
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Build absolute path robustly so it works from any terminal location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'dataset', 'shipsnet', 'shipsnet')
    
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory {data_dir} not found.")
        print("Please download 'Ships in Satellite Imagery' from Kaggle, extract it, and place the 'shipsnet' folder containing the PNGs inside 'ship_detection/dataset/'.")
        return
        
    print("Loading data...")
    X, y = load_data_from_folder(data_dir)
    print(f"Loaded {len(X)} images.")
    
    if len(X) == 0:
        print("No images found. Exiting.")
        return
        
    print("Splitting data for cross validation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Building model...")
    model = build_model()
    
    # Checkpoint to save best model
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("ship_model.keras", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Predict on validation set
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()
    
    # Create results folder
    results_dir = os.path.join(base_dir, 'training_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Ship", "Ship"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Generate Accuracy Curve
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Precisión (Accuracy) del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'accuracy_curve.png'))
    plt.close()
    
    # Generate Loss Curve
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Pérdidas (Loss) del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
    plt.close()
    
    # Generate Classification Report
    report = classification_report(y_test, y_pred, target_names=["No Ship", "Ship"])
    with open(os.path.join(results_dir, 'metrics_report.txt'), 'w') as f:
        f.write("=== Reporte de Clasificación (Validación Cruzada) ===\n")
        f.write(report)
        f.write(f"\nTest Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    
    print(f"\n✅ Resultados de validación guardados en: {results_dir}")
    
    # Save the model
    model.save("ship_model_final.keras")
    print("Model saved as ship_model_final.keras")

if __name__ == "__main__":
    main()
