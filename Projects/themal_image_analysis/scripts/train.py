import os
from preprocess import prepare_data
from model import create_cnn_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(data_dir, model_save_path, epochs=20, batch_size=32):
    """
    Train the CNN model on the dataset.
    
    Args:
    - data_dir (str): Path to the dataset directory.
    - model_save_path (str): Path to save the trained model.
    - epochs (int): Number of epochs to train.
    - batch_size (int): Size of each training batch.
    """
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data_dir)
    input_shape = X_train.shape[1:]
    
    # Create model
    model = create_cnn_model(input_shape)
    
    # Callbacks
    checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    
    # Train model
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping]
    )

    # Save final model
    model.save(model_save_path)
    print("Model training completed and saved.")

if __name__ == "__main__":
    data_dir = 'themal_image_analysis\\data\\train'
    model_save_path = './models/thermal_cnn.h5'
    train_model(data_dir, model_save_path)
