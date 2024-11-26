import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocess import prepare_data

def evaluate_model(model_path, data_dir):
    """
    Evaluate the trained model on the test dataset.
    
    Args:
    - model_path (str): Path to the trained model file.
    - data_dir (str): Path to the test dataset directory.
    """
    X_train, X_test, y_train, y_test = prepare_data(data_dir)
    
    # Load model
    model = load_model(model_path)
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Plot some predictions
    predictions = model.predict(X_test)
    
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(X_test[i].reshape(64, 64), cmap='hot')
        plt.title(f"True: {y_test[i]}, Pred: {predictions[i][0]:.2f}")
        plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    model_path = './models/thermal_cnn.h5'
    data_dir = './data/test'
    evaluate_model(model_path, data_dir)
