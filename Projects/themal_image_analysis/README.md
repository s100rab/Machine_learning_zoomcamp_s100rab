# Thermal Image Analysis Project - Chips Dataset

## Project Overview

This project analyzes thermal images of chips using machine learning techniques to detect classification and bounding box features. The analysis involves preprocessing thermal images, training a deep learning model with a dual output for classification and bounding box regression, and evaluating model performance. The dataset consists of thermal images captured at various time intervals and annotated using the YOLO format.

---

## Features

1. **Data Preprocessing:**
   - Conversion of pixel values to temperature readings.
   - Support for YOLO-format annotations for bounding box localization.
   - Normalization and padding of bounding box data for consistent model input.

2. **Model Training:**
   - A Convolutional Neural Network (CNN) with dual output:
     - Classification Output: Binary classification of the chip's condition.
     - Bounding Box Output: Regression of bounding box coordinates for detected features.
   - Weighted loss functions to balance classification and bounding box prediction tasks.

3. **Evaluation:**
   - Visualization of training history for accuracy and loss curves.
   - Evaluation metrics for both classification accuracy and bounding box Mean Squared Error (MSE).
   - Visualization of predictions on test data.

4. **Temperature Curve Analysis:**
   - Extraction of temperature data from thermal images.
   - Calculation of temperature differences over time.
   - Visualization of temperature changes using plots.

---

## Dataset

The project uses the **Chips Dataset**, consisting of:
- **Thermal Images**: Grayscale images representing temperature data.
- **YOLO Annotations**: Text files specifying bounding box coordinates and class labels.
- **Directory Structure**:
  ```
  dataset/
  ├── train/
  │   ├── images/                  # Thermal images for training
  │   └── annotations_yolo_format/ # YOLO-format bounding box annotations
  ├── test/
  │   ├── images/                  # Thermal images for testing
  │   └── annotations_yolo_format/ # YOLO-format bounding box annotations
  ```

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the directory structure and place the dataset in the `dataset` directory.

---

## Usage

### Preprocess the Data

1. **Prepare Training Data**:
   - Load thermal images and YOLO annotations.
   - Normalize pixel values and bounding box coordinates.
   - Pad bounding box data for consistency.

   Example:
   ```python
   from preprocess import prepare_data
   data_dir = "./dataset/train"
   X_train, X_test, y_train, y_test, padded_boxes = prepare_data(data_dir)
   ```

2. **Plot Temperature Differences**:
   - Extract temperature differences from thermal images and plot the curve.

   Example:
   ```python
   from temperature_analysis import compute_temperature_differences, plot_temperature_differences
   temperature_differences = compute_temperature_differences(thermal_images, roi=(10, 10, 50, 50))
   plot_temperature_differences(temperature_differences)
   ```

---

### Train the Model

1. **Initialize the CNN Model**:
   ```python
   from model import create_cnn_model_with_bboxes
   input_shape = X_train.shape[1:]
   model = create_cnn_model_with_bboxes(input_shape)
   ```

2. **Train the Model**:
   ```python
   from train import train_model
   model_save_path = "./models/thermal_cnn"
   history = train_model(X_train, X_test, y_train, y_test, padded_boxes, model_save_path)
   ```

---

### Evaluate the Model

1. **Plot Training History**:
   ```python
   from visualize import plot_training_history
   plot_training_history(history)
   ```

2. **Evaluate Performance**:
   ```python
   from evaluate import evaluate_model
   evaluate_model(model_save_path, X_test, y_test, padded_boxes[len(X_train):])
   ```

---

## Results

- **Classification Accuracy**: Achieved ~99% accuracy on training and validation sets.
- **Bounding Box MSE**: Bounding box predictions show improvements with loss weighting and data normalization.
- **Temperature Analysis**: Plotted temperature difference curves to track chip heating patterns over time.

---

## Future Enhancements

1. **Enhanced Model Architecture**:
   - Incorporate advanced models like Faster R-CNN for bounding box predictions.
   - Use transfer learning with pretrained networks for better feature extraction.

2. **Improved Data Augmentation**:
   - Apply thermal-specific augmentations such as temperature scaling and simulated noise.

3. **Additional Metrics**:
   - Implement precision-recall and IoU (Intersection over Union) metrics for bounding box evaluation.

---

## Dependencies

- Python >= 3.8
- TensorFlow >= 2.0
- OpenCV
- Matplotlib
- NumPy
- Scikit-learn

---

## Acknowledgments

Special thanks to the contributors of the **Chips Dataset** and the community for providing tools and resources for thermal image analysis.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## Contact

For questions or feedback, please contact:
- **Email**: [sourabh.lakhera2015@gmail.com]
- **Feel_free_to_connect**: [[Connect_me](https://linktr.ee/s100rab)]
