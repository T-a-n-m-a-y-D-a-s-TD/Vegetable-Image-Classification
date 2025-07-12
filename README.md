🥦 Vegetable Image Classification

This project compares two image classification techniques using the **Vegetable Image Dataset** from Kaggle. It includes:

1. Handcrafted Feature Extraction + SVM
2. Convolutional Neural Network (CNN)


📁 Dataset

- **Name**: [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- **Total Images**: 21,000
- **Classes**: 15 (e.g., Tomato, Potato, Onion, Ginger, etc.)
- **Image per class**: 1,400
- **Split**: Train / Validation / Test folders


📌 Project Structure

├── handcrafted.ipynb # Traditional ML with HOG + LBP + SVM
├── cnn-classification.ipynb # CNN-based deep learning classification
├── README.md


🧠 Method 1: Handcrafted Features + SVM

- **Preprocessing**: Resize to 64×64, grayscale
- **Feature Extraction**:
  - HOG (Histogram of Oriented Gradients)
  - LBP (Local Binary Pattern)
- **Model Used**: Support Vector Machine (SVM)
- **Libraries**: OpenCV, scikit-image, scikit-learn
- **Accuracy Achieved**: ~95%


⚙️ Method 2: CNN-Based Deep Learning

- **Preprocessing**: Resize to 224×224, rescale pixels, data augmentation
- **Model Architecture**:
  - Multiple Conv2D + MaxPooling layers
  - Flatten + Dense + Dropout layers
- **Libraries**: TensorFlow, Keras
- **Training**: 30 epochs, Adam optimizer, batch size 64
- **Accuracy Achieved**: ~93%


📊 Comparison Summary

| Feature             | Handcrafted + SVM | CNN-Based Deep Learning |
|---------------------|-------------------|--------------------------|
| Feature Extraction  | Manual (HOG, LBP) | Automatic (CNN filters) |
| Input Image Size    | 64×64 (gray)      | 224×224 (RGB)           |
| Accuracy            | ~95%              | ~93%                    |
| Training Time       | Short             | Longer                  |
| Flexibility         | Low               | High                    |
| Learning Type       | Supervised        | Supervised              |


✅ Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib, Seaborn
- OpenCV, scikit-learn, scikit-image (for handcrafted)
- TensorFlow/Keras (for CNN)


▶️ How to Run

1. Clone the repo:
   git clone https://github.com/your-username/vegetable-classification.git
   cd vegetable-classification
Run notebooks:

handcrafted.ipynb – Traditional ML

cnn-classification.ipynb – Deep Learning

Dataset must be downloaded from Kaggle and placed in the proper directory (update path if needed).

📌 License
This project is open source under the MIT License.

🙌 Acknowledgements
Dataset by Ahmed Misrak on Kaggle
