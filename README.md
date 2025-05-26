# 🐾 Animal Image Classification using CNN with Keras

This project involves building, training, and evaluating a Convolutional Neural Network (CNN) to classify images from the [Animals10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10) into 10 different animal categories. It is part of a research initiative to learn CNN fundamentals before transitioning to medical image analysis using X-rays for swallowing studies.

## 📌 Project Objectives

- Understand and build a basic CNN from scratch using Keras.
- Apply data preprocessing and augmentation to improve generalization.
- Prevent model overfitting using augmentation and dropout layers.
- Evaluate model performance using accuracy, loss, and test set metrics.
- Set up the foundation for future medical image classification tasks (CT and VFSS).

## 🧠 Model Architecture

The CNN architecture includes:
- Multiple convolutional and max-pooling layers
- Dropout layers for regularization
- Dense layers with ReLU and softmax activations
- Categorical cross-entropy loss and Adam optimizer

> Trained using 25 epochs on the augmented Animals10 dataset.

## 📈 Final Results

| Metric        | Value        |
|---------------|--------------|
| Training Accuracy | 92.8% |
| Test Accuracy     | 94.8% |
| Final Loss        | 0.17 |

Visual results (loss and accuracy graphs) are available in the `results/` folder.

## 📊 Data Augmentation Techniques Used

- Rotation
- Horizontal Flip
- Zoom
- Width and Height Shift
- Shear

## 📁 Folder Structure
├── train_model.py # Main training script
├── evaluate_model.py # Evaluation and testing script
├── /images # Sample dataset images
├── /results # Graphs for accuracy/loss
└── README.md # This file


## 🚀 Next Steps

This CNN classification task is the foundation for upcoming work involving:
- Visual LLMs for multimodal learning
- Fine-tuning LLMs using Hugging Face datasets (e.g., CT-RATE)
- Integrating image and report analysis for medical datasets

## 🤝 Acknowledgements

This work is done under the supervision of **Dr. Ayman Anwar**, and is part of a research path focused on **swallowing kinematics and multimodal learning** using deep learning.

---

✅ Built with: `TensorFlow`, `Keras`, `Matplotlib`, `NumPy`, `Google Colab`.

---

## 📬 Contact

If you'd like to learn more or collaborate, feel free to reach out via [LinkedIn](https://linkedin.com/in/selinazarzour/).
