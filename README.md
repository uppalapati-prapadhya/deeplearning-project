

# Pneumonia Detection Using VGG16

This project implements a deep learning model using the **VGG16 architecture** to detect **Pneumonia** from chest X-ray images. The model is built using **TensorFlow** and **Keras** and deployed using **Streamlit**.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Accuracy](#model-accuracy)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to build a binary classifier that can accurately identify chest X-rays as **Pneumonia** or **Normal** using the VGG16 pre-trained model. The project also includes a web interface built with **Streamlit** where users can upload X-ray images for predictions.

## Dataset
The dataset used in this project is the **Chest X-ray Images (Pneumonia)** dataset, available on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

The dataset is split into three categories:
- **Training Data:** Used for training the model.
- **Validation Data:** Used for evaluating the model during training.
- **Test Data:** Used for final evaluation of model performance.

## Technologies Used
- **Python 3.10+**
- **TensorFlow 2.16.2**
- **Keras**
- **VGG16**
- **Streamlit** (for deployment)
- **NumPy**
- **Pillow**

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/deeplearning-project.git
cd deeplearning-project
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
You can download the Chest X-ray Pneumonia dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and unzip it into your project directory.

### 5. Run the Model Training
```bash
python train_model.py
```

### 6. Run the Streamlit Web App
```bash
streamlit run app.py
```

## Usage
1. Upload a chest X-ray image.
2. The model will classify the image as **Pneumonia** or **Normal**.
3. The result will be displayed with the prediction confidence.

## Model Accuracy
The model achieves a **97.04% accuracy** on the test set and has been trained using transfer learning from the **VGG16** pre-trained model.

## Contributing
Contributions are welcome! Please fork this repository and submit pull requests with detailed descriptions of your changes.


