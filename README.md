
# 🧠 Brain Tumor Classification using Deep Learning and PyTorch

![image](https://github.com/user-attachments/assets/746c067f-3b9a-46bb-8165-a6d9d0c7ebbe)




## 🚀 Project Overview

### Motivation
Brain tumor detection and classification is one of the most challenging areas in medical imaging. Traditional methods often suffer from:
- **Time-consuming processes**
- **Subjectivity** in diagnosis
- **Human errors**

This project leverages **Deep Learning** techniques to provide a solution that automates the classification of brain tumors from MRI scans. The goal is to enhance diagnostic efficiency and accuracy.

### Scientific Significance
Using state-of-the-art **Deep Learning** models, this project aims to:
- **Automate medical image analysis**
- **Classify tumor types** with high precision
- **Improve diagnostic speed** and reduce human intervention

## 🧬 Medical Imaging & Deep Learning

### Key Technologies
- **Framework**: [PyTorch](https://pytorch.org/) - A leading framework for deep learning.
- **Model Architecture**: **Convolutional Neural Network (CNN)** - A powerful architecture for image analysis.
- **Image Type**: **MRI Scans** - A common imaging modality for brain tumor detection.

![image](https://github.com/user-attachments/assets/36b1e7c6-ee0a-4991-9e93-c2317fe4d245)


### Research Objectives
1. **Develop an automated system** for brain tumor detection using CNN.
2. **Achieve high accuracy** in classifying multiple tumor types (e.g., glioma, meningioma, pituitary tumor).
3. **Reduce diagnostic time** and provide support for clinicians.

## 🔬 Methodology

### Data Processing Workflow
1. **Data Exploration**: Conducting in-depth analysis to understand data patterns, missing values, and distribution.
2. **Image Augmentation**: Applying transformations like rotations, flips, and zoom to increase dataset diversity and improve generalization.
3. **Model Training**: Training the CNN using a supervised learning approach on labeled MRI data.
4. **Multi-Class Classification**: The model classifies MRI scans into different categories based on tumor types.

![image](https://github.com/user-attachments/assets/cb4fc4b2-3163-4c71-905f-3691b4d87a18)


### Technical Components
- **Dataset Splitting**: The dataset is split into training, validation, and test sets for robust model evaluation.
- **Convolution Output Dimension Calculation**: Dynamically compute the output dimensions of convolution layers.
- **Max Pooling**: Use max pooling to reduce dimensionality while preserving critical features.
- **Feature Extraction**: Convolutional layers automatically extract relevant features from raw images.

## 🚀 Key Features
- **Automated MRI scan analysis**: AI-driven system to classify tumor types.
- **Multi-class classification**: Detect and classify various tumor types.
- **Computational efficiency**: Designed to be fast and resource-efficient.
- **Customizable deep learning architecture**: Adaptable for other medical image classification tasks.

![image](https://github.com/user-attachments/assets/8daa2dcd-514f-4592-acb4-26fd725783be)


## 🎯 Expected Outcomes
- **Improved diagnostic accuracy**: Reducing human error in detecting brain tumors.
- **Faster medical image processing**: Speeding up the diagnosis and treatment planning.
- **Reduced human bias**: Delivering consistent results in tumor classification.
- **Support for healthcare professionals**: Assisting clinicians in accurate decision-making.

## 📊 Performance Metrics
To evaluate the performance of the model, the following metrics are used:
- **Accuracy**: Overall correctness of the model’s predictions.
- **Precision**: The proportion of true positive predictions.
- **Recall**: The proportion of true positives among all actual positive instances.
- **F1 Score**: A balanced metric combining precision and recall.

![image](https://github.com/user-attachments/assets/13c2311e-6849-44b4-9178-21765b157393)


## 🔍 Potential Applications
- **Neuro-oncology**: Use in clinical settings to assist oncologists in tumor detection and treatment planning.
- **Radiology departments**: Enhancing the workflow of radiologists by automating classification.
- **Clinical Decision Support Systems**: AI-powered tools that provide support for clinicians in diagnosing brain tumors.
- **Medical Research**: Facilitating further research on brain tumor types and AI models for medical imaging.

## 🛠 Requirements
To run this project, the following packages are required:
- **Python 3.8+**
- **PyTorch** (Deep learning framework)
- **NumPy** (Numerical operations)
- **Pandas** (Data handling)
- **Matplotlib** (Data visualization)
- **scikit-learn** (For model evaluation)

## 📦 Installation

### Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier
pip install -r requirements.txt
```

## 🧪 Usage

### Training the Model
Train the model using the following command:
```bash
python train.py
```

### Performing Inference
To classify a new MRI scan, use the following Python script:

```python
from model import CNN_TUMOR
import torch
from torchvision import transforms
from PIL import Image

# Load the trained model
model = CNN_TUMOR(params)
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# Prepare the image for prediction
img = Image.open("path_to_image.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
img_tensor = transform(img).unsqueeze(0)

# Predict the tumor type
output = model(img_tensor)
prediction = torch.argmax(output, dim=1)
print(f"Predicted tumor type: {prediction.item()}")
```

## 📅 Milestones

- **Phase 1**: Data collection and preprocessing (Completed)
- **Phase 2**: Model training and optimization (In progress)
- **Phase 3**: Evaluation and real-world testing (Upcoming)

## 📝 License
MIT license 

## 🤝 Contributing
We welcome contributions! If you'd like to help with the project, please fork the repository, submit issues, or create pull requests.

## 📬 Contact
For questions or feedback, reach out to:
- **Email**: [dhouha.meliane@esprit.tn]
- **LinkedIn**: [https://www.linkedin.com/in/dhouha-meliane/]

---

**Disclaimer**: This project is intended as a research tool and should not replace professional medical diagnoses. Always consult healthcare professionals for accurate diagnoses.

---
