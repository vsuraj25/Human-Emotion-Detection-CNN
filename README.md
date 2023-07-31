[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

<div id="top"></div>

<div align="center">
  <a href="https://github.com/vsuraj25">
    <img src="https://img.icons8.com/water-color/50/happy.png" alt="Logo" width="80" height="80"/> 
  </a>

    
<h3 align="center">Human Emotion Detection Using CNN</h3>

 <p align="center">
    Deep Learning Project 
    <br />
    <a href="https://github.com/vsuraj25"><strong>Explore my Repositories. »</strong></a>
    <br />
    <br />
    <a href="#intro">Introduction</a>
    ·
    <a href="#data"> Data Information</a>
    ·
    <a href="#contact">Contact</a>
  </p>
</div>

<div id="intro"></div>

## 1) **Introduction**
This project aims to build and compare three different models for human emotion detection using the Fer2013 dataset. The models include a custom Lenet-like model, an implementation of the ResNet18 model from scratch, and a pretrained model using EfficientNetB4. The project explores data ingestion, transformation, and training processes, along with presenting the model results.

<div id="data"></div>

## 2) **Dataset Information**
The Fer2013 dataset is a widely used benchmark for facial emotion recognition. It contains 35,887 grayscale images of size 48x48 pixels, categorized into seven emotion classes: angry, disgust, fear, happy, sad, surprise, and neutral. Download the dataset here [FER2013 Dataset](https://www.dl.dropboxusercontent.com/s/skptyhjlrubpvgq/fer2013_zipped.zip?dl=0)

## 3) **Stages of the Project**
a. Data Ingestion
The dataset is automatically extracted from the provided zip file link. The Fer2013 dataset contains images and corresponding labels for training and validation.

b. Data Transformation
Data augmentation techniques are applied to increase the diversity of the training dataset. The transformed data is converted into TensorFlow records for efficient training.

c. Custom Model Training
A custom Lenet-like model is implemented and trained on the transformed data. The model has approximately 6.15 million parameters.

d. ResNet18 Model
An implementation of the ResNet18 model is created from scratch and trained on the same dataset. The ResNet18 model contains around 11.19 million parameters.

e. Pretrained Model
A pretrained EfficientNetB4 model is used for transfer learning. The model is fine-tuned on the Fer2013 dataset. The EfficientNetB4 model has around 19.65 million parameters.

## 4) **CNN Architectures Used**
- Custom Lenet-like Model: This architecture is inspired by the Lenet model and customized for emotion detection. It consists of convolutional and pooling layers, followed by fully connected layers. Despite being a simple model, it can still achieve reasonable performance.

- ResNet18 Model: ResNet18 is a popular deep residual network architecture. It utilizes skip connections to mitigate the vanishing gradient problem, allowing for the successful training of deeper networks. The model's architecture includes residual blocks with varying depths.

- Pretrained EfficientNetB4 Model: EfficientNetB4 is an efficient and effective convolutional neural network that balances model size and accuracy. The pretrained model is fine-tuned on the emotion detection task, enabling it to leverage prior knowledge learned on a large dataset.

## 5) **Model Results**

| Model                  | Accuracy | Validation Accuracy | Loss   | Epochs |
|------------------------|----------|---------------------|--------|--------|
| Custom Lenet-like      | 61.45%   | 54.39%              | 1.024  | 60     |
| ResNet18               | 68.90%   | 64.03%              | 0.934  | 25     |
| Pretrained EfficientNetB4 | 78.35%   | 68.49%              | 0.332  | 40     |

## 5) **Setup Information**
To set up and install the project, follow these steps:

1. Clone the repository from GitHub.
2. Install the required dependencies listed in the `requirements.txt` file.
3. Run the data ingestion script to extract the dataset.
4. Run the data transformation script to augment the data and convert it into TensorFlow records.
5. Train the custom Lenet-like model, ResNet18 model, and the pretrained EfficientNetB4 model using the respective training scripts.

## 6) **Reproducibility using DVC**
This project uses DVC (Data Version Control) to manage the data and model versions. To reproduce the entire project, follow these steps:

1. Install DVC on your system.
2. Use the command `dvc repro` to reproduce the entire pipeline, from data ingestion to model training.

By following these steps, you can recreate the exact environment and reproduce the results of the project easily.

## 7) **Further Improvements**
Experimentation by increasing the epochs and other dependable parameters can improve the model performance even more.

## 8) **Requirements**
* Python 3.7
* Numpy
* Tensorflow
* Streamlit
* Pytest
* Tox
* DVC
* MLFlow
* Checkout requirements.txt for more information.

## 9) **Technologies used**
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dataversioncontrol&logoColor=white)
![MLFlow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![Keras](https://img.shields.io/badge/Keras-D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Yaml](https://img.shields.io/badge/YAML-CB171E.svg?style=for-the-badge&logo=YAML&logoColor=white)
![Json](https://img.shields.io/badge/JSON-000000.svg?style=for-the-badge&logo=JSON&logoColor=white)


## 10) **Tools used**
![Visual Studio Code](https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![Conda](https://img.shields.io/badge/Anaconda-44A833.svg?style=for-the-badge&logo=Anaconda&logoColor=white)

<!-- CONTACT -->
<div id="contact"></div>

## 11) **Contact**
[![Suraj Verma | LinkedIn](https://img.shields.io/badge/Suraj_Verma-eeeeee?style=for-the-badge&logo=linkedin&logoColor=ffffff&labelColor=0A66C2)][reach_linkedin]
[![Suraj Verma | G Mail](https://img.shields.io/badge/sv255255-eeeeee?style=for-the-badge&logo=gmail&logoColor=ffffff&labelColor=EA4335)][reach_gmail]
[![Suraj Verma | G Mail](https://img.shields.io/badge/My_Portfolio-eeeeee?style=for-the-badge)][reach_gmail]

[reach_linkedin]: https://www.linkedin.com/in/suraj-verma-982b31157/
[reach_gmail]: mailto:sv255255@gmail.com?subject=Github


<p align="right">(<a href="#top">back to top</a>)</p>