
# Smile Detection using CNN and TensorFlow

This project uses a Convolutional Neural Network (CNN) to classify facial images based on whether the person is smiling or not. The model is built using TensorFlow and Keras, and trained on a labeled dataset of facial images.

## 📚 Description

- Detects smiles using image classification.
- Preprocessing includes face detection, resizing, and normalization.
- Uses OpenCV for image manipulation and TensorFlow/Keras for deep learning.

## 🧰 Libraries Used

- TensorFlow  
- Keras  
- OpenCV (cv2)  
- NumPy  
- Matplotlib  
- OS, glob, random  

## 🛠️ Features

- Face detection using Haar cascades.  
- CNN architecture with Conv2D, MaxPooling, Dense layers.  
- Binary classification of 'smiling' vs 'not smiling'.  
- Model evaluation with training and validation accuracy/loss plots.  

## 🚀 How to Run

1. Clone this repository.  
2. Install dependencies from \`requirements.txt\`.  
3. Run the notebook using Jupyter.

```bash
pip install -r requirements.txt
jupyter notebook "SMILE DETECTION USING CNN AND TENSORFLOW.ipynb"
```

## 📁 File Structure

- `SMILE DETECTION USING CNN AND TENSORFLOW.ipynb` - Main notebook with all code.  
- `requirements.txt` - List of Python libraries needed.  
- Images and trained models are generated during execution.

## 🔮 Future Enhancements

- Real-time smile detection using webcam.  
- More advanced models (e.g., MobileNet, ResNet).  
- Dataset augmentation and hyperparameter tuning.  

## 📄 License

This project is licensed under the MIT License.

