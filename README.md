# Classification on Describable Textures Dataset

This project implements classification on the Describable Textures Dataset (DTD) using a baseline SimpleCNN and a deeper ResNet-18 model trained from scratch. The workflow includes data preparation, EDA, hyperparameter tuning, model training, and final evaluation.

## Files

- Descriptive_Statistics_DTD.ipynb : Descriptive Statistics
- dtd_dataset_eda.ipynb : EDA Visualization
- DTD_SIMPLE_CNN.ipynb : SimpleCNN model training, saving model
- simple_cnn_dtd_model3.pth : Saved SimpleCNN model
- resnet18_train.ipynb : (ResNet-18) Hyperparameter tuning, final training, saving best model
- resnet18_dtd_best.pth : Saved best (ResNet-18) model
- test.py : Loads the saved ResNet-18 model and evaluates it on the DTD test set (accuracy, precision, recall, F1 score, confusion matrix, classification report)

## How to run

1. Make sure the following two files are in the same folder:
   - resnet18_dtd_best.pth (saved best model)
   - test.py
2. Install dependencies: pip install torch torchvision numpy matplotlib seaborn scikit-learn
3. Run the test script to evaluate the model: python test.py

This will compute the accuracy, precision, recall, F1 score, display the confusion matrix for all 47 DTD classes, and print the classification report.

# How to run the website

1. install streamlit and pillow, `pip install streamlit pillow`
2. Make sure the `resnet18_dtd_best.pth` file is in the same directory as the `resnetmodel_website.py`.
3. Run streamlit code, `streamlit run resnetmodel_website.py`
4. The website will open in a browser
5. To see the list of classes, click on the `Show DTD Class Names` button.
6. Upload an image using the `Browse Files` Button.
7. Once the image is loaded, the model will run inference and provide the top 5 predicted classes and confidence.

## Additional Resources

- [Project Blog Post](https://medium.com/@harshitamandalika029/classification-on-describable-textures-dataset-dtd-0b5b4849c214)
- [Report](https://github.com/vishwamraval/ECEN-758-Final-Project-Group8/blob/main/Group_8_Report.pdf)
