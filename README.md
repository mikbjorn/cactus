# Cactus Identification
Authors: Mikkel Bjornson, Rory Higgins, & Mary Kate Thomerson
## Background
This is an image classification analysis of the Aerial Cactus Identification Dataset (López-Jiménez et al,
2019). It explores the ability to count flora from aerial photography, allowing remote assessment of large areas of wilderness.  Four different models and a weighted average of all four are compared. Model features include RGB color values for each pixel as well as edge features, created by the difference of adjacent pixels. 
## Goals
-	Assess the ability of machine learning models to identify cactus through use of aerial photography.
-	Compare accuracy of SVM, Random Forest, XGBoost, and CNN models. 
## Analysis and Findings
Models were first tested against a locally created hold out dataset, then again against the Kaggle test set. The CNN model produced the highest levels of accuracy (test: 97.8%, Kaggle: 95.6%).  The next closest model was the XGBoost model (test: 95.4%, Kaggle: 93.8%). The weighted average model obtained best results with the weights 47.8% CNN, 48.8% XGBoost, and 3.4% Random Forest. The SVM model provided no additional increase in accuracy with a weight of 0%. This produced slight gains in accuracy (test: 97.4%, Kaggle: 96.1%).  
## Recommendations 
The weighted average model does produce the highest accuracy. The fitting of all three models may prove to be computationally expensive and time consuming. With identification of cactus at over 95% accuracy, the CNN model is likely the best mix of accuracy and efficiency. With further training or increased image resolution, the CNN model may still be able to provide even higher levels of accuracy. 
## Benefits
The accuracy rate is likely high enough to produce meaningful insights into the population of Cactus by aerial photography. This can reduce the need for in person observation of environments to assess population health of different flora. 
## Citation 
López-Jiménez, E., Vasquez-Gomez, J.I., Sanchez-Acevedo, M.A., Herrera-Lozada, J.C., Uriarte-Arcia, A.V..
(March 8, 2019). Aerial Cactus Identification. Retrieved [May 11, 2021] from https://www.kaggle.com/c/aerial-cactus-identification/data.
