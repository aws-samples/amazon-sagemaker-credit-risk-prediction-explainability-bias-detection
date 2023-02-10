## Credit risk prediction and explainability and bias detection with Amazon SageMaker

In this workshop, we demonstrate a end to end ML use case of credit risk prediction with model explainability and bias mitigation. We use a well known open source dataset https://archive.ics.uci.edu/ml/datasets/South+German+Credit+%28UPDATE%29 .
We show how to use SageMaker to mitigate bias in models trained on a biased dataset with a SageMaker inference pipeline model. 

![Credit risk explainability use case](images/NewTitleCard.PNG)

## 1. Overview
Amazon SageMaker helps data scientists and developers to prepare, build, train, and deploy high-quality machine learning (ML) models quickly by bringing together a broad set of capabilities purpose-built for ML.

[Amazon SageMaker Automatic Model Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html), also known as hyperparameter tuning, finds the best version of a model by running many training jobs on your dataset using the algorithm and ranges of hyperparameters. It then chooses the hyperparameter values that result in a model that performs the best, as measured by a single performance metric (e.g., accuracy, auc, recall) that we define.

Amazon SageMaker provides pre-made images for machine and deep learning frameworks for supported frameworks such as Scikit-Learn, XGBoost, TensorFlow, PyTorch, MXNet, or Chainer. These are preloaded with the corresponding framework and some additional Python packages, such as Pandas and NumPy, so you can write your own code for model training. See [here](https://docs.aws.amazon.com/sagemaker/latest/dg/algorithms-choose.html#supported-frameworks-benefits) for more information.

[Amazon SageMaker Studio](https://aws.amazon.com/sagemaker/studio/) provides a single, web-based visual interface where you can perform all ML development activities including notebooks, experiment management, automatic model creation, debugging, and model and data drift detection.

In this SageMaker Studio notebook, we highlight how you can use SageMaker to train a model, while considering multiple objectives & fairness when dealing with a biased dataset. 

Below is a high level view of the architecture this lab will take you through:  
![Credit risk explainability model inference](images/Architecture.png)

Below is the architecture diagram used in the solution:
![alt text](clarify_inf_pipeline_arch.jpg)


The three notebooks will perform the following steps:

1. Prepare raw training and test data by generating bias in the dataset 
2. Create a SageMaker Processing job which performs preprocessing on the raw training data.
3. Train an XGBoost model on the processed data using SageMaker's built-in XGBoost container. This model will focus on optimizing for a single objective
4. Create a SageMaker model endpoint for inference
5. Perform inference by supplying processed test data
6. Train another XGBoost model on the processed data using SageMaker's built-in XGBoost container. This model will focus on optimizing for multiple objectives (accuracy & fairness)
7. Create a SageMaker model endpoint for the multi objective model as well and perform inference using processed test data
8. Compare inferences of the two models 
9. Clean up


## Lab Instructions


1. In the terminal, type the following command:

git clone https://github.com/aws-samples/amazon-sagemaker-credit-risk-prediction-explainability-bias-detection

![alt text](static/18.png)

2. After completion of step 1 you will have amazon-sagemaker-credit-risk-prediction-explainability-bias-detection folder created in left panel of the studio:

![alt text](static/19.png)

3. Under amazon-sagemaker-credit-risk-prediction-explainability-bias-detection double click on credit_risk_explainability_inference_pipelines_with_output.ipynb and Select Kernel as Python 3 (Data Science)

![alt text](static/20.png)

Congratulations!! You have successfully downloaded the content of the Credit Risk Explainability lab, please follow the instructions in the jupyter notebook.
![alt text](static/21.png)




