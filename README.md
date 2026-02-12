# Video-Classification-and-Summarisation
## Summary
This project is a dissertation project:
Classify and summarise the action from the input videos by using pyTorch environment with action video dataset
- The UCF101-action dataset was used to train CNN-LSTM model.
- The two models are successfully implemented and trained in 150 epochs.

> It includes detailed explanation and results on the bottom of the description.
> Please click on the link to check entire report
## Guideline
- Download UCF101 dataset
- Run dataset.py to run the dataset preparation
### To check the GUI alication
- Run classification_app.py to check classification part
- Run Summarisation_app.py to check summarisation part
- First upload a .pth file to upload and load the trained model
- Upload input video to generate classified label or summarised video.
## Tech Stack
- Python
- PyTorch
- NumPy
- Pandas
# Results
| Metric                 | Value    |
|------------------------|----------|
| Training Loss          | 0.0606   |
| Validation Loss        | 0.0520   |
| Training Accuracy      | 98.65    |
| Validation Accuracy    | 98.97    |
| Training F1-score      | 0.9866   |
| Validation F1-score    | 0.9898   |
| Training Precision     | 0.9874   |
| Validation Precision   | 0.9907   |
| Training Recall        | 0.9865   |
| Validation Recall      | 0.9897   |

- To test and execute the model, two GUI applications were implemented. One supports classification task and the other for summarisation task.
- Classification Introduction
  - Upload model file
  - Upload video to classify or Click random video button to classify the input.
- Summarisation Introdcution
  - Upload model file
  - Upload input video to summarise certain classified categories from the input video.

More information about the project:
[Please check the following report](https://github.com/ImSH715/Video-Classification-and-Summarisation/blob/main/Report/Seunghyun%20Im%20-%20Dissertation.pdf)
