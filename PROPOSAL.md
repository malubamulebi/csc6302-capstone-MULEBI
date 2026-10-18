
# CSC6302 Practical 7 — Capstone Proposal

**Project:** Landscape Image Classification Using Transfer Learning

**Student:** Maluba Mulebi  
**Computer Number:** 2025025124  
**Programme:** MSc Artificial Intelligence  
**Institution:** University of Zambia  
**Submission:** Individual

## 1. Problem Statement

Image classification can support the organisation and preliminary analysis of environmental photographs. This project develops a web application that classifies user-uploaded images into three categories: buildings, forest and sea.

The application demonstrates how transfer learning can support accessible computer vision applications using limited computational resources.

## 2. Dataset Plan

The project reuses a three-class subset of the Intel Image Classification dataset prepared during Practical 4.

The existing dataset contains separate training, validation and test partitions. Evaluation uses 135 test images, comprising 45 images per class.

## 3. Model Plan

The application uses a fine-tuned MobileNetV3-Small convolutional neural network developed during Practical 4.

Input images are resized to 224 × 224 pixels and normalised using ImageNet statistics.

Predictions include class probabilities, confidence scores and plain-language explanations. A confidence threshold is used to indicate uncertainty.

## 4. Implementation and Deployment

The application was developed using Python, PyTorch and Streamlit.

GitHub provides version control, while GitHub Actions performs automated linting and testing. Streamlit Community Cloud hosts the application and automatically rebuilds it following updates to the connected GitHub branch.

## 5. Evaluation Plan

Performance is assessed using accuracy, precision, recall, F1-score and a confusion matrix on held-out test images.

The current evaluation reports 100% accuracy on 135 images. Further testing on unfamiliar photographs is necessary to assess real-world generalisation.

## 6. Ethics and Limitations

The system is an educational demonstration and must not be used for safety-critical or environmental enforcement decisions.

Potential limitations include dataset bias, incorrect predictions, lighting variations and poor performance on images outside the three supported classes.

## 7. Individual Contribution

The student completed the implementation, model integration, evaluation, testing, deployment and documentation individually.

The project reuses Practical 4 model weights and acknowledges open-source libraries and AI-assisted development.
