
# CSC6302 Practical 7 — Computer Vision Capstone

**Student:** Maluba Mulebi  
**Computer Number:** 2025025124  
**Programme:** MSc Artificial Intelligence, University of Zambia

## Project Overview

A web-based image classification application using a fine-tuned MobileNetV3-Small model. Users upload photographs and receive predictions for three categories: buildings, forest, and sea.

The project reuses the transfer-learning model developed in Practical 4.

## Model and Evaluation

- Architecture: MobileNetV3-Small
- Input resolution: 224 × 224
- Classes: Buildings, Forest, Sea
- Test dataset: 135 images (45 per class)
- Recorded test accuracy: 100%
- Macro F1-score: 1.00

These results describe the supplied test set and do not guarantee equivalent performance on unfamiliar images.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Automated Testing

```bash
python -m pytest -q
python -m ruff check . --select E4,E7,E9,F --ignore E402
```

The project includes seven automated tests. GitHub Actions runs lint checks and tests on pushes and pull requests.

## Deployment

The application is deployed using Streamlit Community Cloud, connected to the GitHub main branch.

## Limitations and Ethics

The classifier only recognises three categories. It should not be used for environmental enforcement, safety-critical decisions or identifying individuals.

Performance may vary with image quality, lighting, unfamiliar environments and domain differences.

## Development Challenges

Hugging Face Gradio hosting required a paid subscription during deployment. Streamlit Community Cloud was used as an alternative.

Initial Streamlit deployment also encountered PyTorch dependency incompatibilities, which were addressed by updating package versions.

## Acknowledgements

This project builds upon Practical 4. PyTorch, torchvision, Streamlit, scikit-learn and other open-source libraries were used. AI assistance supported coding, debugging and documentation.

**Contribution:** Individual implementation by Maluba Mulebi.
