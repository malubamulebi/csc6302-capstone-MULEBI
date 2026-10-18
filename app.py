
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import ImageFilter
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "task2_finetuned_model.pth"

CLASSES = ["buildings", "forest", "sea"]
CONFIDENCE_THRESHOLD = 0.60

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(
        model.classifier[3].in_features,
        len(CLASSES)
    )

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model


model = load_model()


def classify_image(image):
    if image is None:
        return {}, "Please upload an image."

    try:
        image = image.convert("RGB")

        if min(image.size) < 32:
            return {}, "Image is too small. Upload a clearer image."

        # Detect extremely blurry images
        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        sharpness = float(np.asarray(edges).var())

        if sharpness < 5:
            return {}, "Image appears blurry or featureless. Try another photo."

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        scores = {
            name: float(probabilities[i])
            for i, name in enumerate(CLASSES)
        }

        best_class = max(scores, key=scores.get)
        confidence = scores[best_class]

        if confidence < CONFIDENCE_THRESHOLD:
            explanation = (
                "The model is unsure about this image. "
                "Try a clearer photograph showing a building, "
                "forest, or sea."
            )
        else:
            explanation = (
                f"The image most closely resembles '{best_class}' "
                f"with {confidence:.1%} model confidence. "
                "This is an automated prediction, not a verified fact."
            )

        return scores, explanation

    except Exception:
        return {}, "The image could not be processed. Try a valid JPG or PNG."


with gr.Blocks(title="Landscape Image Classifier") as demo:
    gr.Markdown(
        """
        # Landscape Image Classifier

        **CSC6302 Practical 7 — Computer Vision Capstone**

        Upload a photograph to classify it as **Buildings, Forest, or Sea**.

        **Limitations:** This application is a demonstration.
        It must not be used for safety-critical decisions,
        environmental enforcement, or identifying individuals.
        """
    )

    with gr.Row():
        image_input = gr.Image(
            type="pil",
            label="Upload an image"
        )

        with gr.Column():
            prediction = gr.Label(
                num_top_classes=3,
                label="Model predictions"
            )
            explanation = gr.Textbox(
                label="Explanation",
                lines=4
            )

    submit = gr.Button("Classify Image")

    submit.click(
        fn=classify_image,
        inputs=image_input,
        outputs=[prediction, explanation]
    )


if __name__ == "__main__":
    demo.launch()
