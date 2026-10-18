
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from app import model, transform, CLASSES

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "data" / "test"
REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

true_labels = []
predicted_labels = []
records = []

model.eval()

# Evaluate every image in each class folder
with torch.inference_mode():
    for true_class in CLASSES:
        folder = TEST_DIR / true_class

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        image_files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

        print(f"Evaluating {true_class}: {len(image_files)} images")

        for image_path in image_files:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            tensor = transform(image).unsqueeze(0)
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

            predicted_index = int(
                torch.argmax(probabilities).item()
            )

            predicted_class = CLASSES[predicted_index]
            confidence = float(probabilities[predicted_index])

            true_labels.append(true_class)
            predicted_labels.append(predicted_class)

            records.append({
                "image": image_path.name,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": round(confidence, 4),
                "correct": predicted_class == true_class
            })

if len(records) < 30:
    raise RuntimeError(
        "At least 30 held-out images are required."
    )

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)

report_text = classification_report(
    true_labels,
    predicted_labels,
    labels=CLASSES,
    zero_division=0,
    digits=4
)

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=CLASSES
)

# Save prediction results
csv_path = REPORT_DIR / "task7_evaluation_results.csv"

with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=records[0].keys()
    )
    writer.writeheader()
    writer.writerows(records)

# Save summary
summary = (
    "CSC6302 Practical 7 - Model Evaluation\n"
    f"Test images: {len(records)}\n"
    f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n"
    + report_text
)

summary_path = REPORT_DIR / "task7_evaluation_summary.txt"
summary_path.write_text(summary)

# Save confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=CLASSES
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("MobileNetV3 - Held-out Test Set")
plt.tight_layout()

figure_path = REPORT_DIR / "task7_confusion_matrix.png"
fig.savefig(figure_path, dpi=200)
plt.close(fig)

print("\n" + summary)
print("Saved:", csv_path)
print("Saved:", summary_path)
print("Saved:", figure_path)
