from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

# Load pretrained ResNet
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            return render_template("index.html", result="No file selected ❌")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            img = Image.open(filepath).convert("RGB")
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(img)
                probs = F.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)

                pred = pred.item()
                confidence = confidence.item() * 100

            if pred == 0:
                result = f"Original Image ✅ ({confidence:.2f}%)"
            else:
                result = f"Tampered Image ❌ ({confidence:.2f}%)"

            image_path = filepath

        except Exception as e:
            result = f"Error processing image ❌"

    return render_template("index.html", result=result, image_path=image_path)


if __name__ == "__main__":
    app.run()