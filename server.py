import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile
from PIL import Image
from torchvision import transforms
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define your model (same as training)
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

model = SimpleNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Same preprocessing as training (without rotation)
transform = transforms.Compose([
    transforms.Grayscale(),       # ensure single channel
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.post("/predict")
async def predict(file: UploadFile):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))

    img = transform(img).unsqueeze(0)  # shape: [1,1,28,28]

    with torch.no_grad():
        output = model(img)
        prediction = torch.argmax(output, 1).item()

    return {"prediction": prediction}
