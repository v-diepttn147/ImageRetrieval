# posts/search/embedding.py

import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# Load model once
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Preprocessing
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_embedding(image: Image.Image):
    """Takes a PIL image and returns its embedding as np.ndarray."""
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        vec = resnet(tensor).squeeze().numpy()
    return vec
