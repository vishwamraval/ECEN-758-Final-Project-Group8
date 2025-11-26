import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

class_names = [
    "banded",
    "blotchy",
    "braided",
    "bubbly",
    "bumpy",
    "chequered",
    "cobwebbed",
    "cracked",
    "crosshatched",
    "crystalline",
    "dotted",
    "fibrous",
    "flecked",
    "freckled",
    "frilly",
    "gauzy",
    "grid",
    "grooved",
    "honeycombed",
    "interlaced",
    "knitted",
    "lacelike",
    "lined",
    "marbled",
    "matted",
    "meshed",
    "paisley",
    "perforated",
    "pitted",
    "pleated",
    "polka-dotted",
    "porous",
    "potholed",
    "scaly",
    "smeared",
    "spiralled",
    "sprinkled",
    "stained",
    "stratified",
    "striped",
    "studded",
    "swirly",
    "veined",
    "waffled",
    "woven",
    "wrinkled",
    "zigzagged",
]
num_classes = len(class_names)


def build_resnet18(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.0),
        nn.ReLU(),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def load_model():
    model = build_resnet18(num_classes)
    state_dict = torch.load("resnet18_dtd_best.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


st.title("ResNet-18 Texture Classifier (DTD)")
st.write("Upload an image to classify its texture into one of 47 DTD classes.")

if st.button("Show DTD Class Names"):
    st.write(", ".join(class_names))


uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running inference with ResNet-18..."):
        model = load_model()
        x = test_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu()

        topk = torch.topk(probs, k=5)

        st.subheader("Top-5 Predictions")
        for i in range(5):
            idx = topk.indices[i].item()
            name = class_names[idx]
            conf = topk.values[i].item() * 100
            st.write(f"{i + 1}. **{name}** - {conf:.2f}%")
