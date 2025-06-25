import streamlit as st
from PIL import Image
import pickle
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import io
import base64

# Load saved embeddings
with open("all_embeddings.pkl", "rb") as f:
    all_vectors = pickle.load(f)

# Extract embeddings and metadata
ids = [item[0] for item in all_vectors]
embeddings = [item[1] for item in all_vectors]
metadata = [item[2] for item in all_vectors]

# Initialize face detector and embedder
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Compute cosine similarity
def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Find top-k similar faces
def find_similar(embedding, top_k=5):
    scores = [cosine_similarity(embedding, vec) for vec in embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [metadata[i] for i in top_indices]

# Display results
def display_result(results):
    st.subheader("Top Matching Faces:")
    cols = st.columns(len(results))
    for i, meta in enumerate(results):
        with cols[i]:
            img_url = f"https://image.tmdb.org/t/p/h632/{meta['profile_path']}"
            st.image(img_url, width=150, caption=meta['name'])

# Streamlit UI
st.title("🎭 Similar Face Search Engine")

uploaded_file = st.file_uploader("Upload an image to search for similar faces", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # Detect and embed face
    face = mtcnn(image)
    if face is not None:
        aligned = face.unsqueeze(0).to(device)
        emb = resnet(aligned).detach().cpu().squeeze().tolist()

        results = find_similar(emb, top_k=5)
        display_result(results)
    else:
        st.warning("No face detected in the image. Please try another one.")
