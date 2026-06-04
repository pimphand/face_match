import cv2
import numpy as np

from insightface.app import FaceAnalysis


# Load model
app = FaceAnalysis(
    providers=['CPUExecutionProvider']
)

app.prepare(ctx_id=0)


def get_embedding(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise Exception(f"Gagal membuka {image_path}")

    faces = app.get(img)

    if len(faces) == 0:
        raise Exception(
            f"Tidak ditemukan wajah pada {image_path}"
        )

    return faces[0].embedding


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) *
        np.linalg.norm(v2)
    )


# Ambil embedding
emb1 = get_embedding("known_faces/faisal.jpeg")
emb2 = get_embedding("image copy 2.png")

# Hitung similarity
score = cosine_similarity(emb1, emb2)

print(f"Similarity Score: {score:.4f}")

# Threshold
THRESHOLD = 0.45

if score >= THRESHOLD:
    print("✅ Kemungkinan ORANG YANG SAMA")
else:
    print("❌ Kemungkinan ORANG BERBEDA")