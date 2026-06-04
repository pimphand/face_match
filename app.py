import os
import cv2
import numpy as np
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from insightface.app import FaceAnalysis

app = FastAPI(
    title="Face Compare API",
    version="1.0.0",
    description="Compare two faces using InsightFace"
)

face_app = FaceAnalysis(
    providers=["CPUExecutionProvider"]
)

face_app.prepare(ctx_id=0)


def get_embedding(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise Exception("Cannot read image")

    faces = face_app.get(img)

    if len(faces) == 0:
        raise Exception("No face detected")

    return faces[0].embedding


def cosine_similarity(v1, v2):
    return float(
        np.dot(v1, v2)
        / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/compare")
async def compare_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):

    tmp1 = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    tmp2 = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    try:

        tmp1.write(await image1.read())
        tmp2.write(await image2.read())

        tmp1.close()
        tmp2.close()

        emb1 = get_embedding(tmp1.name)
        emb2 = get_embedding(tmp2.name)

        score = cosine_similarity(
            emb1,
            emb2
        )

        same_person = score >= 0.45

        return {
            "similarity": round(score, 4),
            "same_person": same_person
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    finally:

        if os.path.exists(tmp1.name):
            os.remove(tmp1.name)

        if os.path.exists(tmp2.name):
            os.remove(tmp2.name)