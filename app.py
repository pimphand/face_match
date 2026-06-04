import os
import cv2
import numpy as np
import tempfile
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException
from insightface.app import FaceAnalysis
from pydantic import BaseModel, Field

FACE_VALIDATION_RULES = """
| Aturan | Kriteria | Pesan error (HTTP 400) |
|--------|----------|------------------------|
| Wajah manusia | Model deteksi wajah asli (bukan ilustrasi/statue) | `Gambar tidak valid: tidak terdeteksi wajah manusia` |
| Satu wajah saja | Maksimal 1 wajah | `Gambar tidak valid: lebih dari satu wajah terdeteksi` |
| Kejelasan wajah | Skor deteksi ≥ 0.85 | `Gambar tidak valid: wajah terlalu blur / tidak jelas` |
| Ukuran wajah | Area wajah ≥ 5% area gambar | `Gambar tidak valid: wajah terlalu kecil (bukan pas foto)` |
"""

FACE_VALIDATION_DESCRIPTION = f"""
Setiap gambar (`image1`, `image2`) divalidasi sebelum dibandingkan:

{FACE_VALIDATION_RULES}

Gunakan foto pas (close-up), satu orang, tidak blur.
"""

CHECK_FACE_DESCRIPTION = f"""
Memeriksa apakah foto berisi **wajah manusia asli** (foto/live), bukan gambar yang hanya menyerupai manusia
(ilustrasi, kartun, patung, topeng, atau cetakan wajah).

{FACE_VALIDATION_RULES}

Upload satu foto pas (close-up) untuk pengecekan.
"""


class ValidateFaceResponse(BaseModel):
    valid: bool = Field(
        ...,
        examples=[True],
        description="`true` jika foto lulus semua aturan validasi wajah manusia.",
    )
    message: str = Field(
        ...,
        examples=["Wajah manusia terdeteksi"],
        description="Pesan hasil pengecekan.",
    )
    det_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        examples=[0.9234],
        description="Skor kepercayaan deteksi wajah manusia (semakin tinggi semakin yakin).",
    )
    face_area_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        examples=[0.32],
        description="Perbandingan area wajah terhadap area gambar.",
    )


class CompareResponse(BaseModel):
    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        examples=[0.8234],
        description="Skor cosine similarity antara kedua wajah (0 = berbeda, 1 = identik).",
    )
    same_person: bool = Field(
        ...,
        examples=[True],
        description="`true` jika similarity ≥ 0.45 (dianggap orang yang sama).",
    )


class ErrorResponse(BaseModel):
    detail: str = Field(
        ...,
        examples=["Gambar tidak valid: tidak terdeteksi wajah manusia"],
    )


VALIDATION_ERROR_RESPONSES = {
    400: {
        "description": "Validasi gambar gagal atau file tidak dapat dibaca",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "examples": {
                    "no_face": {
                        "summary": "Tidak ada wajah manusia",
                        "value": {
                            "detail": "Gambar tidak valid: tidak terdeteksi wajah manusia"
                        },
                    },
                    "multiple_faces": {
                        "summary": "Lebih dari satu wajah",
                        "value": {
                            "detail": "Gambar tidak valid: lebih dari satu wajah terdeteksi"
                        },
                    },
                    "blur": {
                        "summary": "Wajah blur / tidak jelas",
                        "value": {
                            "detail": "Gambar tidak valid: wajah terlalu blur / tidak jelas"
                        },
                    },
                    "face_too_small": {
                        "summary": "Wajah terlalu kecil",
                        "value": {
                            "detail": "Gambar tidak valid: wajah terlalu kecil (bukan pas foto)"
                        },
                    },
                    "cannot_read": {
                        "summary": "File gambar rusak",
                        "value": {"detail": "Cannot read image"},
                    },
                }
            }
        },
    },
}


app = FastAPI(
    title="Face Compare API",
    version="1.0.0",
    description=(
        "API perbandingan wajah menggunakan InsightFace. "
        "Gunakan `POST /validate-face` untuk cek wajah manusia pada satu foto, "
        "atau `POST /compare` untuk membandingkan dua foto."
    ),
)

face_app = FaceAnalysis(
    providers=["CPUExecutionProvider"]
)

face_app.prepare(ctx_id=0)


def validate_face(image, faces):
    if len(faces) == 0:
        raise Exception("Gambar tidak valid: tidak terdeteksi wajah manusia")

    if len(faces) > 1:
        raise Exception("Gambar tidak valid: lebih dari satu wajah terdeteksi")

    face = faces[0]

    if face.det_score < 0.85:
        raise Exception("Gambar tidak valid: wajah terlalu blur / tidak jelas")

    x1, y1, x2, y2 = face.bbox.astype(int)
    h, w = image.shape[:2]

    face_area = (x2 - x1) * (y2 - y1)
    image_area = h * w

    if face_area / image_area < 0.05:
        raise Exception("Gambar tidak valid: wajah terlalu kecil (bukan pas foto)")

    return face


def detect_face(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise Exception("Cannot read image")

    faces = face_app.get(img)
    face = validate_face(img, faces)

    x1, y1, x2, y2 = face.bbox.astype(int)
    h, w = img.shape[:2]
    face_area_ratio = (x2 - x1) * (y2 - y1) / (h * w)

    return {
        "det_score": round(float(face.det_score), 4),
        "face_area_ratio": round(float(face_area_ratio), 4),
    }


def get_embedding(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise Exception("Cannot read image")

    faces = face_app.get(img)
    validate_face(img, faces)

    return faces[0].embedding


async def save_upload_to_temp(upload: UploadFile) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        tmp.write(await upload.read())
        tmp.close()
        return tmp.name
    except Exception:
        tmp.close()
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        raise


def http_error_from_exception(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def cosine_similarity(v1, v2):
    return float(
        np.dot(v1, v2)
        / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )


@app.get("/", summary="Health check")
def root():
    return {"status": "running"}


@app.post(
    "/validate-face",
    response_model=ValidateFaceResponse,
    summary="Cek wajah manusia pada foto",
    description=CHECK_FACE_DESCRIPTION,
    responses=VALIDATION_ERROR_RESPONSES,
)
async def validate_face_endpoint(
    image: Annotated[
        UploadFile,
        File(description="Foto untuk dicek (JPEG/PNG). Satu wajah manusia asli, pas foto, tidak blur."),
    ],
):
    path = await save_upload_to_temp(image)

    try:
        metrics = detect_face(path)
        return {
            "valid": True,
            "message": "Wajah manusia terdeteksi",
            **metrics,
        }
    except Exception as e:
        raise http_error_from_exception(e) from e
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post(
    "/compare",
    response_model=CompareResponse,
    summary="Bandingkan dua wajah",
    description=FACE_VALIDATION_DESCRIPTION,
    responses=VALIDATION_ERROR_RESPONSES,
)
async def compare_faces(
    image1: Annotated[
        UploadFile,
        File(description="Foto pertama (JPEG/PNG). Satu wajah, pas foto, tidak blur."),
    ],
    image2: Annotated[
        UploadFile,
        File(description="Foto kedua (JPEG/PNG). Satu wajah, pas foto, tidak blur."),
    ],
):

    path1 = await save_upload_to_temp(image1)
    path2 = await save_upload_to_temp(image2)

    try:
        emb1 = get_embedding(path1)
        emb2 = get_embedding(path2)

        score = cosine_similarity(emb1, emb2)
        same_person = score >= 0.45

        return {
            "similarity": round(score, 4),
            "same_person": same_person,
        }

    except Exception as e:
        raise http_error_from_exception(e) from e

    finally:
        for path in (path1, path2):
            if os.path.exists(path):
                os.remove(path)