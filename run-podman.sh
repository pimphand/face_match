#!/bin/bash
# Jalankan Face Compare API di Podman (2 CPU, 4 GB RAM)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="face-compare:latest"
CONTAINER_NAME="face-compare"
HOST_PORT="${HOST_PORT:-8000}"
BUILD=1

usage() {
    cat <<'EOF'
Usage: ./run-podman.sh [OPTIONS]

  Jalankan API di Podman dengan limit 2 CPU dan 4 GB RAM.

Options:
  --no-build    Lewati build image (gunakan image yang sudah ada)
  -h, --help    Tampilkan bantuan ini

Environment:
  HOST_PORT     Port host (default: 8000)

Contoh:
  ./run-podman.sh
  ./run-podman.sh --no-build
  HOST_PORT=9000 ./run-podman.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) BUILD=0 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Opsi tidak dikenal: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

if ! command -v podman >/dev/null 2>&1; then
    echo "Error: podman tidak ditemukan. Install podman terlebih dahulu."
    exit 1
fi

if [[ "$BUILD" -eq 1 ]]; then
    echo "Membangun image (slim, headless OpenCV, validasi wajah)..."
    podman build -t "$IMAGE_NAME" -f Containerfile .
else
    echo "Melewati build (--no-build)."
    if ! podman image exists "$IMAGE_NAME" 2>/dev/null; then
        echo "Error: image $IMAGE_NAME belum ada. Jalankan tanpa --no-build dulu."
        exit 1
    fi
fi

podman rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Menjalankan container (2 CPU, 4 GB RAM, port $HOST_PORT)..."
podman run -d \
    --name "$CONTAINER_NAME" \
    --cpus 2 \
    --memory 4g \
    --memory-swap 4g \
    --restart unless-stopped \
    -p "${HOST_PORT}:8000" \
    "$IMAGE_NAME"

echo "Menunggu API siap (model InsightFace dimuat)..."
READY=0
for _ in $(seq 1 30); do
    if curl -sf "http://localhost:${HOST_PORT}/" >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 2
done

if [[ "$READY" -ne 1 ]]; then
    echo "Peringatan: API belum merespons. Cek log: podman logs -f $CONTAINER_NAME"
else
    echo "API siap."
fi

echo ""
echo "=== Face Compare API (Podman) ==="
echo "Swagger : http://localhost:${HOST_PORT}/docs"
echo "OpenAPI : http://localhost:${HOST_PORT}/openapi.json"
echo "Validate: POST http://localhost:${HOST_PORT}/validate-face"
echo "Compare : POST http://localhost:${HOST_PORT}/compare"
echo ""
echo "Validasi gambar (image1 & image2):"
echo "  - Tepat 1 wajah terdeteksi"
echo "  - Skor deteksi >= 0.85 (tidak blur)"
echo "  - Area wajah >= 5% area gambar (pas foto)"
echo ""
echo "Contoh error di Swagger: Responses -> 400 pada POST /compare"
echo ""
echo "Log   : podman logs -f $CONTAINER_NAME"
echo "Stop  : podman stop $CONTAINER_NAME"
echo "Hapus : podman rm -f $CONTAINER_NAME"
echo "Rebuild: ./run-podman.sh"
