#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Virtual environment tidak ditemukan. Membuat venv..."
    python3 -m venv venv
fi

source venv/bin/activate

if ! command -v uvicorn >/dev/null 2>&1; then
    echo "Dependensi belum terpasang. Menginstall dari requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo "Menjalankan Face Compare API..."
echo "Swagger: http://localhost:8000/docs"
echo "OpenAPI : http://localhost:8000/openapi.json"

python -m uvicorn app:app --host 0.0.0.0 --port 8000