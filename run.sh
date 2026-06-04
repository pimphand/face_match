#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Virtual environment tidak ditemukan"
    echo "Buat terlebih dahulu:"
    echo "python3 -m venv venv"
    exit 1
fi

source venv/bin/activate

echo "Menjalankan Face Compare API..."
echo "Swagger: http://localhost:8000/docs"
echo "OpenAPI : http://localhost:8000/openapi.json"

uvicorn app:app --host 0.0.0.0 --port 8000