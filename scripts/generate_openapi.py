# scripts/generate_openapi.py
import json
import sys
import os

# Ensure we can import core
sys.path.append(os.getcwd())

from core.engine.api import app

def generate():
    openapi_data = app.openapi()

    # Save to root or docs
    with open("openapi.json", "w") as f:
        json.dump(openapi_data, f, indent=2)

    print("✅ Canonical OpenAPI Spec generated at openapi.json")

if __name__ == "__main__":
    generate()
