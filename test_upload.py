import asyncio
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload():
    with open("test.pdf", "wb") as f:
        f.write(b"%PDF-1.4 fake pdf")

    with open("test.pdf", "rb") as f:
        response = client.post(
            "/documents/",
            files={"file": ("test.pdf", f, "application/pdf")},
            data={"kb_name": "Test KB"}
        )
    print("Response:", response.json())
    
    from app.config import get_config
    import os
    cfg = get_config()
    print("Files in", cfg.document_store_path / "files", ":")
    try:
        for p in (cfg.document_store_path / "files").iterdir():
            print(p, p.stat().st_size)
    except FileNotFoundError:
        print("Folder not created")

if __name__ == "__main__":
    test_upload()
