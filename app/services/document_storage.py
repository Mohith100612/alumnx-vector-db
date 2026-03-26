import shutil
from pathlib import Path
from typing import BinaryIO
from app.config import get_config

class DocumentStorageBackend:
    def save(self, file_id: str, file_name: str, content: bytes) -> str:
        raise NotImplementedError

    def get_path(self, file_id: str) -> Path:
        raise NotImplementedError

    def delete(self, file_id: str):
        raise NotImplementedError

class LocalDocumentStorage(DocumentStorageBackend):
    def __init__(self):
        self.base_path = get_config().document_store_path / "files"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, file_id: str, file_name: str, content: bytes) -> str:
        extension = Path(file_name).suffix
        file_path = self.base_path / f"{file_id}{extension}"
        with file_path.open("wb") as f:
            f.write(content)
        return str(file_path)

    def get_path(self, file_id: str) -> Path:
        for child in self.base_path.iterdir():
            if child.is_file() and child.stem == file_id:
                return child
        raise FileNotFoundError(f"File for ID {file_id} not found.")

    def delete(self, file_id: str):
        try:
            path = self.get_path(file_id)
            path.unlink()
        except FileNotFoundError:
            pass

def get_storage_backend() -> DocumentStorageBackend:
    return LocalDocumentStorage()
