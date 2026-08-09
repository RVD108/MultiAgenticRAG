"""Shared utility functions and global config loader."""

import hashlib
import uuid
from typing import Any, Literal, Optional, Union
import yaml
from langchain_core.documents import Document


def load_config(file_path: str = "./config.yaml") -> dict:
    """Load and return the YAML configuration file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def new_uuid() -> str:
    """Generate a new random UUID string."""
    return str(uuid.uuid4())


def _generate_uuid(page_content: str) -> str:
    """Deterministic UUID based on MD5 hash of page content."""
    md5_hash = hashlib.md5(page_content.encode()).hexdigest()
    return str(uuid.UUID(md5_hash))


def reduce_docs(
    existing: Optional[list[Document]],
    new: Union[
        list[Document],
        list[dict[str, Any]],
        list[str],
        str,
        Literal["delete"],
    ],
) -> list[Document]:
    """Merge new documents into the existing list, deduplicating by UUID.

    Supports:
    - ``"delete"`` — clears all existing documents.
    - ``str`` — wraps in a Document.
    - ``list[str | dict | Document]`` — upserts by UUID.
    """
    if new == "delete":
        return []

    existing_list = list(existing) if existing else []
    if isinstance(new, str):
        return existing_list + [
            Document(page_content=new, metadata={"uuid": _generate_uuid(new)})
        ]

    new_list: list[Document] = []
    if isinstance(new, list):
        existing_ids = {doc.metadata.get("uuid") for doc in existing_list}
        for item in new:
            if isinstance(item, str):
                item_id = _generate_uuid(item)
                if item_id not in existing_ids:
                    new_list.append(Document(page_content=item, metadata={"uuid": item_id}))
                    existing_ids.add(item_id)
            elif isinstance(item, dict):
                metadata = item.get("metadata", {})
                item_id = metadata.get("uuid") or _generate_uuid(item.get("page_content", ""))
                if item_id not in existing_ids:
                    new_list.append(
                        Document(**{**item, "metadata": {**metadata, "uuid": item_id}})
                    )
                    existing_ids.add(item_id)
            elif isinstance(item, Document):
                item_id = item.metadata.get("uuid", "")
                if not item_id:
                    item_id = _generate_uuid(item.page_content)
                    new_item = item.copy(deep=True)
                    new_item.metadata["uuid"] = item_id
                else:
                    new_item = item
                if item_id not in existing_ids:
                    new_list.append(new_item)
                    existing_ids.add(item_id)

    return existing_list + new_list


# Global config (loaded once at import time)
config = load_config()
