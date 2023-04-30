from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

class STDInLoader(BaseLoader):
    """Load text files."""

    def __init__(self, text: str):
        """Initialize with file path."""
        self.text = text

    def load(self) -> List[Document]:
        """Load from file path."""
        metadata = {"source": "stdin"}
        return [Document(page_content=self.text, metadata=metadata)]
