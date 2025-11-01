from docling.document_converter import DocumentConverter
from pathlib import Path


def read(file_path: str) -> str:
    """
    Read a document file and extract text content.

    Args:
        file_path: path to the document file

    Returns:
        Extracted text content from the document file.
    """
    
    # Define the path to your document
    input_doc_path = Path(file_path)
    # Initialize the DocumentConverter
    doc_converter = DocumentConverter() 
    # Convert the document
    conv_res = doc_converter.convert(input_doc_path)
    plain_text = conv_res.document.export_to_text()
    return plain_text.strip()
