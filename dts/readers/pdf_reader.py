import pymupdf  # PyMuPDF

def read(file_path: str) -> str:
    """
    Read a PDF file and extract text content.

    Args:
        file_path: path to the .pdf file

    Returns:
        Extracted text content from the PDF file.
    """
    text = ""
    with pymupdf.open(file_path) as doc:
        for page in doc:
            text += page.get_text()  + "\n"
    return text.strip()
