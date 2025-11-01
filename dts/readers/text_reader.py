def read(file_path: str) -> str:
    """
    Read a text file and extract its content.

    Args:
        file_path: path to the .txt file    
    Returns:
        Extracted text content from the text file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.strip() 

