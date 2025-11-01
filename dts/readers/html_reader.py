from bs4 import BeautifulSoup

def read(file_path: str) -> str:
    """
    Read an HTML file and extract visible text content.

    Args:
        file_path: path to the .html file

    Returns:
        Clean text string with whitespace normalized.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, noscript tags etc.
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    # Get text
    text = soup.get_text(separator=" ")

    # Normalize whitespace
    text = " ".join(text.split())

    return text