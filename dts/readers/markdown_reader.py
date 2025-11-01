import markdown2
import html2text

def read(file_path: str) -> str:
    """
    Read a Markdown file and convert it to plain text.

    Args:
        file_path: path to the .md file 
    Returns:
        Converted plain text from the Markdown file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    # Convert Markdown to HTML
    html_content = markdown2.markdown(markdown_content)

    # Convert HTML to plain text
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True
    plain_text = text_maker.handle(html_content)

    return plain_text.strip()
