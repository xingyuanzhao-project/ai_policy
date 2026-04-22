import sys
import zipfile
import re


def extract_text(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    text = re.sub(r"<w:p[^>]*>", "\n", xml)
    text = re.sub(r"<w:tab[^/]*/>", "\t", text)
    text = re.sub(r"<w:br[^/]*/>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&apos;", "'")
    return text


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for p in sys.argv[1:]:
        print(f"===== {p} =====")
        print(extract_text(p))
        print()
