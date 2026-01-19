import os

PDF_DIR = os.getenv("PDF_DIR", "/home/vrai/Leo/GenAI-VR-teaching/shared/knowledge")

PDF_MAP = {
    "pdf1": os.path.join(PDF_DIR, "pdf1.pdf"),
    "pdf2": os.path.join(PDF_DIR, "pdf2.pdf"),
    "pdf3": os.path.join(PDF_DIR, "pdf3.pdf"),
}
