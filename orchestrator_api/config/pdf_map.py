import os

PDF_DIR = os.getenv("PDF_DIR")

PDF_MAP = {
    "sistema_solare": os.path.join(PDF_DIR, "sistema_solare.pdf"),
    "avanzamenti_800_900": os.path.join(PDF_DIR, "avanzamenti_800_900.pdf"),
    "sviluppo_cervello": os.path.join(PDF_DIR, "sviluppo_cervello.pdf"),
    "educazione_alimentare": os.path.join(PDF_DIR, "educazione_alimentare.pdf")
}
