from PyPDF2 import PdfReader

pdf_path = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\RAG_Backend\inputs\2506.23338v1.pdf"
reader = PdfReader(pdf_path)
print(f"Number of pages: {len(reader.pages)}")
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    print(f"Page {i+1} text length: {len(text) if text else 0}")
    if not text:
        print(f"Warning: No text extracted from page {i+1}")