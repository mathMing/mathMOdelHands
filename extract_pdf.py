import pypdf
import os

pdf_path = "problem.pdf"
if not os.path.exists(pdf_path):
    print("problem.pdf not found!")
    exit(1)

reader = pypdf.PdfReader(pdf_path)
print(f"Total pages: {len(reader.pages)}")

with open("problem_extracted_text.txt", "w", encoding="utf-8") as f:
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        header = f"\n\n==================== PAGE {i+1} ====================\n\n"
        print(f"Page {i+1} length: {len(text)}")
        f.write(header + text)

print("Saved to problem_extracted_text.txt")
