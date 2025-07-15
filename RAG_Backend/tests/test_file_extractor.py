import importlib.util
import os

# Dynamically import file_extractor.py from the app directory
file_extractor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/file_extractor.py'))
spec = importlib.util.spec_from_file_location("file_extractor", file_extractor_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load file_extractor.py from {file_extractor_path}")
file_extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(file_extractor)

def main():
    # TODO: Update this path to point to a real test file (PDF, DOCX, PPTX, or MD)
    test_file = "C:\\Users\\kuppa\\DS_Projects\\Generalized_RAG_Backend3\\rag_storage\\2506.23338v1.pdf"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}\nPlease update the path in test_file_extractor.py.")
        return
    result = file_extractor.extract_structured_content_sync(test_file)
    print("Extraction result:")
    print(result)

if __name__ == "__main__":
    main()              