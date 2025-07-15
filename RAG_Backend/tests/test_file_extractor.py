import importlib.util
import os

# Dynamically import test2.py from the app directory
file_extractor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../app/test2.py'))
spec = importlib.util.spec_from_file_location("test2", file_extractor_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load test2.py from {file_extractor_path}")
test2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test2)

def main():
    # Use the provided test file path
    test_file = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\RAG_Backend\inputs\2506.23338v1.pdf"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}\nPlease update the path in test_file_extractor.py.")
        return
    result = test2.extract_structured_content_sync(test_file)
    print("Extraction result:")
    print(result)

if __name__ == "__main__":
    main()