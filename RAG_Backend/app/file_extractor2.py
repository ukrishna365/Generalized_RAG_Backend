import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import json
from typing import Dict, Any, Optional
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURABLE PATHS (edit these as needed) ---
INPUT_FOLDER = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\RAG_Backend\inputs"
OUTPUT_FOLDER = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\RAG_Backend\outputs"
MAX_WORKERS = 3 # Adjust based on your CPU
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.md'}
# --------------------------------------------------

def find_supported_files(input_folder):
    files = []
    for root, dirs, filenames in os.walk(input_folder):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, filename))
    return files

def extract_structured_content_sync(filepath: str, output_dir: str = OUTPUT_FOLDER) -> Optional[Dict[str, Any]]:
    import asyncio
    try:
        # Import here to avoid issues with multiprocessing
        from RAG_Backend.app.file_extractor import extract_structured_content
        return asyncio.run(extract_structured_content(filepath, output_dir))
    except Exception as e:
        print(f"Error extracting {filepath}: {e}")
        return None

def process_file(file_path, output_dir):
    try:
        result = extract_structured_content_sync(file_path, output_dir)
        return (file_path, True if result else False)
    except Exception as e:
        print(f"Exception in process_file for {file_path}: {e}")
        return (file_path, False)

def parallel_batch_extract(input_folder: str, output_folder: str, max_workers: int = MAX_WORKERS):
    files = find_supported_files(input_folder)
    if not files:
        print(f"No supported files found in {input_folder}.")
        return
    print(f"Found {len(files)} files to process.")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file_path, output_folder): file_path for file_path in files}
        for i, future in enumerate(as_completed(future_to_file), 1):
            file_path = future_to_file[future]
            try:
                file_path, success = future.result()
                status = "Success" if success else "Failed"
            except Exception as exc:
                status = f"Exception: {exc}"
            print(f"[{i}/{len(files)}] {file_path}: {status}")
            results.append((file_path, status))
    print("\nSummary:")
    for file_path, status in results:
        print(f"{file_path}: {status}")

def safe_equation(equation: str) -> str:
    """Escape backslashes and control characters for JSON safety."""
    if not isinstance(equation, str):
        return equation
    return equation.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel batch extract files from a folder using RAG-Anything.")
    parser.add_argument('--input-folder', type=str, default=INPUT_FOLDER, help='Folder containing files to extract')
    parser.add_argument('--output-folder', type=str, default=OUTPUT_FOLDER, help='Folder to save outputs')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, help='Number of parallel workers')
    args = parser.parse_args()
    parallel_batch_extract(args.input_folder, args.output_folder, args.max_workers) 