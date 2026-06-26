import os
import json

def compile_repo_jsonl(repo_path: str, output_jsonl: dict) -> str:
    dataset = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.go') and not file.endswith('_test.go'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_content = f.read()

                    dataset.append({"text": f"// File: {file}\n\n{code_content}"})
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
    
    with open(output_jsonl, 'w', encoding='utf-8') as out:
        for entry in dataset:
            out.write(json.dumps(entry) + '\n')

    return dataset

def compile_repo_txt(repo_path: str, output_text: dict) -> str:
    dataset = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.go') and not file.endswith('_test.go'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_content = f.read()

                    dataset.append({"text": f"// File: {file}\n\n{code_content}"})
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
    
    with open(output_text, 'w', encoding='utf-8') as out:
        for entry in dataset:
            out.write(entry["text"] + '\n\n')

    return dataset

if __name__ == "__main__":
    # Point this to your local repo path
    repo = os.getenv("PATH_TO_TRAINING_CODE")
    dataset = compile_repo_txt(repo, "train.txt")
    print(f"Dataset created with {len(dataset)} source files.")