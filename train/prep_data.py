import os
import json

# Attempt to load .env file manually
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

def compile_repo_jsonl(repo_path: str, output_jsonl: str) -> list:
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

def compile_repo_txt(repo_path: str, output_text: str) -> list:
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
    if not repo:
        raise SystemExit("[-] Error: PATH_TO_TRAINING_CODE environment variable is not set.")

    # Relative paths are resolved from this script's directory (where .env lives)
    repo = os.path.expanduser(repo.strip('"\''))
    if not os.path.isabs(repo):
        repo = os.path.abspath(os.path.join(script_dir, repo))
    if not os.path.isdir(repo):
        raise SystemExit(f"[-] Error: PATH_TO_TRAINING_CODE is set to '{repo}', but this is not a valid directory.")

    dataset = compile_repo_txt(repo, "train.txt")
    print(f"Dataset created with {len(dataset)} source files.")
