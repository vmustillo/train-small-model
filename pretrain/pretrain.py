import os
import glob
import json
import concurrent.futures
import multiprocessing
import numpy as np
import tiktoken

# Attempt to load .env file manually
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

# --- Configuration ---

# Supported source file extensions for code training
SUPPORTED_EXTENSIONS = {'.go', '.py', '.js', '.ts', '.rs', '.cpp', '.h', '.java'}
OUTPUT_DIR = "./pretokenized_data"
VAL_PERCENT = 0.10  # 10% validation split

def process_single_file(file_path):
    """
    Reads a single file, tokenizes it using tiktoken, 
    and returns a list of token integers.
    """
    try:
        # Determine relative filename to preserve context conditioning
        relative_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code_content = f.read()
        
        # Format with our "Copilot style" header
        formatted_text = f"// File: {relative_name}\n\n{code_content}\n\n"
        
        # Encode using the cl100k_base (GPT-4) tokenizer
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(formatted_text, allowed_special={"<|endoftext|>"})
        return tokens
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def process_repository(repo_path):
    """
    Scans a repository and processes its files in parallel 
    using a CPU worker pool.
    """
    print(f"[+] Scanning repository: {repo_path}")
    
    # Gather all candidate code files
    all_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                all_files.append(os.path.join(root, file))
                
    print(f"[+] Found {len(all_files)} target code files in {os.path.basename(repo_path)}")
    
    # Process files concurrently using all available CPU cores
    num_workers = multiprocessing.cpu_count()
    print(f"[+] Spawning {num_workers} CPU workers for tokenization...")
    
    repo_tokens = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map file processing tasks to our worker processes
        results = executor.map(process_single_file, all_files)
        for result in results:
            if result:
                repo_tokens.extend(result)
                
    return repo_tokens

def main():
    repos_folder = os.getenv("PATH_TO_REPOS_FOLDER")
    if not repos_folder:
        print("[-] Error: PATH_TO_REPOS_FOLDER environment variable is not set.")
        return
        
    repos_folder = os.path.expanduser(repos_folder.strip('"\'')) # remove quotes if any and expand ~
    
    # If it's a relative path, resolve it relative to this script's directory (since .env is there)
    if not os.path.isabs(repos_folder):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repos_folder = os.path.abspath(os.path.join(script_dir, repos_folder))

    if not os.path.isdir(repos_folder):
        print(f"[-] Error: PATH_TO_REPOS_FOLDER is set to '{repos_folder}', but this is not a valid directory.")
        return
    # Crawl all repos in that folder
    repositories = [os.path.join(repos_folder, d) for d in os.listdir(repos_folder) if os.path.isdir(os.path.join(repos_folder, d))]
    print(f"[+] Found {len(repositories)} repositories in {repos_folder}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_compiled_tokens = []
    
    # 1. Process all specified repositories
    for repo in repositories:
        if not os.path.exists(repo):
            print(f"[!] Warning: Path '{repo}' does not exist. Skipping.")
            continue
        tokens = process_repository(repo)
        print(f"[+] Processed {len(tokens):,} tokens from {os.path.basename(repo)}\n")
        all_compiled_tokens.extend(tokens)
        
    if not all_compiled_tokens:
        print("[-] No tokens compiled. Make sure repo paths and file extensions are correct.")
        return
        
    total_tokens = len(all_compiled_tokens)
    print(f"[+] Total Compiled Tokens: {total_tokens:,}")
    
    # 2. Split into Train and Validation sets
    split_idx = int(total_tokens * (1.0 - VAL_PERCENT))
    train_tokens = all_compiled_tokens[:split_idx]
    val_tokens = all_compiled_tokens[split_idx:]
    
    # 3. Export as raw binary files (uint32)
    # uint32 uses 4 bytes per token. Max vocabulary ID can be up to 4.2 Billion (cl100k is ~100k)
    train_arr = np.array(train_tokens, dtype=np.uint32)
    val_arr = np.array(val_tokens, dtype=np.uint32)
    
    train_file = os.path.join(OUTPUT_DIR, "train.bin")
    val_file = os.path.join(OUTPUT_DIR, "val.bin")
    
    print(f"[+] Writing binary payloads to disk...")
    train_arr.tofile(train_file)
    val_arr.tofile(val_file)
    
    print(f"[✔] Pipeline complete!")
    print(f"    Train Binary: {train_file} ({os.path.getsize(train_file)/1e6:.2f} MB)")
    print(f"    Val Binary:   {val_file} ({os.path.getsize(val_file)/1e6:.2f} MB)")

if __name__ == "__main__":
    main()