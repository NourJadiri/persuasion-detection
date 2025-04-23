import os

def clean_dot_underscore_files(root_dir):
    removed_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("._"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    removed_files.append(file_path)
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")
    print(f"Removed {len(removed_files)} files.")
    if removed_files:
        print("Files removed:")
        for f in removed_files:
            print(f)

if __name__ == "__main__":
    clean_dot_underscore_files("data/raw")