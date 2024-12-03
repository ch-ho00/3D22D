import os

def append_to_txt_files(folder_path, fixed_string):
    """
    Append a fixed string to all .txt files in the specified folder.

    :param folder_path: Path to the folder containing .txt files.
    :param fixed_string: The string to append to each .txt file.
    """
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Target only .txt files
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "a") as file:  # Open in append mode
                    file.write(fixed_string)
                print(f"Appended to {filename}")
            except Exception as e:
                print(f"Failed to append to {filename}: {e}")

# Example usage
folder_path = "/root/style/STRSTY-style-cap"
fixed_string = "The product(s) is/are placed in a studio of STRSTY."
append_to_txt_files(folder_path, fixed_string)

