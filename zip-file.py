import zipfile
import os

# Folder path to zip
folder_path = r"C:\Users\DELL\Desktop\iot-project"
# Output zip file path
zip_path = os.path.join(folder_path, "iot_project_files.zip")

# Create a zip file and add all files from the folder
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Add file to zip, preserving the folder structure
            zipf.write(file_path, os.path.relpath(file_path, folder_path))

print(f"All files zipped successfully into {zip_path}")
