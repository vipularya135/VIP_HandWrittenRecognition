import os

def print_directory_structure(rootdir):
    for dirpath, dirnames, filenames in os.walk(rootdir):
        level = dirpath.replace(rootdir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(dirpath)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in filenames:
            if not f.endswith('.png'):  # Exclude .png files
                print(f"{subindent}{f}")

if __name__ == "__main__":
    root_directory = '.'  # or specify your project path
    print_directory_structure(root_directory)
