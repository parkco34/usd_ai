#!/usr/bin/env python
import pandas as pd

def read_file(file_path):
    """
    Needs input validation and to generalize to different directory structures
    ?
    """
    # Move back to parent directory
    os.chdir("../")
    # Assign absolute path to current directory
    daddy = os.getcwd()

    # Check if file exists in current directory
    if os.path.exists(file_path):
        print(f"{file_path} found in parent directory")

        return pd.read_csv(file_path)

    # Ignore certain directories and files
    ignore_dirs = {"__pycache__", "lib", "include", "share", "bin"}
    ignore_files = {".DS_Store"}


    # Case when file is not found in current directory, recursively looking at directories from the top-down (includes symlinks and excludes files of the type '.' and '..')
    # os.walk(top, topdown=True, onerror=None, followlinks=False)
    for dirpath, dirnames, filenames in os.walk(daddy, topdown=True):
        """
        Modifying dirnames when topdown is False has no effect on the behavior of the walk, because in bottom-up mode the directories in dirnames are generated before dirpath itself is generated
        """

        # Filtering items to ignore
        dirnames[:] = [div for div in dirnames if div not in ignore_dirs]
        filenames[:] = [file for file in filenames if file not in
                                                ignore_files]

        # Check if filename is in the files
        if file_path in filenames:
            print(f"Found {file_path} in  {dirpath}/")

            # Concatenate the components of the absolute file path
            abs_file_path = os.path.join(dirpath, file_path)

            return pd.read_csv(abs_file_path)

    # Raise exception when file still not found
    raise FileNotFoundError(
        f"'{file_path}' not found in any of the subsequent directories or "
        f"any subdirectory "
    )

# Additional features needed for this to be a proper package/module 
