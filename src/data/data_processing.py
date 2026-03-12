import zipfile

class DataProcessor:
    def __init__(self, zip_file_path: str, extract_dir: str):
        self.zip_path = zip_file_path
        self.extract_dir = extract_dir

    def unzip_file(self):
        try:
            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                zip_ref.extractall(self.extract_dir)
                print(f"Zip file {self.zip_path} extracted to {self.extract_dir}")
        except zipfile.BadZipFile:
            print("Bad Zip File Error: This is a bad zip or not a zip.")
        except FileNotFoundError:
            print("There is no file or dir.")
        except Exception as unexpected_error:
            print(f"Unexpected error: {unexpected_error}")
