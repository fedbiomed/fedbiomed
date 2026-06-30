import os
import time

import requests
import tqdm


def download_file(url, filename, retries=2):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunk_size = 1024
    for attempt in range(retries):
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            total = r.headers.get("Content-Length")
            with open(filename, "wb") as f:
                pbar = tqdm.tqdm(unit="B", total=int(total) if total else None)
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        pbar.update(len(chunk))
                        f.write(chunk)
            return filename
        except requests.RequestException as e:
            print(e)
            if attempt == retries - 1:
                raise
            time.sleep(3)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    root_folder = "./"
    zip_filename = os.path.join(root_folder, "7kd5wj7v7p-3.zip")
    data_folder = os.path.join(root_folder)
    extracted_folder = os.path.join(data_folder, "7kd5wj7v7p-3", "IXI_sample")

    download_file(
        "https://data.mendeley.com/public-api/zip/7kd5wj7v7p/download/3", zip_filename
    )
