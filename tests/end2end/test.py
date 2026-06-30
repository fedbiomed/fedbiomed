import os
import time

import requests
import tqdm


def download_file(url, filename, retries=2, headers=None):
    """
    Helper method handling downloading large files.
    """
    chunk_size = 1024
    for attempt in range(retries):
        try:
            r = requests.get(
                url, headers=headers if headers else {}, stream=True, timeout=60
            )
            r.raise_for_status()
            total = r.headers.get("Content-Length")
            with open(filename, "wb") as f:
                pbar = tqdm.tqdm(
                    unit="B", unit_scale=True, total=int(total) if total else None
                )
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        pbar.update(len(chunk))
                        f.write(chunk)
            return filename
        except requests.RequestException as e:
            # Never print `url` directly if it could ever contain a token as a query param.
            # Here the token is only in headers, so this is safe, but we still avoid
            # printing headers or any request object that might expose it.
            print(
                f"Download attempt {attempt + 1}/{retries} failed for {filename}: {type(e).__name__}"
            )
            if attempt == retries - 1:
                raise requests.RequestException(
                    f"Failed to download {filename} after {retries} attempts"
                ) from None
            time.sleep(3)
        except Exception as e:
            print(f"Unexpected error downloading {filename}: {type(e).__name__}")


if __name__ == "__main__":
    root_folder = "./"
    zip_filename = os.path.join(root_folder, "7kd5wj7v7p-3.zip")
    data_folder = os.path.join(root_folder)
    extracted_folder = os.path.join(data_folder, "7kd5wj7v7p-3", "IXI_sample")
    url = os.getenv("FBM_DATASET_IXI_DOWNLOAD_URL")
    headers = {"PRIVATE-TOKEN": os.getenv("FBM_GITLAB_TOKEN")}
    download_file(url, zip_filename, headers=headers)
