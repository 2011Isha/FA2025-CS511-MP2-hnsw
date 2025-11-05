import faiss
import h5py
import numpy as np
import os
import requests


DATA_URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
DATA_PATH = os.path.join(os.getcwd(), "sift-128-euclidean.hdf5")
OUT_PATH = os.path.join(os.getcwd(), "output.txt")

def _download_if_needed(url: str, dst: str):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    print(f"Downloading {url} -> {dst} (≈1.8 GB)...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print("Download complete.")


def evaluate_hnsw():

  _download_if_needed(DATA_URL, DATA_PATH) #Downloading the dataset

  #HDF5 groups typically include: "train" (db), "test" (queries), and sometimes "neighbors"
  with h5py.File(DATA_PATH, "r") as f:
    xb = f["train"][:].astype("float32")   # database / base vectors
    xq_first = f["test"][:1].astype("float32") # first query only

  # Build FAISS HNSW (no PQ), required params: M = 16, efConstruction = 200, efSearch = 200
  d = xb.shape[1]
  index = faiss.IndexHNSWFlat(d, 16)     # M=16
  index.hnsw.efConstruction = 200
  index.hnsw.efSearch = 200
  index.add(xb)

  # 3) Search top-10 for the first query vector
  _, I = index.search(xq_first, 10) # I shape: (1, 10)

  # 4) Write the 10 indices to ./output.txt (one index per line)
  np.savetxt(OUT_PATH, I[0], fmt="%d")
  print(f"Wrote top-10 neighbor IDs for the first query to: {OUT_PATH}")

if __name__ == "__main__":
    evaluate_hnsw()
