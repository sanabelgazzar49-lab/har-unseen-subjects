# download_data.py: downloads and unpacks the UCI HAR dataset into data/raw/uci_har.
import glob
import os
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ZIP = "data/raw/uci_har_download.zip"
OUT = "data/raw/uci_har"


def main():
    os.makedirs("data/raw", exist_ok=True)
    if not os.path.exists(ZIP) or os.path.getsize(ZIP) < 1_000_000:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(ZIP, "wb") as f:
            f.write(r.read())
    if not os.path.exists(OUT) or not os.listdir(OUT):
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(OUT)
        for inner in glob.glob(f"{OUT}/**/*.zip", recursive=True):
            with zipfile.ZipFile(inner) as z:
                z.extractall(OUT)
    hits = glob.glob(f"{OUT}/**/features.txt", recursive=True)
    assert hits, "features.txt not found after extraction"
    print("Dataset ready at:", os.path.dirname(hits[0]))


if __name__ == "__main__":
    main()
