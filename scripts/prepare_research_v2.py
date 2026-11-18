"""Fetch the five registered dataset files and prepare the two token splits."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optcourse.research.data import prepare_datasets, sha256

WIKI = "Salesforce/wikitext"
WIKI_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
STORIES = "roneneldan/TinyStories"
STORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
SOURCES = (
    ("wiki_train", WIKI, WIKI_REVISION, "wikitext-103-raw-v1/train-00000-of-00002.parquet", 156987808,
     "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"),
    ("wiki_val", WIKI, WIKI_REVISION, "wikitext-103-raw-v1/validation-00000-of-00001.parquet", 657209,
     "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"),
    ("wiki_test", WIKI, WIKI_REVISION, "wikitext-103-raw-v1/test-00000-of-00001.parquet", 732610,
     "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"),
    ("stories_train", STORIES, STORIES_REVISION, "data/train-00000-of-00004-2d5a1467fff1081b.parquet", 248731111,
     "77cf780cebe52b6e83e3a2ac84bc56d8059363113e41d17a023f1d8b2ed0fc0b"),
    ("stories_val", STORIES, STORIES_REVISION, "data/validation-00000-of-00001-869c898b519ad725.parquet", 9989127,
     "33406a6206554cfc279c29c11f4df51528af487aa1a602b075566fc83c49dcab"),
)


def download(spec, raw):
    name, repository, revision, relative, size, digest = spec
    source = f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{relative}"
    file = raw / f"{name}.parquet"
    if not file.exists():
        temporary = file.with_suffix(".partial")
        for attempt in range(4):
            try:
                request = Request(source, headers={"User-Agent": "OptimizationCourseResearch/2"})
                with urlopen(request, timeout=90) as stream, temporary.open("wb") as output:
                    shutil.copyfileobj(stream, output, length=2**20)
                if temporary.stat().st_size != size or sha256(temporary) != digest:
                    raise ValueError(f"Downloaded source identity differs: {name}")
                temporary.replace(file)
                break
            except (ValueError, OSError):
                if attempt == 3:
                    raise
                time.sleep(2)
    if file.stat().st_size != size or sha256(file) != digest:
        raise ValueError(f"Existing source identity differs: {name}")
    return dict(name=name, path=str(file.relative_to(ROOT)).replace("/", "\\"), repository=repository,
                revision=revision, file=relative, url=source, size=size, sha256=digest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "NOT_RUN", "files": len(SOURCES), "source_bytes": sum(row[4] for row in SOURCES)}))
        return
    data = ROOT / "data/research_v2"
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "work/tokenizer_cache"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda spec: download(spec, raw), SOURCES))
    (raw / "source_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    prepare_datasets(raw, data / "tokens")


if __name__ == "__main__":
    main()
