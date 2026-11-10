"""Read the 10-K filings, cut them into chunks, and store the chunks in Qdrant.

Run it with:  uv run python -m backend.ingest
"""
import csv
import re
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend import rag

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ITEM = re.compile(r"^## Item (\w+)\. (.+)$", re.M)  # a heading such as "## Item 1A. Risk Factors"
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)


def read_filings() -> list[dict]:
    """One dict per filing in data/filings.csv."""
    with open(DATA_DIR / "filings.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_items(path: Path) -> list[dict]:
    """Split one filing into Items. Each dict has the Item code, title, and text."""
    text = path.read_text(encoding="utf-8")
    heads = list(ITEM.finditer(text))
    items = []
    for i, head in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        items.append({"code": head.group(1), "title": head.group(2), "text": text[head.end():end].strip()})
    return items


def chunk_id(company: str, year: int, item: str, n: int) -> str:
    """A stable ID: the same chunk always gets the same ID, so running again overwrites it."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{company}/{year}/{item}/{n}"))


def chunks_for(filing: dict) -> tuple[list[Document], list[str]]:
    """Cut one filing into chunks. Each Item is split on its own, so no chunk crosses into the next Item."""
    company, year = filing["ticker"], int(filing["fiscal_year"])
    docs, ids = [], []
    for item in read_items(DATA_DIR / filing["file"]):
        for n, chunk in enumerate(splitter.split_text(item["text"])):
            metadata = {"company": company, "year": year, "item": item["code"],
                        "item_title": item["title"], "source": filing["source_url"]}
            docs.append(Document(page_content=chunk, metadata=metadata))
            ids.append(chunk_id(company, year, item["code"], n))
    return docs, ids


def main() -> None:
    rag.ensure_collection()
    store = rag.get_store()
    for filing in read_filings():
        company, year = filing["ticker"], int(filing["fiscal_year"])
        docs, ids = chunks_for(filing)
        # A filing is complete when Qdrant already holds all of its chunks.
        if rag.count_points([company], [year]) == len(docs):
            print(f"{company} {year}: already complete ({len(docs)} chunks), skipped")
            continue
        store.add_documents(docs, ids=ids, batch_size=200)
        print(f"{company} {year}: stored {len(docs)} chunks")
    print(f"Done. The collection holds {rag.count_points()} chunks.")


if __name__ == "__main__":
    main()
