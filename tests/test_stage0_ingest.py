"""Stage 0: ingest. The first test needs no network. The second one talks to Qdrant."""
from backend import ingest, rag


def test_chunks_are_complete(test_data):
    """The 20 filings split into the expected number of chunks, and every chunk has its metadata."""
    filings = ingest.read_filings()
    assert len(filings) == 20

    total = 0
    for filing in filings:
        docs, ids = ingest.chunks_for(filing)
        assert len(set(ids)) == len(ids)  # no two chunks share an ID
        for doc in docs:
            for field in ["company", "year", "item", "item_title", "source"]:
                assert doc.metadata[field] not in (None, "")
            assert len(doc.page_content) <= 1500
        total += len(docs)

    assert total == test_data["checked_values"]["total_chunks"] == 10229


def test_ingest_twice_gives_same_count(test_data):
    """After ingest the collection holds 10,229 chunks, and a second run changes nothing."""
    expected = test_data["checked_values"]["total_chunks"]

    ingest.main()  # the first run calls OpenRouter only if chunks are missing
    assert rag.count_points() == expected

    ingest.main()  # the second run should skip every filing
    assert rag.count_points() == expected
