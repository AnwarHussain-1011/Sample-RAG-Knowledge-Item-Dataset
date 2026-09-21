from helpdesk_rag.chunking import chunk_documents

def test_chunking_has_stable_ids_and_overlap():
    records=[{"document_id":"D1","title":"Title","topic":"IT","text":"one two three four five six seven","source":"test","question":"q","answer":"a"}]
    chunks=chunk_documents(records, chunk_size=4, overlap=1)
    assert len(chunks) == 3
    assert chunks[0].chunk_id == "D1::chunk_0000"
    assert chunks[1].chunk_id == "D1::chunk_0001"
    assert chunks[0].text.split()[-1] == chunks[1].text.split()[0]

def test_invalid_chunk_parameters():
    try: chunk_documents([], 2, 2)
    except ValueError: pass
    else: raise AssertionError("expected invalid overlap to fail")
