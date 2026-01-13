from chunkrank import Chunker, ChunkerConfig

def test_chunking_basic():
    text = "Hello people " * 500
    ch = Chunker(ChunkerConfig(model="gpt-4o-mini"))
    chunks = ch.split(text)
    assert len(chunks) > 0
