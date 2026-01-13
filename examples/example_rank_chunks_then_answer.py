from chunkrank.chunker import Chunker, ChunkerConfig
from chunkrank.ranker import Ranker
from chunkrank.answerers import LocalExtractiveAnswerer

DOC = """
ChunkRank is a library idea that does model-aware chunking and answer ranking.
Each LLM has a different context window and tokenizer. For example, some models support 128k tokens.
Once documents are chunked, answering per chunk can create multiple candidate answers.
A ranking step is needed to select the best answer.
BM25 is a classic ranking method used in information retrieval.
TF-IDF cosine similarity is another lightweight alternative.
""" * 20

def main():
    question = "Why do we need ranking after chunking?"
    model = "gpt-4o-mini"

    # 1)chunk
    chunker = Chunker(ChunkerConfig(model=model, overlap_tokens=64))
    chunks = chunker.split(DOC)

    # 2)rank chunks by relevance to question (BM25)
    ranker = Ranker(method="bm25")
    ranked_chunks = ranker.rank_texts(question, chunks)
    top_chunk, top_score = ranked_chunks[0]

    # 3)answer using the best chunk (local test answerer)
    answerer = LocalExtractiveAnswerer(min_overlap=2)
    answer = answerer.answer(question, top_chunk)

    print(f"Chunks: {len(chunks)}")
    print(f"Top chunk score: {top_score:.4f}")
    print("Answer:", answer or "[no answer found]")

if __name__ == "__main__":
    main()
