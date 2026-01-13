from chunkrank.chunker import Chunker, ChunkerConfig
from chunkrank.ranker import Ranker
from chunkrank.answerers import LocalExtractiveAnswerer

DOC = """
ChunkRank aims to combine model-aware text chunking with answer ranking.
Model-aware chunking means selecting chunk sizes based on the model's context window and tokenizer.
After chunking, you might get multiple answers from different chunks.
To pick the best one, you rank the answers against the question.
BM25 often works well for ranking short answer candidates.
""" * 30

def main():
    question = "What is model-aware chunking?"
    model = "gpt-4o-mini"

    chunker = Chunker(ChunkerConfig(model=model, overlap_tokens=64))
    chunks = chunker.split(DOC)

    answerer = LocalExtractiveAnswerer(min_overlap=2)
    answers = []
    for i, ch in enumerate(chunks, 1):
        a = answerer.answer(question, ch)
        if a:
            answers.append(a)

    if not answers:
        print("No answers produced.")
        return

    ranker = Ranker(method="tfidf")
    ranked_answers = ranker.rank(question, answers)

    print(f"Chunks: {len(chunks)}")
    print(f"Candidate answers: {len(answers)}")
    print("Best answer:", ranked_answers[0][0])
    print("\nTop 5:")
    for ans, score in ranked_answers[:5]:
        print(f"- {score:.4f}: {ans}")

if __name__ == "__main__":
    main()
