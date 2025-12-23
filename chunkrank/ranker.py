from sentence_transformers import CrossEncoder

class Ranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rank():
        ...
    def rank_answer(self):
        ...
