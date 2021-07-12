class Model:

    def train_batch(self) -> float:
        raise NotImplemented()

    def eval_batch(self) -> float:
        raise NotImplemented()

    def save(self, fp):
        raise NotImplemented()

    def load(self, fp):
        raise NotImplemented()
