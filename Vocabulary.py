from CustomToken import CustomToken

class Vocabulary:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.token2index = {}
        self.token2count = {}
        self.index2token = {
            CustomToken.SOS.value: "SOS",
            CustomToken.EOS.value: "EOS",
            CustomToken.PAD.value: "PAD",
            CustomToken.UNK.value: "UNK"
        }

        self.vocabulary_size = len(self.index2token)

    def add_sentence(self, sentence):
        for token in self.tokenizer(sentence):
            self._add_token(token)

    def _add_token(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.vocabulary_size
            self.token2count[token] = 1
            self.index2token[self.vocabulary_size] = token
            self.vocabulary_size += 1
        else:
            self.token2count[token] += 1

    def filter_out_rare_keys(self, threshold):
        print(f"token count before filtering out rare words: {len(self.token2index)}")
        # delete rare keys
        for token_id, token_count in self.token2count.items():
            if token_count < threshold:
                del self.token2index[token_id]

        print(f"token count after filtering out rare words: {len(self.token2index)}")


