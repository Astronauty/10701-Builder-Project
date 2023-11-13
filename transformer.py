from torch.nn import Module


class TransformerMT(Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # init embedding tensor

        # init position encoding tensor

        # init transformer

    def forward(self, input):
        # get dense embeddings tensor given input sparse vocab index tensor

        # apply positional encoding transformation

        # apply transformer
        pass

    # TODO: add other function(s) for usability during inference/translation
