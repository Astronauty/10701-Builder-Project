import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchtext.data.metrics import bleu_score
from data_loader import CustomTokens
from transformer import TransformerMT
from data_loader import Langs

class Prediction:

    def __init__(self, model: TransformerMT, test_dataloader, vocabulary: Langs):
        self.model: TransformerMT = model
        self.test_dataloader: DataLoader = test_dataloader
        self.vocabulary = vocabulary

        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predicted_toks = []
        self.reference_toks = []


    def _add_list_of_tokens(self, batch_token_ids, reference: bool):
        stringified_token_ids = []

        custom_token_ids_to_remove = set([CustomTokens.SOS.value, CustomTokens.EOS.value, CustomTokens.PAD.value])
        for token_ids in batch_token_ids.tolist():
            temp = [str(token_id) for token_id in token_ids if token_id not in custom_token_ids_to_remove]

            if reference:
                self.reference_toks.append([temp])
            else:
                self.predicted_toks.append(temp)

        return stringified_token_ids

    def get_bleu_score(self, num_batches_to_eval=None):
        num_batches_to_eval = len(self.test_dataloader) if num_batches_to_eval is None else min(num_batches_to_eval, len(self.test_dataloader))

        cumulative_bleu_score = 0
        batch_num = 1
        for en_token_ids, fr_token_ids in tqdm(self.test_dataloader, total=num_batches_to_eval):
            en_token_ids = en_token_ids.to(self.device)
            fr_token_ids = fr_token_ids.to(self.device)

            en_token_ids = torch.squeeze(en_token_ids)
            fr_token_ids = torch.squeeze(fr_token_ids)

            predicted_token_ids : Tensor = self._get_predicted_token_ids(en_token_ids=en_token_ids)

            self._add_list_of_tokens(predicted_token_ids, reference=False)
            self._add_list_of_tokens(fr_token_ids, reference=True)

            if batch_num % 5 == 0:
                score = bleu_score(self.predicted_toks, self.reference_toks)
                print(len(self.predicted_toks), len(self.reference_toks))
                print(f"batch_num: {batch_num}, bleu_score: {round(score, 5)}")

            if batch_num >= num_batches_to_eval:
                break
            batch_num += 1

        return bleu_score(self.predicted_toks, self.reference_toks)

    def _get_predicted_token_ids(self, en_token_ids, max_tokens_to_generate=50):
        encoder_out = self.model.encode(en_token_ids)

        batch_size = en_token_ids.shape[0]

        # sos only to start
        tgt = torch.LongTensor([CustomTokens.SOS.value]).unsqueeze(0).cuda().view(1, 1)   # borrowed from Conan
        tgt = tgt.repeat(batch_size, 1)

        for num_tokens_generated in range(1, max_tokens_to_generate + 1):
            target_vocabulary_logits = self.model.decode(encoder_out=encoder_out, tgt=tgt)

            predicted_token_id = torch.argmax(input=target_vocabulary_logits[:, tgt.shape[1] - 1: tgt.shape[1], :], dim=2)

            tgt = torch.cat([tgt, predicted_token_id.view(batch_size, 1)], dim=1)

        return tgt
