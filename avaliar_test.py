"""
avaliar_test.py — avalia o BERTimbau JÁ TREINADO no conjunto de TESTE.
Dá os números finais e honestos (redações que o modelo nunca viu, e que não foram
usadas para escolher a melhor época). Rode DEPOIS que o treino terminar:
    python avaliar_test.py

Ele usa o modelo salvo (bertimbau_redacao.pt) e o tokenizer (bertimbau_redacao_tok/).
Não retreina nada. Precisa rodar de dentro da pasta essay-br.
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from build_dataset import Corpus

MODEL_NAME = "neuralmind/bert-large-portuguese-cased"   # tem que ser o MESMO usado no treino
MAXLEN = 512
BATCH = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMP = ["c1", "c2", "c3", "c4", "c5"]
BANDAS = np.array([0, 40, 80, 120, 160, 200])


def texto(row):
    e = row["essay"]
    corpo = "\n".join(e) if isinstance(e, list) else str(e)
    return f"{row['title']} [SEP] {corpo}"


class EssayDS(Dataset):
    def __init__(self, df, tok):
        self.textos = [texto(r) for _, r in df.iterrows()]
        self.y = df[COMP].astype(float).values     # notas reais (0-200)
        self.tok = tok

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, i):
        enc = self.tok(self.textos[i], truncation=True, max_length=MAXLEN,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.y[i], dtype=torch.float)
        return item


class BertScorer(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(name)
        h = self.bert.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h, len(COMP))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        return torch.sigmoid(self.head(self.drop(out.last_hidden_state[:, 0])))


def para_banda(v):
    pontos = v * 200.0
    idx = np.abs(pontos[:, :, None] - BANDAS[None, None, :]).argmin(axis=2)
    return BANDAS[idx]


@torch.no_grad()
def main():
    print(f"Dispositivo: {DEVICE}")
    _, _, test = Corpus().read_splits()
    tok = AutoTokenizer.from_pretrained("bertimbau_redacao_tok")
    model = BertScorer(MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load("bertimbau_redacao.pt", map_location=DEVICE))
    model.eval()

    dl = DataLoader(EssayDS(test, tok), batch_size=BATCH)
    preds, reais = [], []
    for batch in dl:
        labels = batch.pop("labels").numpy()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        preds.append(model(**batch).cpu().numpy())
        reais.append(labels)
    preds = para_banda(np.concatenate(preds))
    reais = np.concatenate(reais).round().astype(int)

    print(f"\n=== TESTE — {len(reais)} redações (números finais) ===")
    qwks = []
    for j, comp in enumerate(COMP):
        yt, yp = reais[:, j], preds[:, j].astype(int)
        mae = mean_absolute_error(yt, yp)
        try:
            qwk = cohen_kappa_score(yt, yp, labels=list(BANDAS), weights="quadratic")
        except Exception:
            qwk = float("nan")
        qwks.append(qwk)
        print(f"  {comp.upper()}:  MAE={mae:5.1f}   QWK={qwk:.2f}")
    print(f"  NOTA FINAL: MAE={mean_absolute_error(reais.sum(1), preds.sum(1)):.1f}"
          f"   |   QWK médio: {np.nanmean(qwks):.2f}")


if __name__ == "__main__":
    main()
