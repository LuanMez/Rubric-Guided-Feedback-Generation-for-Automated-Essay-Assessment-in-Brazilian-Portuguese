"""
treinar_bertimbau.py — Caminho B: fine-tuning de um encoder (BERTimbau) para
prever as 5 notas de competência do ENEM, treinando no Essay-BR INTEIRO.

Por que encoder e não LLM: para a tarefa de NOTA, modelos como o BERTimbau
costumam pontuar tão bem ou melhor que LLMs, treinam em minutos e são leves —
cabem com folga na sua RTX 3060. O feedback continua pelo caminho do prompt (LLM).

INSTALAÇÃO (uma vez só, no terminal):
    pip install transformers accelerate
    # PyTorch com CUDA, para usar a GPU da 3060:
    pip install torch --index-url https://download.pytorch.org/whl/cu121

COMO RODAR (de dentro da pasta essay-br):
    python treinar_bertimbau.py

Saída: treina, imprime QWK e MAE por competência na validação a cada época
(compare com o avaliar_metricas.py do few-shot) e salva o modelo treinado.
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from build_dataset import Corpus

MODEL_NAME = "neuralmind/bert-large-portuguese-cased"   # versão large: mais precisa
MAXLEN = 512        # BERT vê no máximo 512 tokens; redações longas são truncadas
EPOCHS = 5
BATCH = 4           # large é pesado; batch menor + acumulação cabe na 3060 12GB
ACCUM = 2           # acumula gradiente: batch efetivo = BATCH * ACCUM = 8
LR = 1e-5           # large costuma treinar melhor com LR mais baixa
USE_AMP = True      # precisão mista (fp16): metade da memória e mais rápido
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
        self.y = df[COMP].astype(float).values / 200.0     # normaliza para [0,1]
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
    """BERTimbau + uma cabeça de regressão com 5 saídas (uma por competência)."""
    def __init__(self, name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(name)
        h = self.bert.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h, len(COMP))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0]          # vetor [CLS]
        return torch.sigmoid(self.head(self.drop(cls)))   # saída em [0,1]


def para_banda(valores_0a1):
    """Converte saída [0,1] na nota {0,40,...,200} da banda mais próxima."""
    pontos = valores_0a1 * 200.0
    idx = np.abs(pontos[:, :, None] - BANDAS[None, None, :]).argmin(axis=2)
    return BANDAS[idx]


@torch.no_grad()
def avaliar(model, loader):
    model.eval()
    preds, reais = [], []
    for batch in loader:
        labels = batch.pop("labels").numpy()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        preds.append(model(**batch).cpu().numpy())
        reais.append(labels)
    preds = para_banda(np.concatenate(preds))
    reais = (np.concatenate(reais) * 200.0).round().astype(int)
    print("=== Validação (BERTimbau) ===")
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
    mae_total = mean_absolute_error(reais.sum(1), preds.sum(1))
    qwk_medio = float(np.nanmean(qwks))
    print(f"  NOTA FINAL: MAE={mae_total:.1f}   |   QWK médio: {qwk_medio:.2f}")
    return qwk_medio


def prever(model, tok, titulo, corpo):
    """Prevê as 5 notas de uma redação nova (use depois de treinar)."""
    model.eval()
    enc = tok(f"{titulo} [SEP] {corpo}", truncation=True, max_length=MAXLEN,
              padding="max_length", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**enc).cpu().numpy()
    notas = para_banda(out)[0]
    return {COMP[i]: int(notas[i]) for i in range(5)} | {"nota_final": int(notas.sum())}


def main():
    print(f"Dispositivo: {DEVICE}")
    if DEVICE == "cpu":
        print("[!] CUDA não detectada — treinaria na CPU (muito lento). Cheque a instalação do torch.")

    train, valid, test = Corpus().read_splits()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    dl_train = DataLoader(EssayDS(train, tok), batch_size=BATCH, shuffle=True)
    dl_valid = DataLoader(EssayDS(valid, tok), batch_size=BATCH)

    model = BertScorer(MODEL_NAME).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    usar_amp = USE_AMP and DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)

    melhor_qwk = -1.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        opt.zero_grad()
        for passo, batch in enumerate(dl_train, 1):
            labels = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=usar_amp):
                loss = lossf(model(**batch), labels) / ACCUM
            scaler.scale(loss).backward()
            if passo % ACCUM == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            total += loss.item() * ACCUM
        print(f"\nÉpoca {ep}/{EPOCHS}  loss médio={total/len(dl_train):.4f}")
        qwk_medio = avaliar(model, dl_valid)
        if qwk_medio > melhor_qwk:                     # guarda só a melhor época
            melhor_qwk = qwk_medio
            torch.save(model.state_dict(), "bertimbau_redacao.pt")
            tok.save_pretrained("bertimbau_redacao_tok")
            print(f"  -> melhor modelo até agora (QWK médio {qwk_medio:.2f}) salvo.")

    print(f"\nTreino concluído. Melhor QWK médio na validação: {melhor_qwk:.2f}")
    print("Modelo salvo em bertimbau_redacao.pt (tokenizer em bertimbau_redacao_tok/).")
    print("Para usar depois: recrie BertScorer, carregue o state_dict e chame prever().")


if __name__ == "__main__":
    main()
