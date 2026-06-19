"""
avaliar_metricas.py — calcula métricas de avaliação do corretor sobre o Essay-BR.
Rode de dentro da pasta essay-br:
    python avaliar_metricas.py

Métricas calculadas por competência:
  - Accuracy, Precision, Recall, F1  -> visão de CLASSIFICAÇÃO (acerto exato da faixa)
  - Acerto +/- 1 faixa               -> tolera errar um degrau (40 pontos)
  - MAE                              -> erro médio em pontos (intuitivo)
  - QWK                              -> concordância ordinal (a métrica-padrão da área)
E o MAE da nota final (0 a 1000).

OBS sobre os dados: o Essay-BR não traz o enunciado completo do tema, só o título da
redação. Por isso usamos o título como "proposta" aqui — suficiente para medir as notas,
mas não ideal para avaliar fuga ao tema (C2). As métricas refletem a concordância com os
corretores do Essay-BR, que pode diferir do padrão de um professor específico seu.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    mean_absolute_error,
)

# Reaproveita o corretor já configurado (3 exemplos few-shot, prompt, etc.)
from corretor_redacao5 import corrigir, valid, texto_redacao

N = 30          # quantas redações avaliar. Cada uma chama o modelo (leva tempo).
                # Use 10 para um teste rápido; 50-100 para um número mais estável.
CLASSES = [0, 40, 80, 120, 160, 200]   # as 6 notas possíveis por competência


def coletar(amostra):
    """Roda o modelo e devolve notas reais e previstas por competência + total."""
    y_true = {f"c{i}": [] for i in range(1, 6)}
    y_pred = {f"c{i}": [] for i in range(1, 6)}
    total_true, total_pred = [], []

    for k, (_, row) in enumerate(amostra.iterrows(), 1):
        proposta = str(row["title"])               # o dataset só tem o título como tema
        pred = corrigir(proposta, texto_redacao(row))
        print(f"[{k}/{len(amostra)}] processada")
        if not pred:
            continue
        try:
            notas = [int(pred[f"c{i}"]["nota"]) for i in range(1, 6)]
        except (KeyError, TypeError, ValueError):
            continue                                # JSON incompleto: pula esta redação
        for i in range(1, 6):
            y_true[f"c{i}"].append(int(row[f"c{i}"]))
            y_pred[f"c{i}"].append(notas[i - 1])
        total_true.append(int(row["score"]))
        total_pred.append(sum(notas))

    return y_true, y_pred, total_true, total_pred


def metricas_competencia(yt, yp):
    acc = accuracy_score(yt, yp)
    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=CLASSES, average="macro", zero_division=0
    )
    off1 = float(np.mean([abs(a - b) <= 40 for a, b in zip(yt, yp)]))
    mae = mean_absolute_error(yt, yp)
    try:
        qwk = cohen_kappa_score(yt, yp, labels=CLASSES, weights="quadratic")
    except Exception:
        qwk = float("nan")
    return acc, p, r, f1, off1, mae, qwk


if __name__ == "__main__":
    amostra = valid.sample(n=min(N, len(valid)), random_state=42)
    print(f"Avaliando {len(amostra)} redações... (cada uma chama o modelo, seja paciente)\n")
    y_true, y_pred, total_true, total_pred = coletar(amostra)

    print("\n" + "=" * 72)
    print("MÉTRICAS POR COMPETÊNCIA")
    print("=" * 72)
    for i in range(1, 6):
        yt, yp = y_true[f"c{i}"], y_pred[f"c{i}"]
        if not yt:
            print(f"\nC{i}: sem dados válidos.")
            continue
        acc, p, r, f1, off1, mae, qwk = metricas_competencia(yt, yp)
        print(f"\nC{i}  (n={len(yt)})")
        print(f"  Accuracy (acerto exato da faixa) : {acc:.2f}")
        print(f"  Precision / Recall / F1 (macro)  : {p:.2f} / {r:.2f} / {f1:.2f}")
        print(f"  Acerto dentro de +/- 1 faixa     : {off1:.2f}")
        print(f"  MAE (pontos, 0-200)              : {mae:.1f}")
        print(f"  QWK (concordância ordinal)       : {qwk:.2f}")

    if total_true:
        print("\n" + "=" * 72)
        print("NOTA FINAL (0 a 1000)")
        print("=" * 72)
        print(f"  MAE: {mean_absolute_error(total_true, total_pred):.1f} pontos   (n={len(total_true)})")

    print("\nDica: rode duas vezes mudando algo (prompt, exemplos few-shot) e compare o QWK")
    print("e o MAE. Se subirem o QWK e baixarem o MAE, a mudança melhorou o modelo.")
