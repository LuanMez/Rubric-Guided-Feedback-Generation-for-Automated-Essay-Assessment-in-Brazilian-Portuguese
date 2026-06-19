"""
apurar_escolhas.py — apura, em cada dataset julgado, se a escolha do juiz foi do
feedback humano (specific_c*) ou do modelo (llm_c*), e gera um dataset-resumo com
a porcentagem de cada fonte por competência.

Como funciona:
  - Em cada dataset_c{n}.csv, a coluna `random` diz qual fonte foi para entrada_A
    e qual foi para entrada_B (ex.: "entrada_A=specific_c1; entrada_B=llm_c1").
  - A coluna `escolha_llm` diz qual posição (A ou B) o juiz preferiu.
  - Cruzando as duas, descobrimos se a escolha caiu no specific ou no llm.

Saída: apuracao_escolhas.csv, com uma linha por competência e as colunas:
    competencia, total_julgado, n_specific, n_llm, pct_specific, pct_llm

Uso:
    python apurar_escolhas.py                 # lê dataset_c1..c5.csv na pasta atual
    python apurar_escolhas.py datasets/       # lê de uma subpasta
    python apurar_escolhas.py datasets/ resumo.csv   # define o nome do arquivo de saída
"""
import os
import re
import sys

import pandas as pd


def _fonte(random_str, posicao):
    """Dada a coluna `random` e a posição ('A' ou 'B'), retorna 'specific' ou 'llm'."""
    m = re.search(rf"entrada_{posicao}=([^;]+)", str(random_str))
    if not m:
        return None
    col = m.group(1).strip()
    if col.startswith("specific"):
        return "specific"
    if col.startswith("llm"):
        return "llm"
    return None


def apurar(pasta=".", saida="apuracao_escolhas.csv"):
    linhas_resumo = []

    for n in range(1, 6):
        caminho = os.path.join(pasta, f"dataset_c{n}.csv")
        if not os.path.isfile(caminho):
            print(f"[!] Não encontrei '{caminho}' — pulando C{n}.")
            continue

        df = pd.read_csv(caminho, dtype=str, keep_default_na=False, encoding="utf-8")
        if "escolha_llm" not in df.columns or "random" not in df.columns:
            print(f"[!] '{caminho}' não tem as colunas escolha_llm/random — pulando C{n}.")
            continue

        n_specific = n_llm = invalidas = 0
        for _, linha in df.iterrows():
            esc = str(linha.get("escolha_llm", "")).strip().upper()
            if esc not in ("A", "B"):
                invalidas += 1
                continue
            fonte = _fonte(linha.get("random", ""), esc)
            if fonte == "specific":
                n_specific += 1
            elif fonte == "llm":
                n_llm += 1
            else:
                invalidas += 1

        total = n_specific + n_llm
        pct_specific = round(100 * n_specific / total, 2) if total else 0.0
        pct_llm = round(100 * n_llm / total, 2) if total else 0.0

        linhas_resumo.append({
            "competencia": f"C{n}",
            "total_julgado": total,
            "n_specific": n_specific,
            "n_llm": n_llm,
            "pct_specific": pct_specific,
            "pct_llm": pct_llm,
        })
        aviso = f"  ({invalidas} linha(s) sem escolha válida)" if invalidas else ""
        print(f"C{n}: specific {pct_specific}%  |  llm {pct_llm}%  "
              f"(specific={n_specific}, llm={n_llm}, total={total}){aviso}")

    if not linhas_resumo:
        print("\nNenhum dataset apurado.")
        return

    resumo = pd.DataFrame(linhas_resumo, columns=[
        "competencia", "total_julgado", "n_specific", "n_llm", "pct_specific", "pct_llm",
    ])

    # Linha final com o total geral (todas as competências somadas).
    tot_spec = int(resumo["n_specific"].sum())
    tot_llm = int(resumo["n_llm"].sum())
    tot = tot_spec + tot_llm
    resumo.loc[len(resumo)] = {
        "competencia": "GERAL",
        "total_julgado": tot,
        "n_specific": tot_spec,
        "n_llm": tot_llm,
        "pct_specific": round(100 * tot_spec / tot, 2) if tot else 0.0,
        "pct_llm": round(100 * tot_llm / tot, 2) if tot else 0.0,
    }

    caminho_saida = saida if os.path.isabs(saida) else os.path.join(pasta, saida)
    resumo.to_csv(caminho_saida, index=False, encoding="utf-8")
    print(f"\nResumo salvo em '{caminho_saida}'.")


if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else "."
    saida = sys.argv[2] if len(sys.argv) > 2 else "apuracao_escolhas.csv"
    apurar(pasta, saida)
