"""
juiz_llm.py — LLM como juiz para os datasets de comparação A/B.

Para cada dataset_c{n}.csv, o modelo lê a PROPOSTA (prompt) e a REDAÇÃO
(essay_text), compara os dois feedbacks (entrada_A e entrada_B) sobre AQUELA
competência e decide qual é o melhor. O resultado vai numa coluna nova:

    escolha_llm  ->  "A"  ou  "B"

Observações:
  - O juiz NÃO sabe qual feedback é humano e qual é do modelo (comparação cega);
    essa informação fica só na coluna `random`. Depois você cruza `escolha_llm`
    com `random` para saber se ele preferiu o feedback humano ou o do LLM.
  - Os datasets já têm A/B sorteados por linha, o que dilui o viés de posição
    do juiz ao longo do conjunto.
  - Salva a cada linha e retoma de onde parou (pula linhas já julgadas).

REQUISITOS: Ollama rodando com  ollama pull llama3.1:8b

Uso:
    python juiz_llm.py                 # julga dataset_c1..c5.csv na pasta atual
    python juiz_llm.py .               # idem, pasta explícita
    python juiz_llm.py datasets/       # julga os dataset_c*.csv dentro de datasets/
    python juiz_llm.py datasets/ 3     # julga apenas o dataset_c3.csv
"""
import os
import re
import sys
import json

import pandas as pd
import ollama

LLM = "llama3.1:8b"

COMP_DESC = {
    1: "C1 - Domínio da norma-padrão da língua escrita (gramática, ortografia, pontuação, registro).",
    2: "C2 - Compreensão da proposta e estrutura dissertativo-argumentativa, com tese clara.",
    3: "C3 - Seleção, organização e interpretação de argumentos e repertório em defesa de um ponto de vista.",
    4: "C4 - Mecanismos de coesão (conectivos, referenciação) a serviço da argumentação.",
    5: "C5 - Proposta de intervenção completa (agente, ação, meio, finalidade, detalhamento).",
}

SYSTEM = """Você é um avaliador imparcial de FEEDBACKS de redação do ENEM. Você recebe a \
proposta, a redação do aluno, a competência em foco e DOIS feedbacks (A e B) sobre essa \
competência. Sua tarefa é decidir qual feedback é MELHOR.

Considere, nesta ordem:
1. Precisão: o feedback descreve corretamente o que está (ou não) na redação?
2. Especificidade: cita trechos concretos e aponta problemas/acertos reais, sem frases genéricas?
3. Utilidade: orienta o aluno de forma clara sobre como melhorar?
4. Coerência com a competência avaliada.

NÃO favoreça um feedback por causa da posição (A ou B) nem por ser mais longo; avalie só o \
conteúdo. Você é obrigado a escolher um vencedor.

Responda SOMENTE com um JSON válido, sem texto fora dele: {"escolha":"A"} ou {"escolha":"B"}"""


def montar_user(prompt, essay, n, fb_a, fb_b):
    return (
        f"PROPOSTA:\n{prompt}\n\n"
        f"REDAÇÃO DO ALUNO:\n{essay}\n\n"
        f"COMPETÊNCIA AVALIADA: {COMP_DESC[n]}\n\n"
        f"[FEEDBACK A]\n{fb_a}\n\n"
        f"[FEEDBACK B]\n{fb_b}\n\n"
        'Qual feedback é melhor? Responda só com {"escolha":"A"} ou {"escolha":"B"}.'
    )


def _parse_escolha(txt):
    # 1) tenta JSON
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if m:
        try:
            esc = str(json.loads(m.group(0)).get("escolha", "")).strip().upper()
            if esc in ("A", "B"):
                return esc
        except json.JSONDecodeError:
            pass
    # 2) fallback: primeira letra A ou B isolada na resposta
    m = re.search(r"\b([AB])\b", txt.upper())
    return m.group(1) if m else ""


def julgar(prompt, essay, n, fb_a, fb_b):
    resp = ollama.chat(
        model=LLM,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": montar_user(prompt, essay, n, fb_a, fb_b)}],
        options={"num_ctx": 8192, "temperature": 0.0},
    )
    return _parse_escolha(resp["message"]["content"])


def processar_dataset(caminho, n):
    if not os.path.isfile(caminho):
        print(f"[!] Pulando: não encontrei '{caminho}'.")
        return

    df = pd.read_csv(caminho, dtype=str, keep_default_na=False, encoding="utf-8")
    if "escolha_llm" not in df.columns:
        df["escolha_llm"] = ""

    total = len(df)
    print(f"\n=== dataset_c{n} ({total} linhas) ===")
    for i, linha in df.iterrows():
        if str(linha.get("escolha_llm", "")).strip():      # já julgada -> retoma
            continue
        prompt = str(linha.get("prompt", "")).strip()
        essay = str(linha.get("essay_text", "")).strip()
        fb_a = str(linha.get("entrada_A", "")).strip()
        fb_b = str(linha.get("entrada_B", "")).strip()
        if not fb_a or not fb_b:
            print(f"[{i + 1}/{total}] feedback A/B faltando — pulando.")
            continue

        try:
            esc = julgar(prompt, essay, n, fb_a, fb_b)
        except Exception as e:                              # noqa: BLE001
            print(f"[{i + 1}/{total}] ERRO: {e} — pulando.")
            continue

        if esc not in ("A", "B"):
            print(f"[{i + 1}/{total}] juiz não devolveu A/B — deixando em branco.")
            continue

        df.at[i, "escolha_llm"] = esc
        df.to_csv(caminho, index=False, encoding="utf-8")   # salva a cada linha
        print(f"[{i + 1}/{total}] escolha = {esc}")

    print(f"Concluído: '{caminho}'.")


if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else "."
    if len(sys.argv) > 2:                                   # um único C{n}
        n = int(sys.argv[2])
        processar_dataset(os.path.join(pasta, f"dataset_c{n}.csv"), n)
    else:                                                   # todos: C1..C5
        for n in range(1, 6):
            processar_dataset(os.path.join(pasta, f"dataset_c{n}.csv"), n)
        print("\nTodos os datasets processados.")
