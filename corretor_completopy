"""
corretor_completo.py — sistema integrado de correção de redações ENEM.

  BERTimbau (notas precisas)  +  Llama via Ollama (feedback explicativo)

Cada modelo na função em que é melhor: o BERTimbau dá as 5 notas; o Llama escreve
as justificativas e o feedback COERENTES com essas notas (sem alterá-las).

REQUISITOS:
  - O BERTimbau já treinado: arquivos bertimbau_redacao.pt e a pasta
    bertimbau_redacao_tok/ (gerados pelo treino), na pasta atual.
  - O Ollama rodando com o modelo:  ollama pull llama3.1:8b
Rode de dentro da pasta essay-br:
    python corretor_completo.py
"""
import json
import re
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import ollama

# ---------------- Configuração ----------------
BERT_NAME = "neuralmind/bert-large-portuguese-cased"   # o MESMO usado no treino
MAXLEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Se faltar memória de GPU com o Ollama rodando junto, troque a linha acima por:
# DEVICE = "cpu"   (a inferência de uma redação é rápida na CPU também)
LLM = "llama3.1:8b"
COMP = ["c1", "c2", "c3", "c4", "c5"]
BANDAS = np.array([0, 40, 80, 120, 160, 200])
NOMES = {
    "c1": "C1 - Norma culta",
    "c2": "C2 - Compreensão da proposta e estrutura",
    "c3": "C3 - Argumentação e repertório",
    "c4": "C4 - Coesão",
    "c5": "C5 - Proposta de intervenção",
}


# ================= BERTimbau: as NOTAS =================
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


print("Carregando o BERTimbau treinado...")
try:
    _tok = AutoTokenizer.from_pretrained("bertimbau_redacao_tok")
    _model = BertScorer(BERT_NAME).to(DEVICE)
    _model.load_state_dict(torch.load("bertimbau_redacao.pt", map_location=DEVICE))
    _model.eval()
except FileNotFoundError:
    raise SystemExit("[!] Não encontrei o modelo treinado (bertimbau_redacao.pt / "
                     "bertimbau_redacao_tok/). Rode o treino antes, na pasta essay-br.")


def _para_banda(v):
    pontos = v * 200.0
    idx = np.abs(pontos[:, :, None] - BANDAS[None, None, :]).argmin(axis=2)
    return BANDAS[idx]


@torch.no_grad()
def dar_notas(tema_curto, corpo):
    enc = _tok(f"{tema_curto} [SEP] {corpo}", truncation=True, max_length=MAXLEN,
               padding="max_length", return_tensors="pt").to(DEVICE)
    notas = _para_banda(_model(**enc).cpu().numpy())[0]
    d = {COMP[i]: int(notas[i]) for i in range(5)}
    d["nota_final"] = int(sum(d[c] for c in COMP))
    return d


# ================= Llama: o FEEDBACK =================
SYSTEM_FB = """Você é um corretor de redações do ENEM. As NOTAS de cada competência (de 0 a \
200) JÁ FORAM definidas por um avaliador e você NÃO deve alterá-las nem questioná-las. Sua \
tarefa é, para CADA competência, escrever uma justificativa que explique aquela nota — \
citando trechos específicos da redação — e indicar o que o aluno pode melhorar.

Mantenha a coerência com a nota: se a nota é baixa, aponte os problemas que a justificam; \
se é alta, destaque os acertos e diga o que faltaria para o nível máximo.

Competências: C1 - norma culta; C2 - compreensão da proposta e estrutura; C3 - argumentação \
e repertório; C4 - coesão; C5 - proposta de intervenção.

No feedback geral, foque nas competências de nota mais baixa e dê os próximos passos.

Responda SOMENTE com um JSON válido, sem texto fora dele:
{"c1":{"justificativa":"..."},"c2":{"justificativa":"..."},"c3":{"justificativa":"..."},
 "c4":{"justificativa":"..."},"c5":{"justificativa":"..."},"feedback_geral":"..."}"""


# ---- Few-shot: 3 exemplos (nota baixa, média e alta) ----
# As notas vêm do BERTimbau; aqui servem só para o Llama aprender o formato e o
# tom das justificativas coerentes com a nota. (Exemplos das 3 redações do
# corretor_redacao4.py, adaptados para o formato deste corretor.)
EXEMPLOS_FEWSHOT = [
    {  # baixa — 360/1000
        "proposta": (
            "No dia 15 do mês de janeiro passado, o presidente Jair Bolsonaro assinou o "
            "decreto que facilita a posse de armas no Brasil, em meio a críticas a governos "
            "anteriores. A medida foi publicada em edição extra do \"Diário Oficial da União\" "
            "e teve efeito imediato. Já está em vigor. Facilitar o acesso às armas de fogo foi "
            "uma das bandeiras de campanha de Bolsonaro, quando candidato, em nome do direito "
            "à legítima defesa. No entanto, a posse de armas é uma questão muito polêmica. Com "
            "a ajuda dos argumentos apresentados na coletânea e com suas próprias ideias sobre "
            "o assunto, redija uma dissertação argumentativa, posicionando-se em relação ao "
            "decreto presidencial: você acha que ele aumentará a segurança dos cidadãos ou vai "
            "aumentar o nível de violência no campo e nas cidades brasileiras?"
        ),
        "corpo": (
            "Tema muito discutido atualmente, e que teve grande influência nas eleições de "
            "2018. A posse de arma vem gerando debates polêmicos e dividindo o país entre prós "
            "e contras, mas não é uma questão tão simples como aparenta ser. O projeto de lei "
            "jà decretado flexibiliza algumas exigências para a posse de armas de calibre "
            "irrestrito. O que em tese aumentaria a segurança, mas e inegável que teriamos um "
            "aumento significativo na taxa de homicídios, será que na sociedade em que vivemos "
            "hoje estamos devidamente preparados para o uso de armas de fogo.\n"
            "Esse armamento da população também vai contra a ideia de que a violência não é a "
            "melhor opção. E que através da educação encontraremos o caminho da paz.\n"
            "Portanto uma solução eficiente seria, a implantação de projetos educacionais para "
            "crianças e jovens em comunidades e áreas com alto índice de criminalidade, por "
            "parte do ministério da educação e projetos socias. E para um combate ao crime "
            "organizado, mais investimento em forças policiais e políticas públicas de inserção "
            "de reeducandos na sociedade por parte do governo e empresas. Por que como disse "
            "Nelson Mandela, a educação e a arma mais poderosa pra se mudar o mundo."
        ),
        "notas": {"c1": 80, "c2": 80, "c3": 40, "c4": 80, "c5": 80, "nota_final": 360},
        "correcao": {
            "c1": {"justificativa": (
                'A redação apresenta diversos desvios da norma-padrão. Um exemplo é o trecho '
                '"teriamos um aumento significativo", em que o correto seria "teríamos". Há '
                'também problemas em "Por que como disse Nelson Mandela", quando o adequado '
                'seria "Porque, como disse Nelson Mandela,", além de falhas de acentuação em '
                'palavras como "já", "é" e "à" ao longo do texto. Esses erros comprometem o '
                'domínio da modalidade escrita formal. Para melhorar, o aluno deve revisar '
                'regras de acentuação gráfica, uso dos porquês e pontuação, e reler o texto '
                'com cuidado antes da entrega.'
            )},
            "c2": {"justificativa": (
                'O texto compreende parcialmente a proposta e se posiciona contra a '
                'flexibilização da posse de armas, como no questionamento "será que na '
                'sociedade em que vivemos hoje estamos devidamente preparados para o uso de '
                'armas de fogo". Entretanto, o desenvolvimento é superficial: a proposta exigia '
                'analisar se o decreto aumentaria a segurança ou ampliaria a violência, e isso é '
                'tratado de forma breve. Para subir, discuta mais diretamente os impactos do '
                'decreto na segurança pública, aprofundando a tese.'
            )},
            "c3": {"justificativa": (
                'A argumentação é muito limitada e pouco desenvolvida. O autor afirma que '
                '"teríamos um aumento significativo na taxa de homicídios", mas não apresenta '
                'dados, exemplos ou repertórios que sustentem a afirmação. O segundo parágrafo '
                'resume-se à ideia de que a violência não é a melhor opção e de que a educação '
                'leva à paz, sem aprofundamento. Para melhorar, desenvolva cada argumento com '
                'causas, consequências e repertórios socioculturais pertinentes ao tema.'
            )},
            "c4": {"justificativa": (
                'A redação usa alguns mecanismos de coesão, como "mas", "também" e "Portanto", '
                'que ajudam a relacionar ideias. No entanto, a passagem do primeiro para o '
                'segundo parágrafo é abrupta, sem conexão elaborada entre os argumentos, e há '
                'encadeamento pouco fluido. Para avançar, amplie o repertório de conectivos e '
                'construa relações mais claras entre parágrafos e argumentos.'
            )},
            "c5": {"justificativa": (
                'A proposta de intervenção apresenta agentes e ações, como a "implantação de '
                'projetos educacionais para crianças e jovens" pelo Ministério da Educação e '
                '"mais investimento em forças policiais" pelo governo. Contudo, as medidas são '
                'genéricas, sem detalhamento de como seriam implementadas e como produziriam os '
                'resultados esperados. Para melhorar, detalhe os meios de execução, explicando '
                'como os projetos funcionariam e de que maneira reduziriam a violência.'
            )},
            "feedback_geral": (
                'Ponto mais frágil: a argumentação (C3, nota 40), sem dados ou repertórios que '
                'sustentem a tese; o desenvolvimento da discussão (C2) também está superficial. '
                'Próximos passos: (1) sustente a afirmação sobre o aumento de homicídios com '
                'dados ou repertório sociocultural; (2) aprofunde os impactos do decreto na '
                'segurança pública; (3) detalhe os meios de execução da proposta de intervenção.'
            ),
        },
    },
    {  # média — 680/1000
        "proposta": (
            'Em meados desta década, que se aproxima do fim, diversos estudiosos brasileiros e '
            'latino-americanos começaram a falar no surgimento de uma "onda conservadora" na '
            'política e na sociedade de diversos países da América do Sul. No mês passado, com '
            'a eleição de Jair Bolsonaro à Presidência da República, essa onda teria '
            'definitivamente chegado ao Brasil, após uma campanha marcada pela polarização '
            'entre a direita e a esquerda. Nos textos da coletânea que informa essa proposta de '
            'redação, você encontrará elementos que o farão refletir sobre a própria ideia de '
            'uma onda conservadora. A partir deles e de seus próprios conhecimentos sobre o '
            'assunto, redija uma dissertação argumentativa, explicando como você vê esse '
            'fenômeno da onda conservadora, se é que ele existe, e o que acredita que vai '
            'acontecer no Brasil, durante os próximos quatro anos, tanto em termos políticos '
            'quanto em termos sociais. Você encara a mudança de orientação política do país de '
            'modo positivo ou negativo? Por quê?'
        ),
        "corpo": (
            'A maioria dos países da América do Sul passaram para o lado conservador. '
            'Paralelamente a isso, o Brasil presenciou na última eleição presidencial uma '
            'dicotomia política: direita e esquerda. O primeiro é conservador; o segundo, '
            'socialista. Em uma disputa à presidência acirrada entre esses dois polos opostos, '
            'o país decidiu democraticamente pelo conservadorismo. Mas afinal, o que mudará '
            'daqui pra frente no país com a nova "onda conservadora"? Há muito assunto em '
            'discussão, mas o principal é de que o país entraria em uma ditadura militar. Antes '
            'de tudo, o presidente deve honrar a Constituição Brasileira de 1988 (CF/88), ou '
            'seja, não pode desrespeitar a lei maior por ser presidente. Além disso, tal questão '
            'feriria as cláusulas pétreas: princípios fundamentais que regem o país, guiando-o '
            'juntamente com a democracia. Com o Jair Bolsonaro no comando do país, o exposto '
            'acima poderia não ser seguido, já que é ultraconservador, e também porque a metade '
            'das cadeiras do Congresso Nacional estão ocupadas por direitistas. Dessa forma, se '
            'dependesse desses líderes para trazer o período militar, tal iniciativa até que se '
            'confirmaria, mas a questão é que não depende, há outros coadjuvantes que '
            'analisariam essa atitude contraia à democracia, como o Supremo Tribunal Federal '
            '(STF), guardião da CF/88. E também porque a população brasileira não aceitaria '
            'apagar sua história democrática em ascensão para regredir ao passado ditatorial. '
            'Além disso, fala-se também na legalização do porte de arma para o brasileiro e, '
            'devido a isso, poderia ser o holocausto dos homossexuais. Não é bem assim. Nas '
            'redes sociais circulam muitas notícias falsas a respeito dessa ideia. Não é porque '
            'o país mudou de esquerda-liberal para direita-extremista que isso vai acontecer. '
            'Pode ser que ocorra em alguns casos isolados visto que o país ainda é muito '
            'homofóbico. Conclui-se, portanto, que há muito em jogo em ralação a um novo período '
            'com um presidente conversador. Este, caso seja contrário à democracia, temos o '
            'impeachment para tirá-lo. Seja de direta ou de esquerda, isso não importa mais, o '
            'importante é que o Brasil precisa de mudanças políticas e socias positivas e '
            'efetivas antes de qualquer opinião precipitada sobre essa nova hegemonia.'
        ),
        "notas": {"c1": 160, "c2": 160, "c3": 120, "c4": 120, "c5": 120, "nota_final": 680},
        "correcao": {
            "c1": {"justificativa": (
                'Bom domínio da norma-padrão, com alguns desvios de revisão. Exemplos: "em '
                'ralação a um novo período com um presidente conversador", onde o correto seria '
                '"relação" e "conservador"; "contraia à democracia", em vez de "contrária à '
                'democracia"; e "socias positivas", em vez de "sociais positivas". Esses erros '
                'não comprometem a compreensão, mas são incompatíveis com os níveis mais altos '
                'da competência. Para subir, faça uma revisão ortográfica mais cuidadosa, atenta '
                'a palavras frequentes e a concordâncias.'
            )},
            "c2": {"justificativa": (
                'O texto compreende a proposta e reflete sobre a "onda conservadora", com tese '
                'apresentada na introdução e retomada no desenvolvimento. No entanto, concentra-se '
                'quase só em dois pontos (ditadura militar e violência contra homossexuais), '
                'deixando de explorar de forma mais ampla as transformações políticas e sociais '
                'pedidas. Para a nota máxima, analise também impactos econômicos, educacionais, '
                'culturais ou institucionais.'
            )},
            "c3": {"justificativa": (
                'Há argumentos pertinentes, como os mecanismos institucionais de controle '
                '(Constituição, STF e impeachment): "há outros coadjuvantes que analisariam essa '
                'atitude contrária à democracia, como o Supremo Tribunal Federal". Porém, parte '
                'da argumentação fica em hipóteses pouco aprofundadas: ao dizer que "poderia ser '
                'o holocausto dos homossexuais" e logo descartar a ideia, o texto não sustenta os '
                'motivos nem traz repertório externo. Para melhorar, use exemplos históricos, '
                'dados ou conceitos jurídicos/sociológicos que fortaleçam o ponto de vista.'
            )},
            "c4": {"justificativa": (
                'A redação usa vários coesivos ("Além disso", "Antes de tudo", "Dessa forma", '
                '"Conclui-se, portanto"), o que ajuda na progressão. Contudo, algumas transições '
                'são bruscas: a passagem do debate sobre a ditadura para a discussão sobre '
                'homossexualidade ocorre sem conexão argumentativa elaborada. Para subir, '
                'construa transições mais consistentes, mostrando como cada argumento se liga à '
                'tese central.'
            )},
            "c5": {"justificativa": (
                'A proposta de intervenção aparece em "temos o impeachment para tirá-lo", uma '
                'ação ligada à preservação da democracia. Mas é incompleta: não há agente '
                'responsável, meio de execução, finalidade específica nem medidas concretas para '
                'os problemas discutidos. Para melhorar, proponha ações completas, indicando quem '
                'executaria, como seriam realizadas e que resultados buscariam.'
            )},
            "feedback_geral": (
                'Pontos mais frágeis: a proposta de intervenção (C5), incompleta, e a argumentação '
                '(C3), com hipóteses pouco sustentadas e sem repertório externo. Próximos passos: '
                '(1) reescreva a conclusão com uma proposta completa (agente, ação, meio, '
                'finalidade, detalhamento); (2) sustente os argumentos com dados ou repertório '
                'sociocultural; (3) amplie o recorte temático para além da ditadura e da violência '
                'contra homossexuais.'
            ),
        },
    },
    {  # alta — 920/1000
        "proposta": (
            'Reportagem publicada pelo UOL Economia no mês passado apresenta uma pesquisa '
            'realizada pelo Datafolha em que se perguntava aos entrevistados qual o fator mais '
            'importante para se conquistar uma vida melhor. A amostragem reflete toda a '
            'população do Brasil, com baixa margem de erro. Os resultados da pesquisa revelaram '
            'que, em primeiro lugar, os brasileiros consideram necessária a fé religiosa. De '
            'acordo com a pesquisa, as pessoas consideram a fé mais importante do que o estudo '
            'ou o trabalho, por exemplo, para melhorar de vida. Leia o texto do UOL que se '
            'transcreve abaixo, preste atenção nos percentuais e redija uma dissertação '
            'argumentativa apresentando sua opinião sobre a questão formulada na pesquisa: para '
            'você, dos itens mencionados, qual é o mais importante para melhorar de vida? Por '
            'quê? Apresente suas razões para justificar o seu ponto de vista.'
        ),
        "corpo": (
            '"Melhorar de vida", isto é, ascender socialmente, está relacionado aos mais '
            'diferentes motivos: sorte, herança da família, estudos, trabalho, fé, etc. Mas, o '
            'que, via de regra, influencia positivamente o status social de qualquer indivíduo '
            'em um país tão diverso como Brasil?\n'
            'Pesquisas socioeconômicas revelam um fator de alta relevância na ascensão de '
            'classe dos brasileiros de qualquer renda, etnia, macrorregião, entre outros: os '
            'estudos. Relatórios do Ministério da Educação apontam que indivíduos com Ensino '
            'Médio Técnico ganham, em média, vinte e oito por cento acima daqueles que '
            'concluíram Ensino Normal apenas. Além das melhorias a si próprio, quanto mais anos '
            'de escolaridade dos pais, maior é o nível educacional dos filhos e, '
            'consequentemente, sua classe social. Portanto, a educação é tão poderosa para a '
            'transformação socioeconômica que muda a vida da família toda.\n'
            'Entretanto, para vinte e oito por cento dos brasileiros, segundo a ONG Oxfam, a fé '
            'é o elemento preponderante para crescer na vida. Em paralelo, a Psicologia indica '
            'que a mentalidade positiva e o otimismo são de suma importância para atingirmos o '
            'sucesso; pois, isso bloqueia a autossabotagem, fortalecendo o indivíduo frente às '
            'dificuldades. E a fé, em uma sociedade majoritariamente teísta, é onde as pessoas '
            'buscam inspiração para superação de obstáculos.\n'
            'Sendo assim, salvo raras exceções, conforme os fatos acima, a educação é o '
            'elemento mais proeminente para ascender economicamente. Mesmo assim, o efeito da '
            'crença como combustível que potencializa a autoconfiança do ser humano em si mesmo '
            'frente aos obstáculos não deve ser negligenciado.'
        ),
        "notas": {"c1": 160, "c2": 200, "c3": 200, "c4": 200, "c5": 160, "nota_final": 920},
        "correcao": {
            "c1": {"justificativa": (
                'Bom domínio da norma-padrão, com vocabulário adequado e sintaxe bem elaborada, '
                'mas há pequenos desvios que impedem a nota máxima. Em "um país tão diverso como '
                'Brasil", o mais adequado seria "como o Brasil"; e "melhorias a si próprio" soa '
                'pouco natural no contexto. São problemas pontuais que não comprometem a '
                'compreensão. Para alcançar 200, faça uma revisão fina, buscando maior precisão '
                'gramatical e estilística.'
            )},
            "c2": {"justificativa": (
                'A redação compreende plenamente a proposta e desenvolve exatamente a discussão '
                'pedida. Desde a introdução, apresenta a questão dos fatores de ascensão social e '
                'defende com clareza a tese de que a educação é o mais importante para melhorar '
                'de vida, sem desvios ou tangenciamentos. Para manter esse nível, continue '
                'apresentando uma tese clara e desenvolvendo-a de forma consistente.'
            )},
            "c3": {"justificativa": (
                'Argumentação consistente e bem fundamentada. O autor usa repertório que sustenta '
                'a tese, como em "Relatórios do Ministério da Educação apontam que indivíduos com '
                'Ensino Médio Técnico ganham, em média, vinte e oito por cento acima daqueles que '
                'concluíram Ensino Normal apenas". Ainda articula a perspectiva da fé como fator '
                'psicológico de motivação sem abandonar a tese principal. Para manter, continue '
                'usando dados, exemplos e relações de causa e consequência.'
            )},
            "c4": {"justificativa": (
                'Excelente articulação textual: conectivos como "Portanto", "Entretanto", "Em '
                'paralelo" e "Sendo assim" estabelecem bem as relações lógicas, e há progressão '
                'temática clara entre os parágrafos até a conclusão. A coesão contribui '
                'efetivamente para a argumentação. Para manter a nota máxima, basta seguir usando '
                'os recursos coesivos de forma natural e funcional.'
            )},
            "c5": {"justificativa": (
                'A redação encerra com uma conclusão consistente, reafirmando a educação e '
                'reconhecendo o papel complementar da fé. Porém, não há uma proposta de ação '
                'concreta para ampliar o acesso à educação ou potencializar os benefícios '
                'discutidos. Após defender a educação como principal fator de ascensão, o autor '
                'poderia sugerir, por exemplo, políticas públicas de ampliação do ensino técnico '
                'e superior, indicando agentes, ações e objetivos.'
            )},
            "feedback_geral": (
                'Redação forte e bem acima da média, com C2, C3 e C4 no nível máximo. Os dois '
                'pontos a lapidar são a proposta de intervenção (C5), que carece de uma ação '
                'concreta com agente, meio e finalidade, e a norma culta (C1), que pede uma '
                'revisão fina de pequenos desvios. Próximos passos: (1) acrescente uma proposta '
                'de intervenção completa voltada à educação; (2) revise trechos como "como '
                'Brasil" e "melhorias a si próprio".'
            ),
        },
    },
]


def _mensagens_fewshot():
    """Monta as mensagens de few-shot no MESMO formato usado em dar_feedback."""
    msgs = []
    for ex in EXEMPLOS_FEWSHOT:
        notas = ex["notas"]
        notas_txt = ", ".join(f"{c.upper()}={notas[c]}" for c in COMP) + f", total={notas['nota_final']}"
        user = (f"PROPOSTA:\n{ex['proposta']}\n\nREDAÇÃO:\n{ex['corpo']}\n\n"
                f"NOTAS JÁ ATRIBUÍDAS (não altere): {notas_txt}\n\n"
                "Escreva as justificativas coerentes com essas notas e o feedback geral, em JSON.")
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": json.dumps(ex["correcao"], ensure_ascii=False)})
    return msgs


def _parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def dar_feedback(proposta, corpo, notas):
    notas_txt = ", ".join(f"{c.upper()}={notas[c]}" for c in COMP) + f", total={notas['nota_final']}"
    user = (f"PROPOSTA:\n{proposta}\n\nREDAÇÃO:\n{corpo}\n\n"
            f"NOTAS JÁ ATRIBUÍDAS (não altere): {notas_txt}\n\n"
            "Escreva as justificativas coerentes com essas notas e o feedback geral, em JSON.")
    resp = ollama.chat(
        model=LLM,
        messages=[{"role": "system", "content": SYSTEM_FB},
                  *_mensagens_fewshot(),
                  {"role": "user", "content": user}],
        options={"num_ctx": 16384, "temperature": 0.3},
    )
    return _parse_json(resp["message"]["content"]) or {}


# ================= Integração =================
def corrigir(proposta, corpo):
    tema_curto = " ".join(proposta.split())[:120]   # BERTimbau treinou com títulos curtos
    notas = dar_notas(tema_curto, corpo)            # 1) notas pelo BERTimbau
    fb = dar_feedback(proposta, corpo, notas)       # 2) feedback pelo Llama (recebe a proposta inteira)

    print("\n" + "=" * 70)
    print("CORREÇÃO")
    print("=" * 70)
    for c in COMP:
        just = fb.get(c, {}).get("justificativa", "(o modelo de feedback não retornou esta justificativa)")
        print(f"\n{NOMES[c]}  —  nota: {notas[c]}")
        print(f"  {just}")
    print("\n" + "-" * 70)
    print(f"NOTA FINAL: {notas['nota_final']} / 1000")
    print("-" * 70)
    print("FEEDBACK GERAL:")
    print(f"  {fb.get('feedback_geral', '(sem feedback geral)')}")
    print("=" * 70)


# ================= Modo interativo =================
def ler_multilinha(msg):
    print(msg)
    print("(cole o texto e, numa linha sozinha, digite  FIM  para terminar)")
    linhas = []
    while True:
        try:
            linha = input()
        except EOFError:
            break
        if linha.strip().upper() == "FIM":
            break
        linhas.append(linha)
    return "\n".join(linhas)


if __name__ == "__main__":
    print("\n=== Corretor completo: BERTimbau (notas) + Llama (feedback) ===\n")
    proposta = ler_multilinha("Informe a PROPOSTA (o tema; pode colar também os textos motivadores):")
    corpo = ler_multilinha("\nAgora cole a REDAÇÃO do aluno:")
    if not corpo.strip() or not proposta.strip():
        print("\nProposta ou redação vazia. Encerrando.")
    else:
        print("\nCorrigindo (BERTimbau dá as notas, Llama escreve o feedback)...")
        corrigir(proposta, corpo)
