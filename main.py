import streamlit as st
import wikipedia
import pandas as pd
import numpy as np
import json, os, re, unicodedata
import plotly.express as px
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

st.set_page_config(page_title="Detetive de Séries (Wikipédia) | BM25 vs Embeddings", layout="wide")

ARQUIVO_CACHE_WIKI = "series_wikipedia.json"

# Lista mapeada com os títulos EXATOS da Wikipédia em Português para evitar erro de desambiguação
# Formato: (Nome de Exibição, Título Exato na Wikipédia PT, Categoria/Gênero)
SERIES_SEMENTE = [
    ("Breaking Bad", "Breaking Bad", "Crime / Drama"),
    ("Stranger Things", "Stranger Things", "Ficção / Terror"),
    ("Todo Mundo Odeia o Chris", "Everybody Hates Chris", "Comédia"),
    ("Chaves", "Chaves (série de televisão)", "Comédia Clássica"),
    ("Round 6", "Squid Game", "Suspense / Sobrevivência"),
    ("La Casa de Papel", "La casa de papel", "Ação / Assalto"),
    ("The Last of Us", "The Last of Us (série de televisão)", "Pós-Apocalíptico"),
    ("Game of Thrones", "Game of Thrones", "Fantasia Medieval"),
    ("The Office", "The Office (Estados Unidos)", "Comédia"),
    ("Friends", "Friends", "Comédia / Sitcom"),
    ("Dark", "Dark (série de televisão)", "Ficção Científica"),
    ("Black Mirror", "Black Mirror", "Ficção / Distopia"),
    ("The Boys", "The Boys (série de televisão)", "Ação / Sátira"),
    ("Wandinha", "Wednesday (série de televisão)", "Mistério / Fantasia"),
    ("Peaky Blinders", "Peaky Blinders (série de televisão)", "Crime Histórico"),
    ("Grey's Anatomy", "Grey's Anatomy", "Drama Médico"),
    ("Narcos", "Narcos", "Crime / Biografia"),
    ("One Piece", "One Piece (série de televisão)", "Aventura"),
    ("The Walking Dead", "The Walking Dead (série de televisão)", "Pós-Apocalíptico"),
    ("Prison Break", "Prison Break", "Ação / Suspense"),
    ("Lost", "Lost (série de televisão)", "Mistério / Aventura"),
    ("A Grande Família", "A Grande Família (2001)", "Comédia Nacional"),
    ("Os Simpsons", "The Simpsons", "Animação / Comédia"),
    ("Rick and Morty", "Rick and Morty", "Animação / Ficção"),
    ("House", "House, M.D.", "Drama Médico"),
    ("Supernatural", "Supernatural (série de televisão)", "Fantasia / Terror"),
    ("Vikings", "Vikings (série de televisão)", "Drama Histórico"),
    ("Chernobyl", "Chernobyl (minissérie)", "Drama Histórico"),
    ("Bridgerton", "Bridgerton", "Romance de Época"),
    ("Suits", "Suits", "Drama Jurídico")
]

def extrair_sinopse_wikipedia(titulo_wiki):
    """
    Busca a página da Wikipédia em português e junta o resumo inicial com a seção de
    Sinopse/Enredo/Premissa (se existir), limpando notas de rodapé como [1], [2].
    """
    pagina = wikipedia.page(titulo_wiki, auto_suggest=False)
    resumo = wikipedia.summary(titulo_wiki, sentences=4, auto_suggest=False)
    
    # Tenta capturar seções ricas de história dentro do conteúdo completo da página
    conteudo = pagina.content
    secoes_alvo = ["Sinopse", "Enredo", "Premissa", "História", "Trama"]
    trecho_enredo = ""
    
    for secao in secoes_alvo:
        padrao = re.compile(rf"==+\s*{secao}\s*==+\n(.*?)(?=\n==+|\Z)", re.DOTALL | re.IGNORECASE)
        match = padrao.search(conteudo)
        if match:
            trecho_enredo = match.group(1).strip()
            # Pega os primeiros parágrafos do enredo para não ficar gigante
            trecho_enredo = " ".join(trecho_enredo.split("\n")[:3])
            break
            
    texto_completo = f"{resumo} {trecho_enredo}".strip()
    # Remove marcações [1], [nota 1] e espaços duplos
    texto_limpo = re.sub(r'\[\d+\]|\[nota \d+\]', '', texto_completo)
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo)
    return texto_limpo[:900] # Limita a ~900 caracteres para leitura rápida no estande

@st.cache_data(show_spinner=False)
def carregar_ou_baixar_series():
    if os.path.exists(ARQUIVO_CACHE_WIKI):
        with open(ARQUIVO_CACHE_WIKI, "r", encoding="utf-8") as f:
            return json.load(f)
            
    wikipedia.set_lang("pt")
    series_coletadas = []
    barra = st.progress(0, text="📥 Baixando artigos das séries na Wikipédia PT-BR (só acontece na 1ª vez)...")
    
    total = len(SERIES_SEMENTE)
    for idx, (nome_exibicao, titulo_wiki, genero) in enumerate(SERIES_SEMENTE):
        barra.progress((idx + 1) / total, text=f"📥 Baixando da Wikipédia: {nome_exibicao} ({idx+1}/{total})...")
        try:
            texto_wiki = extrair_sinopse_wikipedia(titulo_wiki)
            series_coletadas.append({
                "id": idx + 1,
                "titulo": nome_exibicao,
                "titulo_wiki": titulo_wiki,
                "genero": genero,
                "sinopse": texto_wiki
            })
        except Exception as e:
            # Fallback caso algum título específico oscile na API
            try:
                resumo_simples = wikipedia.summary(titulo_wiki, sentences=5)
                series_coletadas.append({
                    "id": idx + 1,
                    "titulo": nome_exibicao,
                    "titulo_wiki": titulo_wiki,
                    "genero": genero,
                    "sinopse": resumo_simples
                })
            except Exception:
                continue
                
    barra.empty()
    with open(ARQUIVO_CACHE_WIKI, "w", encoding="utf-8") as f:
        json.dump(series_coletadas, f, ensure_ascii=False, indent=2)
        
    return series_coletadas

def tokenizar(texto):
    nfkd = unicodedata.normalize('NFKD', texto.lower())
    sem_acento = "".join([c for c in nfkd if not unicodedata.combining(c)])
    stopwords = {
        "de", "da", "do", "das", "dos", "em", "um", "uma", "os", "as", "que", 
        "para", "com", "no", "na", "por", "se", "ao", "mais", "foi", "sao", 
        "serie", "televisao", "temporada", "episodios", "transmitida"
    }
    return [t for t in re.findall(r'\b[a-z0-9]{2,}\b', sem_acento) if t not in stopwords]

def grifar_termos_exatos(texto, tokens_busca):
    texto_mod = texto
    for t in set(tokens_busca):
        padrao = re.compile(rf'\b({re.escape(t)}\w*)\b', re.IGNORECASE)
        texto_mod = padrao.sub(r'**:orange[\1]**', texto_mod)
    return texto_mod

@st.cache_resource
def preparar_motores(series):
    corpus_tokens = [tokenizar(f"{s['titulo']} {s['genero']} {s['sinopse']}") for s in series]
    bm25 = BM25Okapi(corpus_tokens)
    
    modelo_emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    textos_ricos = [f"{s['titulo']} ({s['genero']}): {s['sinopse']}" for s in series]
    vetores = modelo_emb.encode(textos_ricos, normalize_embeddings=True)
    return bm25, modelo_emb, vetores

# --- INTERFACE DO ESTANDE ---
st.title("🎬 Detetive de Séries na Wikipédia: BM25 vs. Embeddings")
st.markdown("Textos extraídos **diretamente da Wikipédia em Português**! Tente descrever uma série com as suas palavras e veja quem encontra melhor: o buscador de palavras exatas (**BM25**) ou o buscador semântico (**Embeddings**).")

series_db = carregar_ou_baixar_series()
bm25, modelo_emb, vetores = preparar_motores(series_db)

with st.sidebar:
    st.header("📚 Base da Wikipédia")
    st.success(f"✅ {len(series_db)} séries indexadas localmente (`{ARQUIVO_CACHE_WIKI}`).")
    if st.button("🔄 Forçar Novo Download da Wikipédia"):
        if os.path.exists(ARQUIVO_CACHE_WIKI):
            os.remove(ARQUIVO_CACHE_WIKI)
        st.cache_data.clear()
        st.rerun()
    st.divider()
    with st.expander("Ver lista de séries na base"):
        for s in series_db:
            st.caption(f"• **{s['titulo']}** (*{s['genero']}*)")

st.markdown("##### 🎯 Teste um desafio (ou digite sua própria descrição):")
c1, c2, c3, c4 = st.columns(4)
# Desafios escolhidos com sinônimos que NÃO aparecem nos textos da Wikipédia,
# para mostrar o Embedding acertando onde o BM25 se confunde.
busca_padrao = "explosão de reator atômico"

if c1.button("☢️ Reator que explodiu"):
    busca_padrao = "explosão de reator atômico"
if c2.button("💼 Rotina no escritório"):
    busca_padrao = "rotina de funcionários num escritório"
if c3.button("🛋️ Amigos no apartamento"):
    busca_padrao = "turma de amigos que vive num apartamento"
if c4.button("🐉 Dragões e o trono"):
    busca_padrao = "reinos lutando pelo trono com dragões"

query = st.text_input("🔍 Descreva a série sem dizer o nome dela:", value=busca_padrao)

if query:
    tokens_q = tokenizar(query)
    
    # 1. BM25 (Léxico)
    scores_bm25 = bm25.get_scores(tokens_q) if tokens_q else np.zeros(len(series_db))
    rank_bm25 = np.argsort(scores_bm25)[::-1][:3]
    
    # 2. Embeddings (Semântico)
    vetor_q = modelo_emb.encode([query], normalize_embeddings=True)
    scores_emb = np.dot(vetores, vetor_q.T).flatten()
    rank_emb = np.argsort(scores_emb)[::-1][:3]

    # Banner de Feedback Gamificado
    if scores_bm25[rank_bm25[0]] == 0 and scores_emb[rank_emb[0]] > 0.40:
        st.success(f"🏆 **VITÓRIA DA SEMÂNTICA!** O BM25 não achou nenhuma palavra exata nos artigos da Wikipédia, mas o modelo de **Embeddings** entendeu o contexto e cravou **{series_db[rank_emb[0]]['titulo']}**!")
    elif rank_bm25[0] != rank_emb[0]:
        st.info(f"⚡ **Divergência no Topo!** Pelas palavras exatas da Wikipédia, o BM25 escolheu **{series_db[rank_bm25[0]]['titulo']}**, mas pelo significado geral o Embedding escolheu **{series_db[rank_emb[0]]['titulo']}**.")

    col_bm25, col_emb = st.columns(2)

    with col_bm25:
        st.subheader("🔤 Modelo 1: BM25 (Léxico)")
        st.caption("Caça as palavras exatas que você digitou dentro do artigo da Wikipédia.")
        
        if scores_bm25[rank_bm25[0]] == 0:
            st.error("❌ **Score 0.0!** Nenhuma das palavras que você usou aparece literalmente nos resumos da Wikipédia.")
        else:
            for pos, idx in enumerate(rank_bm25, start=1):
                nota = scores_bm25[idx]
                if nota > 0:
                    s = series_db[idx]
                    with st.container(border=True):
                        st.markdown(f"**#{pos} — {s['titulo']}** (*{s['genero']}*) | `Score BM25: {nota:.2f}`")
                        st.markdown(grifar_termos_exatos(s['sinopse'], tokens_q))

    with col_emb:
        st.subheader("🧠 Modelo 2: Embeddings (Semântico)")
        st.caption("Transforma o artigo da Wikipédia em coordenadas geométricas e busca pela ideia.")
        
        for pos, idx in enumerate(rank_emb, start=1):
            sim = float(scores_emb[idx])
            s = series_db[idx]
            with st.container(border=True):
                st.markdown(f"**#{pos} — {s['titulo']}** (*{s['genero']}*) | `Similaridade: {sim*100:.1f}%`")
                st.progress(max(0.0, min(1.0, sim)))
                st.write(s['sinopse'])

    # --- MAPA 2D DAS SÉRIES ---
    st.divider()
    st.subheader("🗺️ Mapa Vetorial das Séries da Wikipédia")
    
    matriz_completa = np.vstack([vetores, vetor_q])
    coords = PCA(n_components=2).fit_transform(matriz_completa)
    
    df_mapa = pd.DataFrame({
        "X": coords[:, 0], "Y": coords[:, 1],
        "Série": [s["titulo"] for s in series_db] + ["⭐ SUA BUSCA"],
        "Gênero": [s["genero"] for s in series_db] + ["Sua Busca"],
        "Tamanho": [12]*len(series_db) + [28]
    })
    
    fig = px.scatter(
        df_mapa, x="X", y="Y", text="Série", color="Gênero", size="Tamanho",
        height=480, title="Projeção 2D dos Artigos da Wikipédia (Séries com temas parecidos ficam próximas)"
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)