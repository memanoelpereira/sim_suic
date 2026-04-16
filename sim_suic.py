import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass
from typing import Optional


# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================

st.set_page_config(
    page_title="Simulador BSEM de Risco de Ideação",
    layout="wide",
    page_icon="📊"
)


# =========================================================
# PARÂMETROS DO MODELO
# =========================================================
# Estrutura serial:
# X = Violência percebida
# M1 = Acolhimento escolar
# M2 = Satisfação com a vida
# Y = Ideação suicida

BETA_XY = 0.243
BETA_XM1 = -0.324
BETA_M1M2 = 0.527
BETA_M2Y = -0.170

# Efeitos indiretos reportados por nível de pertença
IND_BAIXA = 0.016
IND_MEDIA = 0.020
IND_ALTA = 0.024

MOD_BAIXA = IND_BAIXA / IND_MEDIA
MOD_ALTA = IND_ALTA / IND_MEDIA

# Médias populacionais
V_POP = 1.51
M1_POP = 3.56
M2_POP = 3.38
Y_POP = 1.69

# Colunas mínimas esperadas no CSV
COLUNAS_MINIMAS = [
    "percepcao_violencia",
    "acolhimento",
    "satisfacao_vida",
    "pertenca_grupal",
    "ideacao_suicida",
]

# Filtros opcionais exibidos se a coluna existir
FILTROS_CONFIG = {
    "sexo": "Sexo",
    "cor_da_pele": "Cor da pele",
    "orientação_sexual": "Orientação sexual",
    "renda": "Renda",
    "cidade": "Cidade",
    "série": "Série",
}


# =========================================================
# ESTRUTURAS DE DADOS
# =========================================================

@dataclass
class Interceptos:
    pertenca_alta: bool
    ic_m1: float
    ic_m2: float
    ic_y: float
    v_m: Optional[float] = None
    m1_m: Optional[float] = None
    m2_m: Optional[float] = None
    y_real: Optional[float] = None
    n: Optional[int] = None


@dataclass
class ResultadoSimulacao:
    y_basal: float
    y_pos_mediacao: float
    y_final: float
    reducao_mediada: float
    ajuste_m2: float
    m2_previsto_por_m1: float
    beta_m1m2_efetivo: float
    fator_mod: float
    efeito_direto_x: float
    efeito_indireto_x: float
    efeito_total_x: float
    saturado: bool


# =========================================================
# FUNÇÕES DE APOIO
# =========================================================

def beta_m1m2_efetivo(pertenca_alta: bool) -> float:
    return BETA_M1M2 * (MOD_ALTA if pertenca_alta else MOD_BAIXA)


def inferir_pertenca_alta(valor_pertenca: float) -> bool:
    """
    Regra simples e explícita.
    Ajuste aqui se sua variável tiver outra codificação.
    """
    return valor_pertenca >= 3.0


def interceptos_populacionais(pertenca_alta: bool) -> Interceptos:
    beta_eff = beta_m1m2_efetivo(pertenca_alta)

    ic_m1 = M1_POP - BETA_XM1 * V_POP
    ic_m2 = M2_POP - beta_eff * M1_POP
    ic_y = Y_POP - BETA_XY * V_POP - BETA_M2Y * M2_POP

    return Interceptos(
        pertenca_alta=pertenca_alta,
        ic_m1=ic_m1,
        ic_m2=ic_m2,
        ic_y=ic_y,
        v_m=V_POP,
        m1_m=M1_POP,
        m2_m=M2_POP,
        y_real=Y_POP,
        n=None
    )


def calibrar_interceptos_subgrupo(df_sub: pd.DataFrame) -> Interceptos:
    v_m = float(df_sub["percepcao_violencia"].mean())
    m1_m = float(df_sub["acolhimento"].mean())
    m2_m = float(df_sub["satisfacao_vida"].mean())
    p_m = float(df_sub["pertenca_grupal"].mean())
    y_m = float(df_sub["ideacao_suicida"].mean())

    pertenca_alta = inferir_pertenca_alta(p_m)
    beta_eff = beta_m1m2_efetivo(pertenca_alta)

    ic_m1 = m1_m - BETA_XM1 * v_m
    ic_m2 = m2_m - beta_eff * m1_m
    ic_y = y_m - BETA_XY * v_m - BETA_M2Y * m2_m

    return Interceptos(
        pertenca_alta=pertenca_alta,
        ic_m1=ic_m1,
        ic_m2=ic_m2,
        ic_y=ic_y,
        v_m=v_m,
        m1_m=m1_m,
        m2_m=m2_m,
        y_real=y_m,
        n=len(df_sub)
    )


@st.cache_data
def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv)

    faltantes = [c for c in COLUNAS_MINIMAS if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Colunas obrigatórias ausentes no CSV: {', '.join(faltantes)}"
        )

    df = df.dropna(subset=COLUNAS_MINIMAS).copy()
    return df


def classificar_risco(y: float) -> str:
    if y < 2.0:
        return "Baixa intensidade estimada"
    elif y < 3.0:
        return "Intensidade moderada estimada"
    elif y < 4.0:
        return "Intensidade elevada estimada"
    return "Intensidade muito elevada estimada"


def calcular_resultado(
    x: float,
    m1: float,
    m2: float,
    interceptos: Interceptos,
    usar_m1: bool,
    usar_m2: bool
) -> ResultadoSimulacao:
    """
    Lógica:
    - X sempre entra.
    - O caminho indireto pode ser ativado/desativado.
    - M1 atua sobre M2 se usar_m1=True.
    - M2 atua sobre Y se usar_m2=True.
    """
    pertenca_alta = interceptos.pertenca_alta
    beta_eff = beta_m1m2_efetivo(pertenca_alta)

    # Nó 1: X sempre presente, sem proteção mediada
    y_basal = interceptos.ic_y + (BETA_XY * x) + (BETA_M2Y * interceptos.ic_m2)

    # M2 previsto a partir de M1
    if usar_m1:
        m2_previsto = interceptos.ic_m2 + beta_eff * m1
    else:
        m2_previsto = interceptos.ic_m2

    # Nó 2: aplica M2 previsto via M1
    if usar_m2:
        y_pos_mediacao = interceptos.ic_y + (BETA_XY * x) + (BETA_M2Y * m2_previsto)
    else:
        y_pos_mediacao = y_basal

    # Nó 3: usa o M2 informado
    if usar_m2:
        y_final_raw = interceptos.ic_y + (BETA_XY * x) + (BETA_M2Y * m2)
    else:
        y_final_raw = y_basal

    y_final = float(np.clip(y_final_raw, 1.0, 5.0))
    saturado = (y_final != y_final_raw)

    reducao_mediada = y_basal - y_pos_mediacao
    ajuste_m2 = y_pos_mediacao - y_final_raw

    efeito_indireto_x = BETA_M2Y * beta_eff * BETA_XM1 * x if (usar_m1 and usar_m2) else 0.0
    efeito_direto_x = BETA_XY * x
    efeito_total_x = efeito_direto_x + efeito_indireto_x

    return ResultadoSimulacao(
        y_basal=y_basal,
        y_pos_mediacao=y_pos_mediacao,
        y_final=y_final,
        reducao_mediada=reducao_mediada,
        ajuste_m2=ajuste_m2,
        m2_previsto_por_m1=float(np.clip(m2_previsto, 1.0, 5.0)),
        beta_m1m2_efetivo=beta_eff,
        fator_mod=(MOD_ALTA if pertenca_alta else MOD_BAIXA),
        efeito_direto_x=efeito_direto_x,
        efeito_indireto_x=efeito_indireto_x,
        efeito_total_x=efeito_total_x,
        saturado=saturado
    )


def renderizar_cascata(resultado: ResultadoSimulacao, usar_m1: bool, usar_m2: bool) -> go.Figure:
    fig = go.Figure()

    x_labels = [
        "1. Pressão Basal por X",
        "2. Após Mediação",
        "3. Escore Final"
    ]

    y1 = resultado.y_basal
    y2 = resultado.y_pos_mediacao
    y3 = resultado.y_final

    # Faixas interpretativas
    fig.add_hrect(y0=1.0, y1=2.0, fillcolor="rgba(80,180,80,0.08)", line_width=0)
    fig.add_hrect(y0=2.0, y1=3.0, fillcolor="rgba(240,200,80,0.08)", line_width=0)
    fig.add_hrect(y0=3.0, y1=4.0, fillcolor="rgba(255,140,80,0.08)", line_width=0)
    fig.add_hrect(y0=4.0, y1=5.0, fillcolor="rgba(220,80,80,0.08)", line_width=0)

    fig.add_hline(
        y=5.0,
        line_dash="dot",
        line_color="red",
        annotation_text="Teto Likert = 5",
        annotation_position="top left"
    )
    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="green",
        annotation_text="Piso Likert = 1",
        annotation_position="bottom left"
    )

    # Guias horizontais
    fig.add_trace(go.Scatter(
        x=[x_labels[0], x_labels[1]],
        y=[y1, y1],
        mode="lines",
        line=dict(color="rgba(140,140,140,0.4)", width=2, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=[x_labels[1], x_labels[2]],
        y=[y2, y2],
        mode="lines",
        line=dict(color="rgba(140,140,140,0.4)", width=2, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Mudança 1
    if abs(resultado.reducao_mediada) > 0.005:
        cor1 = "rgba(220,50,50,0.85)" if resultado.reducao_mediada > 0 else "rgba(50,90,220,0.85)"
        fig.add_trace(go.Scatter(
            x=[x_labels[1], x_labels[1]],
            y=[y1, y2],
            mode="lines",
            line=dict(color=cor1, width=3),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_annotation(
            x=x_labels[1],
            y=(y1 + y2) / 2,
            text=f"<b>{'-' if resultado.reducao_mediada > 0 else '+'}{abs(resultado.reducao_mediada):.3f}</b>",
            showarrow=True,
            ax=45,
            ay=0,
            bgcolor="white",
            bordercolor=cor1,
            borderwidth=1,
        )

    # Mudança 2
    if abs(resultado.ajuste_m2) > 0.005:
        cor2 = "rgba(220,50,50,0.85)" if resultado.ajuste_m2 > 0 else "rgba(50,90,220,0.85)"
        fig.add_trace(go.Scatter(
            x=[x_labels[2], x_labels[2]],
            y=[y2, y3],
            mode="lines",
            line=dict(color=cor2, width=3),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_annotation(
            x=x_labels[2],
            y=(y2 + y3) / 2,
            text=f"<b>{'-' if resultado.ajuste_m2 > 0 else '+'}{abs(resultado.ajuste_m2):.3f}</b>",
            showarrow=True,
            ax=45,
            ay=0,
            bgcolor="white",
            bordercolor=cor2,
            borderwidth=1,
        )

    pontos = [
        (x_labels[0], y1, "#1f77b4", "#0e4e7d", "Base por X"),
        (x_labels[1], y2, "#ff7f0e", "#cc6600", "Após mediação"),
        (x_labels[2], y3, "#2ca02c", "#175e17", "Escore final"),
    ]

    for xlab, yval, cor, borda, nome in pontos:
        fig.add_trace(go.Scatter(
            x=[xlab],
            y=[yval],
            mode="markers+text",
            marker=dict(size=22, color=cor, line=dict(color=borda, width=2)),
            text=[f"<b>{yval:.3f}</b>"],
            textposition="top center",
            textfont=dict(size=14, color=cor),
            name=nome
        ))

    subtitulo = [
        "X sempre ativo",
        f"M1 {'ativo' if usar_m1 else 'desligado'}",
        f"M2 {'ativo' if usar_m2 else 'desligado'}"
    ]

    fig.update_layout(
        title=dict(
            text="Dinâmica Estrutural: Pressão Basal, Mediação e Escore Final<br><sup>" + " | ".join(subtitulo) + "</sup>",
            font=dict(size=18)
        ),
        yaxis=dict(
            title="Escore esperado de ideação (1–5)",
            range=[0.8, 5.4],
            gridcolor="rgba(210,210,210,0.35)"
        ),
        xaxis=dict(showgrid=False),
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=60, r=40, t=90, b=60),
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right")
    )
    return fig


def renderizar_diagrama_estrutural() -> go.Figure:
    fig = go.Figure()

    nos = {
        "X": (0.10, 0.55),
        "M1": (0.38, 0.55),
        "M2": (0.66, 0.55),
        "Y": (0.90, 0.55),
        "W": (0.38, 0.12),
    }

    labels = {
        "X": "Violência<br>(X)",
        "M1": "Acolhimento<br>(M1)",
        "M2": "Satisfação<br>(M2)",
        "Y": "Ideação<br>(Y)",
        "W": "Pertença<br>(modera M1→M2)",
    }

    cores = {
        "X": "#d62728",
        "M1": "#2ca02c",
        "M2": "#2ca02c",
        "Y": "#1f77b4",
        "W": "#ff7f0e",
    }

    for k, (x, y) in nos.items():
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{labels[k]}</b>",
            showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor=cores[k],
            bordercolor="white",
            borderwidth=2,
            borderpad=6,
            opacity=0.94
        )

    linhas = [
        ([0.14, 0.34], [0.55, 0.55]),
        ([0.42, 0.62], [0.55, 0.55]),
        ([0.70, 0.86], [0.55, 0.55]),
        ([0.14, 0.86], [0.67, 0.67]),
        ([0.38, 0.38], [0.18, 0.46]),
    ]

    for xs, ys in linhas:
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="gray", width=1.6),
            showlegend=False
        ))

    anotacoes = [
        (0.24, 0.61, "β X→M1 = -0.324", "#d62728"),
        (0.52, 0.61, "β M1→M2 = 0.527 × mod", "#2ca02c"),
        (0.78, 0.61, "β M2→Y = -0.170", "#2ca02c"),
        (0.50, 0.73, "β X→Y = 0.243", "#1f77b4"),
        (0.44, 0.30, "mod = 0.80–1.20", "#ff7f0e"),
    ]

    for x, y, txt, cor in anotacoes:
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<i>{txt}</i>",
            showarrow=False,
            font=dict(size=9, color=cor)
        )

    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 0.85]),
        plot_bgcolor="white",
        title=dict(text="Estrutura do Modelo Serial Moderado", x=0.5, font=dict(size=13))
    )
    return fig


def exibir_painel_tecnico(inter: Interceptos, r: ResultadoSimulacao, usar_m1: bool, usar_m2: bool):
    st.markdown("### Painel Técnico")

    c1, c2, c3 = st.columns(3)
    c1.metric("Efeito direto de X sobre Y", f"{r.efeito_direto_x:+.3f}")
    c2.metric("Efeito indireto estrutural de X", f"{r.efeito_indireto_x:+.3f}")
    c3.metric("Efeito total estrutural de X", f"{r.efeito_total_x:+.3f}")

    st.markdown("#### Interpretação resumida")

    texto = []
    reducao_mediacao = r.y_basal - r.y_pos_mediacao
    ajuste_adicional_m2 = r.y_pos_mediacao - r.y_final

    texto.append(
        f"O escore basal estimado de ideação foi **{r.y_basal:.3f}**, "
        f"representando a pressão esperada quando **X entra diretamente no modelo**."
    )

    if usar_m1 and usar_m2:
        if abs(reducao_mediacao) < 0.001:
            texto.append(
                f"A ativação do caminho mediado manteve o escore em **{r.y_pos_mediacao:.3f}**, "
                f"sem alteração relevante em relação ao nível basal."
            )
        elif reducao_mediacao > 0:
            texto.append(
                f"A ativação do caminho mediado reduziu esse valor para **{r.y_pos_mediacao:.3f}**, "
                f"com **redução de {reducao_mediacao:.3f} ponto(s)**."
            )
        else:
            texto.append(
                f"A ativação do caminho mediado elevou esse valor para **{r.y_pos_mediacao:.3f}**, "
                f"com **aumento de {abs(reducao_mediacao):.3f} ponto(s)**."
            )

        if abs(ajuste_adicional_m2) < 0.001:
            texto.append(
                f"O uso do valor informado de M2 manteve o escore final em **{r.y_final:.3f}**, "
                f"sem ajuste adicional relevante em relação ao valor já produzido pela mediação."
            )
        elif ajuste_adicional_m2 > 0:
            texto.append(
                f"O uso do valor informado de M2 reduziu adicionalmente o escore para **{r.y_final:.3f}**, "
                f"com **redução adicional de {ajuste_adicional_m2:.3f} ponto(s)**."
            )
        else:
            texto.append(
                f"O uso do valor informado de M2 elevou adicionalmente o escore para **{r.y_final:.3f}**, "
                f"com **aumento adicional de {abs(ajuste_adicional_m2):.3f} ponto(s)**."
            )

    elif usar_m1 and not usar_m2:
        if abs(reducao_mediacao) < 0.001:
            texto.append(
                "O caminho X→M1→M2 foi mantido como referência estrutural, mas sem efeito final sobre Y nesta configuração."
            )
        else:
            texto.append(
                "O caminho X→M1→M2 foi mantido como referência estrutural, "
                "mas o efeito de M2 sobre Y foi desligado; por isso, a mediação não se completa no desfecho."
            )

    elif not usar_m1 and usar_m2:
        if abs(ajuste_adicional_m2) < 0.001:
            texto.append(
                f"O efeito de M2 sobre Y foi mantido, mas o elo X→M1→M2 foi desligado. "
                f"Nesta configuração, o escore final permaneceu em **{r.y_final:.3f}**, sem ajuste adicional relevante."
            )
        elif ajuste_adicional_m2 > 0:
            texto.append(
                f"O efeito de M2 sobre Y foi mantido, mas o elo X→M1→M2 foi desligado. "
                f"Ainda assim, M2 produziu **redução adicional de {ajuste_adicional_m2:.3f} ponto(s)**, "
                f"levando o escore final a **{r.y_final:.3f}**."
            )
        else:
            texto.append(
                f"O efeito de M2 sobre Y foi mantido, mas o elo X→M1→M2 foi desligado. "
                f"Nesta configuração, M2 produziu **aumento adicional de {abs(ajuste_adicional_m2):.3f} ponto(s)**, "
                f"levando o escore final a **{r.y_final:.3f}**."
            )

    else:
        texto.append(
            f"Os caminhos mediadores foram desligados, de modo que o escore final permaneceu em **{r.y_final:.3f}**, "
            f"coincidindo com a pressão basal associada a X."
        )

    texto.append(
        f"A classificação heurística do escore final do risco é **{classificar_risco(r.y_final)}**."
    )

    st.info(" ".join(texto))

    with st.expander("ℹ️ Nota metodológica", expanded=False):
        st.markdown(
            f"""
**Estrutura assumida**
- X = Violência percebida
- M1 = Acolhimento escolar
- M2 = Satisfação com a vida
- Y = Ideação suicida

**Coeficientes**
- X → Y = **{BETA_XY:.3f}**
- X → M1 = **{BETA_XM1:.3f}**
- M1 → M2 = **{BETA_M1M2:.3f}**
- M2 → Y = **{BETA_M2Y:.3f}**

**Moderação por pertença**
- baixa: ×{MOD_BAIXA:.2f}
- alta: ×{MOD_ALTA:.2f}

**Importante**
- Este simulador oferece uma **tradução estrutural e didática** do modelo.
- Quando o modo CSV calibra interceptos com médias do subgrupo, isso deve ser lido como **calibração local**, não como validação preditiva externa.
- X permanece sempre ativo; o que varia é a ativação dos caminhos mediadores.
- Escore final limitado ao intervalo Likert de **1 a 5**.
"""
        )


# =========================================================
# INTERFACE PRINCIPAL
# =========================================================

def main():
    st.title("📊 Simulador do Risco de Ideação Suicida")
    st.caption("Modelo serial moderado: X → M1 → M2 → Y, com X sempre presente.")
    st.caption("Desenvolvido no  OPPES / PPGPSI/ UFS")

    if "mem_x" not in st.session_state:
        st.session_state.mem_x = int(round(V_POP))
        st.session_state.mem_m1 = int(round(M1_POP))
        st.session_state.mem_m2 = int(round(M2_POP))
        st.session_state.mem_pert = "Alta"
        st.session_state.interceptos_cache = interceptos_populacionais(True)
        st.session_state.ultimo_modo = "Modo simulação"

    # Sidebar
    st.sidebar.header("⚙️ Configurações")
    modo = st.sidebar.radio(
        "Origem dos dados",
        ["Dados reais", "Modo simulação"]
    )
    modo_csv = (modo == "Dados reais")

    if modo == "Modo simulação" and st.session_state.ultimo_modo == "Dados reais":
        st.session_state.slider_x = st.session_state.mem_x
        st.session_state.slider_m1 = st.session_state.mem_m1
        st.session_state.slider_m2 = st.session_state.mem_m2
        st.session_state.radio_pert = st.session_state.mem_pert

    st.session_state.ultimo_modo = modo

    usar_m1 = st.sidebar.checkbox("Ativar caminho X→M1→M2", value=True)
    usar_m2 = st.sidebar.checkbox("Ativar efeito M2→Y", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Interpretação do modelo")
    st.sidebar.caption("X entra sempre. M1 e M2 podem ser ligados/desligados como caminhos de proteção/ajuste.")

    inter = interceptos_populacionais(True)
    y_real = None

    if modo_csv:
        try:
            df = carregar_dados("dados_final.csv")
        except FileNotFoundError:
            st.error("Arquivo 'dados_final.csv' não encontrado no diretório do app.")
            st.stop()
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Erro ao carregar CSV: {e}")
            st.stop()

        st.sidebar.markdown("---")
        st.sidebar.subheader("🗂️ Filtros demográficos")

        df_f = df.copy()
        filtros = {}

        for col, rotulo in FILTROS_CONFIG.items():
            if col in df.columns:
                opcoes = ["Todos"] + sorted(df[col].dropna().astype(str).unique().tolist())
                escolha = st.sidebar.selectbox(rotulo, opcoes, key=f"filtro_{col}")
                filtros[col] = escolha

        for col, escolha in filtros.items():
            if escolha != "Todos":
                df_f = df_f[df_f[col].astype(str) == escolha]

        st.sidebar.info(f"👥 Casos no perfil: {len(df_f)}")

        if len(df_f) == 0:
            st.warning("Nenhum caso corresponde ao filtro selecionado.")
            st.stop()

        inter = calibrar_interceptos_subgrupo(df_f)
        y_real = inter.y_real

        x = inter.v_m
        m1 = inter.m1_m
        m2 = inter.m2_m
        pertenca_alta = inter.pertenca_alta

        st.session_state.mem_x = int(round(x))
        st.session_state.mem_m1 = int(round(m1))
        st.session_state.mem_m2 = int(round(m2))
        st.session_state.mem_pert = "Alta" if pertenca_alta else "Baixa"
        st.session_state.interceptos_cache = inter

    else:
        st.sidebar.success(
            "💡 Use os controles para inspecionar cenários com X sempre ativo e com os mediadores ligados ou desligados."
        )

        x = st.session_state.get("slider_x", st.session_state.mem_x)
        m1 = st.session_state.get("slider_m1", st.session_state.mem_m1)
        m2 = st.session_state.get("slider_m2", st.session_state.mem_m2)
        pertenca_escolha = st.session_state.get("radio_pert", st.session_state.mem_pert)
        pertenca_alta = (pertenca_escolha == "Alta")

        inter = interceptos_populacionais(pertenca_alta)

    col_esq, col_dir = st.columns([1, 2.2], gap="large")

    with col_esq:
        if modo_csv:
            st.subheader("Perfil selecionado (médias)")
            st.metric("Violência percebida (X)", f"{x:.2f}")
            st.metric("Acolhimento (M1)", f"{m1:.2f}")
            st.metric("Satisfação com a vida (M2)", f"{m2:.2f}")
            st.write(f"**Pertença:** {'Alta' if inter.pertenca_alta else 'Baixa'}")

            m1_prev = inter.ic_m1 + BETA_XM1 * x
            st.caption(f"M1 previsto pelo caminho X→M1: **{np.clip(m1_prev, 1, 5):.2f}**")
            st.caption(
                f"M1 observado no subgrupo: **{m1:.2f}** "
                f"({'acima' if m1 > m1_prev else 'abaixo' if m1 < m1_prev else 'igual ao'} previsto estruturalmente)"
            )
        else:
            st.subheader("Simulação em escala Likert")
            if "slider_x" not in st.session_state:
                st.session_state.slider_x = int(x)

            if "slider_m1" not in st.session_state:
                st.session_state.slider_m1 = int(m1)

            if "slider_m2" not in st.session_state:
                st.session_state.slider_m2 = int(m2)

            if "radio_pert" not in st.session_state:
                st.session_state.radio_pert = "Alta" if inter.pertenca_alta else "Baixa"

            st.slider("Violência percebida (X)", 1, 5, key="slider_x")
            st.slider("Acolhimento (M1)", 1, 5, key="slider_m1")
            st.slider("Satisfação com a vida (M2)", 1, 5, key="slider_m2")
            st.radio("Pertença grupal", ["Baixa", "Alta"], key="radio_pert")

            x = st.session_state.slider_x
            m1 = st.session_state.slider_m1
            m2 = st.session_state.slider_m2
            pertenca = st.session_state.radio_pert

            inter = interceptos_populacionais(pertenca == "Alta")

        st.markdown("---")
        st.markdown("### Coeficientes estruturais")
        b1, b2 = st.columns(2)
        with b1:
            st.metric("X → M1", f"{BETA_XM1:.3f}")
            st.metric("M1 → M2", f"{BETA_M1M2:.3f} × mod")
            st.metric("X → Y", f"{BETA_XY:.3f}")
        with b2:
            st.metric("M2 → Y", f"{BETA_M2Y:.3f}")
            st.metric("Fator de moderação", f"×{MOD_ALTA if inter.pertenca_alta else MOD_BAIXA:.2f}")
            st.metric(
                "Indireto de referência",
                f"{IND_ALTA if inter.pertenca_alta else IND_BAIXA:.3f}"
            )

        st.markdown("---")
        st.markdown("### Estado dos caminhos")
        st.write(f"- X direto em Y: **ativo**")
        st.write(f"- X→M1→M2: **{'ativo' if usar_m1 else 'desligado'}**")
        st.write(f"- M2→Y: **{'ativo' if usar_m2 else 'desligado'}**")

    resultado = calcular_resultado(
        x=x,
        m1=m1,
        m2=m2,
        interceptos=inter,
        usar_m1=usar_m1,
        usar_m2=usar_m2
    )

    with col_dir:
        st.plotly_chart(
            renderizar_cascata(resultado, usar_m1, usar_m2),
            use_container_width=True
        )

        g1, g2, g3, g4 = st.columns(4)

        reducao_mediacao = resultado.y_basal - resultado.y_pos_mediacao
        ajuste_adicional_m2 = resultado.y_pos_mediacao - resultado.y_final

        g1.metric("Pressão basal por X", f"{resultado.y_basal:.3f}")

        if abs(reducao_mediacao) < 0.001:
            rotulo_mediacao = "Efeito da mediação"
            valor_mediacao = "0.000"
        elif reducao_mediacao > 0:
            rotulo_mediacao = "Redução pela mediação"
            valor_mediacao = f"{reducao_mediacao:.3f}"
        else:
            rotulo_mediacao = "Aumento pela mediação"
            valor_mediacao = f"{abs(reducao_mediacao):.3f}"

        g2.metric(rotulo_mediacao, valor_mediacao)

        if abs(ajuste_adicional_m2) < 0.001:
            rotulo_m2 = "Ajuste adicional por M2"
            valor_m2 = "0.000"
        elif ajuste_adicional_m2 > 0:
            rotulo_m2 = "Redução adicional por M2"
            valor_m2 = f"{ajuste_adicional_m2:.3f}"
        else:
            rotulo_m2 = "Aumento adicional por M2"
            valor_m2 = f"{abs(ajuste_adicional_m2):.3f}"

        g3.metric(rotulo_m2, valor_m2)

        g4.metric(
            "Escore final esperado",
            f"{resultado.y_final:.3f}" + (" ⚠️" if resultado.saturado else "")
        )

        st.markdown("---")
        st.markdown("### Leitura substantiva")
        s1, s2, s3 = st.columns(3)
        s1.metric("M2 previsto a partir de M1", f"{resultado.m2_previsto_por_m1:.3f}")
        s2.metric("Classificação heurística", classificar_risco(resultado.y_final))
        s3.metric("β M1→M2 efetivo", f"{resultado.beta_m1m2_efetivo:.3f}")

        if modo_csv and y_real is not None:
            erro = abs(resultado.y_final - y_real)
            if erro < 0.01:
                st.success(
                    f"📌 **Calibração do subgrupo:** média observada = **{y_real:.3f}** | "
                    f"média reproduzida = **{resultado.y_final:.3f}** | erro = **{erro:.4f}**"
                )
            else:
                st.info(
                    f"📌 **Calibração do subgrupo:** média observada = **{y_real:.3f}** | "
                    f"média reproduzida = **{resultado.y_final:.3f}** | erro = **{erro:.4f}**"
                )

        with st.expander("🔬 Diagrama estrutural do modelo", expanded=False):
            st.plotly_chart(renderizar_diagrama_estrutural(), use_container_width=True)

        exibir_painel_tecnico(inter, resultado, usar_m1, usar_m2)


if __name__ == "__main__":
    main()