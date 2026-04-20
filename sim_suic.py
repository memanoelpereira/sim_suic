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
# Estrutura:
# X  = Violência percebida
# M1 = Acolhimento escolar
# M2 = Satisfação com a vida
# Y  = Ideação suicida

BETA_XY = 0.243
BETA_XM1 = -0.324
BETA_M1M2 = 0.527
BETA_M2Y = -0.170

# HDIs 95% aproximados dos coeficientes
HDI_XY = (0.167, 0.324)
HDI_XM1 = (-0.369, -0.279)
HDI_M1M2 = (0.462, 0.591)
HDI_M2Y = (-0.247, -0.089)

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

# Colunas mínimas obrigatórias
COLUNAS_MINIMAS = [
    "percepcao_violencia",
    "acolhimento",
    "satisfacao_vida",
    "pertenca_grupal",
    "ideacao_suicida",
]

# Filtros opcionais
FILTROS_CONFIG = {
    "sexo": "Sexo",
    "cor_da_pele": "Cor da pele",
    "orientação_sexual": "Orientação sexual",
    "renda": "Renda",
    "cidade": "Cidade",
    "serie": "Série",
}


# =========================================================
# ESTRUTURAS
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
    pertenca_media_ref: Optional[float] = None
    pertenca_dp_ref: Optional[float] = None
    pertenca_z: Optional[float] = None


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
    cadeia_ativa: bool


# =========================================================
# FUNÇÕES BÁSICAS
# =========================================================

def beta_m1m2_efetivo(pertenca_alta: bool) -> float:
    return BETA_M1M2 * (MOD_ALTA if pertenca_alta else MOD_BAIXA)


def inferir_pertenca_alta(valor_pertenca: float, media_ref: float, dp_ref: float) -> tuple[bool, float]:
    """
    Classifica a pertença do subgrupo em relação à distribuição de referência.
    Se z >= 0, trata como alta; se z < 0, trata como baixa.
    """
    if dp_ref <= 0 or np.isnan(dp_ref):
        z = 0.0 if valor_pertenca == media_ref else (1.0 if valor_pertenca > media_ref else -1.0)
        return valor_pertenca >= media_ref, z

    z = (valor_pertenca - media_ref) / dp_ref
    return z >= 0, z


def classificar_risco(y: float) -> str:
    if y < 2.0:
        return "Faixa baixa"
    elif y < 3.0:
        return "Faixa moderada"
    elif y < 4.0:
        return "Faixa elevada"
    return "Faixa muito elevada"


def _sd_aprox_por_hdi(hdi_inf: float, hdi_sup: float) -> float:
    return (hdi_sup - hdi_inf) / (2 * 1.96)


# =========================================================
# INTERCEPTOS
# =========================================================

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


def calibrar_interceptos_subgrupo(df_sub: pd.DataFrame, df_ref: pd.DataFrame) -> Interceptos:
    v_m = float(df_sub["percepcao_violencia"].mean())
    m1_m = float(df_sub["acolhimento"].mean())
    m2_m = float(df_sub["satisfacao_vida"].mean())
    p_m = float(df_sub["pertenca_grupal"].mean())
    y_m = float(df_sub["ideacao_suicida"].mean())

    p_ref_media = float(df_ref["pertenca_grupal"].mean())
    p_ref_dp = float(df_ref["pertenca_grupal"].std())

    pertenca_alta, p_z = inferir_pertenca_alta(p_m, p_ref_media, p_ref_dp)
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
        n=len(df_sub),
        pertenca_media_ref=p_ref_media,
        pertenca_dp_ref=p_ref_dp,
        pertenca_z=p_z
    )


# =========================================================
# DADOS
# =========================================================

@st.cache_data
def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv)

    faltantes = [c for c in COLUNAS_MINIMAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes no CSV: {', '.join(faltantes)}")

    df = df.dropna(subset=COLUNAS_MINIMAS).copy()
    return df


# =========================================================
# MOTOR DO MODELO
# =========================================================

def calcular_resultado(
    x: float,
    m1: float,
    m2: float,
    interceptos: Interceptos,
    usar_m1: bool,
    usar_m2: bool
) -> ResultadoSimulacao:
    pertenca_alta = interceptos.pertenca_alta
    beta_eff = beta_m1m2_efetivo(pertenca_alta)

    # Nó basal: apenas efeito direto de X
    y_basal = interceptos.ic_y + (BETA_XY * x)

    # Regra rígida:
    # se M1=0 e/ou M2=0, some o impacto indireto.
    # idem se o usuário desligar algum dos caminhos.
    cadeia_ativa = usar_m1 and usar_m2 and (m1 > 0) and (m2 > 0)

    if cadeia_ativa:
        m2_previsto = beta_eff * m1
        y_pos_mediacao = y_basal + (BETA_M2Y * m2_previsto)
        y_final_raw = y_basal + (BETA_M2Y * m2)
        efeito_indireto_x = BETA_M2Y * beta_eff * BETA_XM1 * x
    else:
        m2_previsto = 0.0
        y_pos_mediacao = y_basal
        y_final_raw = y_basal
        efeito_indireto_x = 0.0

    y_final = float(np.clip(y_final_raw, 1.0, 5.0))
    saturado = y_final != y_final_raw

    reducao_mediada = y_basal - y_pos_mediacao
    ajuste_m2 = y_pos_mediacao - y_final_raw

    efeito_direto_x = BETA_XY * x
    efeito_total_x = efeito_direto_x + efeito_indireto_x

    return ResultadoSimulacao(
        y_basal=y_basal,
        y_pos_mediacao=y_pos_mediacao,
        y_final=y_final,
        reducao_mediada=reducao_mediada,
        ajuste_m2=ajuste_m2,
        m2_previsto_por_m1=float(np.clip(m2_previsto, 0.0, 5.0)),
        beta_m1m2_efetivo=beta_eff,
        fator_mod=(MOD_ALTA if pertenca_alta else MOD_BAIXA),
        efeito_direto_x=efeito_direto_x,
        efeito_indireto_x=efeito_indireto_x,
        efeito_total_x=efeito_total_x,
        saturado=saturado,
        cadeia_ativa=cadeia_ativa
    )


def simular_incerteza_risco(
    x: float,
    m1: float,
    m2: float,
    interceptos: Interceptos,
    usar_m1: bool,
    usar_m2: bool,
    n_sim: int = 4000,
    seed: int = 123
) -> dict:
    rng = np.random.default_rng(seed)

    beta_xy_s = rng.normal(BETA_XY, _sd_aprox_por_hdi(*HDI_XY), n_sim)
    beta_xm1_s = rng.normal(BETA_XM1, _sd_aprox_por_hdi(*HDI_XM1), n_sim)
    beta_m1m2_s = rng.normal(BETA_M1M2, _sd_aprox_por_hdi(*HDI_M1M2), n_sim)
    beta_m2y_s = rng.normal(BETA_M2Y, _sd_aprox_por_hdi(*HDI_M2Y), n_sim)

    fator_mod = MOD_ALTA if interceptos.pertenca_alta else MOD_BAIXA
    beta_m1m2_eff_s = beta_m1m2_s * fator_mod

    ic_y = interceptos.ic_y

    # Basal: apenas efeito direto
    y_basal_s = ic_y + beta_xy_s * x

    cadeia_ativa = usar_m1 and usar_m2 and (m1 > 0) and (m2 > 0)

    if cadeia_ativa:
        m2_prev_s = beta_m1m2_eff_s * m1
        y_pos_s = ic_y + beta_xy_s * x + beta_m2y_s * m2_prev_s
        y_final_raw_s = ic_y + beta_xy_s * x + beta_m2y_s * m2
    else:
        y_pos_s = y_basal_s.copy()
        y_final_raw_s = y_basal_s.copy()

    y_final_s = np.clip(y_final_raw_s, 1.0, 5.0)

    return {
        "y_basal_med": float(np.median(y_basal_s)),
        "y_basal_hdi": (
            float(np.quantile(y_basal_s, 0.025)),
            float(np.quantile(y_basal_s, 0.975)),
        ),
        "y_pos_med": float(np.median(y_pos_s)),
        "y_pos_hdi": (
            float(np.quantile(y_pos_s, 0.025)),
            float(np.quantile(y_pos_s, 0.975)),
        ),
        "y_final_med": float(np.median(y_final_s)),
        "y_final_hdi": (
            float(np.quantile(y_final_s, 0.025)),
            float(np.quantile(y_final_s, 0.975)),
        ),
        "cadeia_ativa": cadeia_ativa
    }


# =========================================================
# VISUALIZAÇÕES
# =========================================================

def renderizar_cascata(resultado: ResultadoSimulacao, usar_m1: bool, usar_m2: bool, incerteza: Optional[dict] = None) -> go.Figure:
    fig = go.Figure()

    x_labels = [
        "1. Pressão Basal por X",
        "2. Após Mediação",
        "3. Escore Final"
    ]

    y1 = resultado.y_basal
    y2 = resultado.y_pos_mediacao
    y3 = resultado.y_final

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

    error_map = {}
    if incerteza is not None:
        error_map = {
            x_labels[0]: incerteza["y_basal_hdi"],
            x_labels[1]: incerteza["y_pos_hdi"],
            x_labels[2]: incerteza["y_final_hdi"],
        }

    for xlab, yval, cor, borda, nome in pontos:
        error_y = None
        if xlab in error_map:
            inf, sup = error_map[xlab]
            error_y = dict(
                type="data",
                symmetric=False,
                array=[max(sup - yval, 0)],
                arrayminus=[max(yval - inf, 0)],
                visible=True,
                thickness=1.4,
                width=4
            )

        fig.add_trace(go.Scatter(
            x=[xlab],
            y=[yval],
            mode="markers+text",
            marker=dict(size=22, color=cor, line=dict(color=borda, width=2)),
            text=[f"<b>{yval:.3f}</b>"],
            textposition="top center",
            textfont=dict(size=14, color=cor),
            name=nome,
            error_y=error_y
        ))

    subtitulo = [
        "X sempre ativo",
        f"M1 {'ativo' if usar_m1 else 'desligado'}",
        f"M2 {'ativo' if usar_m2 else 'desligado'}",
        f"Cadeia {'ativa' if resultado.cadeia_ativa else 'anulada'}"
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


# =========================================================
# PAINEL TÉCNICO
# =========================================================

def exibir_painel_tecnico(inter: Interceptos, r: ResultadoSimulacao, usar_m1: bool, usar_m2: bool, incerteza: Optional[dict] = None):
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

    if r.cadeia_ativa:
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
    else:
        motivos = []
        if not usar_m1:
            motivos.append("o caminho X→M1→M2 foi desligado")
        if not usar_m2:
            motivos.append("o caminho M2→Y foi desligado")
        if m1_zero := (r.m2_previsto_por_m1 == 0.0):
            motivos.append("acolhimento e/ou satisfação foram zerados")

        texto.append(
            "A cadeia indireta foi anulada, de modo que **restou apenas o efeito direto de X sobre Y**."
        )
        texto.append(
            f"Nesta configuração, o escore pós-mediação (**{r.y_pos_mediacao:.3f}**) e o escore final (**{r.y_final:.3f}**) "
            f"coincidem com o nível basal."
        )

    if inter.pertenca_z is not None:
        direcao = "acima" if inter.pertenca_z >= 0 else "abaixo"
        texto.append(
            f"A pertença média do subgrupo ficou **{direcao} da média da distribuição de referência**, "
            f"com **z = {inter.pertenca_z:.3f}**, e por isso o cenário moderador foi tratado como "
            f"**{'alta' if inter.pertenca_alta else 'baixa'} pertença**."
        )

    if incerteza is not None:
        yf_inf, yf_sup = incerteza["y_final_hdi"]
        texto.append(
            f"Considerando a incerteza aproximada dos coeficientes, a **faixa plausível do escore final** ficou em "
            f"**[{yf_inf:.3f}, {yf_sup:.3f}]** no HDI 95% aproximado."
        )

    texto.append(
        f"A classificação heurística posiciona o escore final na **{classificar_risco(r.y_final).lower()}** da escala simulada."
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

**Coeficientes centrais**
- X → Y = **{BETA_XY:.3f}**
- X → M1 = **{BETA_XM1:.3f}**
- M1 → M2 = **{BETA_M1M2:.3f}**
- M2 → Y = **{BETA_M2Y:.3f}**

**HDIs 95% aproximados dos coeficientes**
- X → Y: **[{HDI_XY[0]:.3f}, {HDI_XY[1]:.3f}]**
- X → M1: **[{HDI_XM1[0]:.3f}, {HDI_XM1[1]:.3f}]**
- M1 → M2: **[{HDI_M1M2[0]:.3f}, {HDI_M1M2[1]:.3f}]**
- M2 → Y: **[{HDI_M2Y[0]:.3f}, {HDI_M2Y[1]:.3f}]**

**Moderação por pertença**
- baixa: ×{MOD_BAIXA:.2f}
- alta: ×{MOD_ALTA:.2f}

**Critério para classificar pertença no modo CSV**
- A média de pertença do subgrupo é comparada à distribuição de referência do banco.
- Calcula-se **z = (média_subgrupo - média_referência) / dp_referência**.
- Se **z >= 0**, o subgrupo é tratado como **alta pertença**.
- Se **z < 0**, o subgrupo é tratado como **baixa pertença**.

**Regra do modo simulação**
- X permanece sempre ativo.
- Se **acolhimento = 0** e/ou **satisfação = 0**, o impacto indireto é anulado.
- Nessa situação, o gráfico e as análises mostram apenas o **efeito direto**.

**Importante**
- Este simulador oferece uma **tradução estrutural e didática** do modelo.
- Quando o modo CSV calibra interceptos com médias do subgrupo, isso deve ser lido como **calibração local**, não como validação preditiva externa.
- Escore final limitado ao intervalo Likert de **1 a 5**.
- A faixa plausível do risco é uma **aproximação por simulação**, baseada nos HDIs dos coeficientes e não nos draws posteriores originais completos.
"""
        )


# =========================================================
# APP
# =========================================================

def main():
    st.title("📊 Simulador Estrutural de Risco de Ideação")
    st.caption("Modelo serial moderado: X → M1 → M2 → Y, com X sempre presente.")

    if "mem_x" not in st.session_state:
        st.session_state.mem_x = int(round(V_POP))
        st.session_state.mem_m1 = max(0, int(round(M1_POP)))
        st.session_state.mem_m2 = max(0, int(round(M2_POP)))
        st.session_state.mem_pert = "Alta"
        st.session_state.interceptos_cache = interceptos_populacionais(True)
        st.session_state.ultimo_modo = "Modo simulação"

    st.sidebar.header("⚙️ Configurações")
    modo = st.sidebar.radio("Origem dos dados", ["Dados reais", "Modo simulação"])
    modo_exibicao = st.sidebar.toggle(
        "Modo simples",
        value=False,
        help="Caso ativado, mostra detalhes técnicos e metodológicos. Destivado: mostra interface mais enxuta."
    )
    modo_csv = modo == "Dados reais"

    if modo == "Modo simulação" and st.session_state.ultimo_modo == "Dados reais":
        st.session_state.slider_x = min(5, max(1, st.session_state.mem_x))
        st.session_state.slider_m1 = min(5, max(0, st.session_state.mem_m1))
        st.session_state.slider_m2 = min(5, max(0, st.session_state.mem_m2))
        st.session_state.radio_pert = st.session_state.mem_pert

    st.session_state.ultimo_modo = modo

    usar_m1 = st.sidebar.checkbox("Ativar caminho X→M1→M2", value=True)
    usar_m2 = st.sidebar.checkbox("Ativar efeito M2→Y", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Interpretação do modelo")
    if modo_exibicao:
        st.sidebar.caption("X entra sempre. Se M1=0 e/ou M2=0, o impacto indireto desaparece e resta apenas o efeito direto.")
    else:
        st.sidebar.caption("Modo gestor: exibe apenas informações essenciais para leitura rápida do cenário.")

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

        inter = calibrar_interceptos_subgrupo(df_f, df)
        y_real = inter.y_real

        x = inter.v_m
        m1 = inter.m1_m
        m2 = inter.m2_m

        st.session_state.mem_x = int(round(x))
        st.session_state.mem_m1 = max(0, int(round(m1)))
        st.session_state.mem_m2 = max(0, int(round(m2)))
        st.session_state.mem_pert = "Alta" if inter.pertenca_alta else "Baixa"
        st.session_state.interceptos_cache = inter

    else:
        st.sidebar.success("💡 Em M1 e M2, o valor 0 anula o impacto indireto correspondente e o gráfico colapsa para o efeito direto.")

        if "slider_x" not in st.session_state:
            st.session_state.slider_x = min(5, max(1, st.session_state.mem_x))
        if "slider_m1" not in st.session_state:
            st.session_state.slider_m1 = min(5, max(0, st.session_state.mem_m1))
        if "slider_m2" not in st.session_state:
            st.session_state.slider_m2 = min(5, max(0, st.session_state.mem_m2))
        if "radio_pert" not in st.session_state:
            st.session_state.radio_pert = st.session_state.mem_pert

    col_esq, col_dir = st.columns([1, 2.2], gap="large")

    with col_esq:
        if modo_csv:
            st.subheader("Perfil selecionado (médias)")
            st.metric("Violência percebida (X)", f"{x:.2f}")
            st.metric("Acolhimento (M1)", f"{m1:.2f}")
            st.metric("Satisfação com a vida (M2)", f"{m2:.2f}")
            st.write(f"**Pertença:** {'Alta' if inter.pertenca_alta else 'Baixa'}")

            if inter.pertenca_media_ref is not None and inter.pertenca_dp_ref is not None and inter.pertenca_z is not None:
                st.caption(
                    f"Pertença do subgrupo: média = **{float(df_f['pertenca_grupal'].mean()):.3f}** | "
                    f"referência = **{inter.pertenca_media_ref:.3f}** | "
                    f"DP ref. = **{inter.pertenca_dp_ref:.3f}** | "
                    f"z = **{inter.pertenca_z:.3f}**"
                )

            m1_prev = inter.ic_m1 + BETA_XM1 * x
            st.caption(f"M1 previsto pelo caminho X→M1: **{np.clip(m1_prev, 1, 5):.2f}**")
            st.caption(
                f"M1 observado no subgrupo: **{m1:.2f}** "
                f"({'acima' if m1 > m1_prev else 'abaixo' if m1 < m1_prev else 'igual ao'} previsto estruturalmente)"
            )
        else:
            st.subheader("Simulação em escala ajustada")
            st.slider("Violência percebida (X)", 1, 5, key="slider_x")
            st.slider("Acolhimento (M1)", 0, 5, key="slider_m1")
            st.slider("Satisfação com a vida (M2)", 0, 5, key="slider_m2")
            st.radio("Pertença grupal", ["Baixa", "Alta"], key="radio_pert")
            st.caption("Em M1 e M2, o valor 0 anula o impacto indireto. Com M1=0 e/ou M2=0, restam apenas os efeitos diretos.")

            x = st.session_state.slider_x
            m1 = st.session_state.slider_m1
            m2 = st.session_state.slider_m2
            pertenca = st.session_state.radio_pert

            inter = interceptos_populacionais(pertenca == "Alta")

        if modo_exibicao:
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
                st.metric("Indireto de referência", f"{IND_ALTA if inter.pertenca_alta else IND_BAIXA:.3f}")

            st.markdown("---")
            st.markdown("### Estado dos caminhos")
            st.write("- X direto em Y: **ativo**")
            st.write(f"- X→M1→M2: **{'ativo' if usar_m1 else 'desligado'}**")
            st.write(f"- M2→Y: **{'ativo' if usar_m2 else 'desligado'}**")
            st.write(f"- Cadeia indireta: **{'ativa' if (usar_m1 and usar_m2 and m1 > 0 and m2 > 0) else 'anulada'}**")
        else:
            st.markdown("---")
            st.markdown("### Leitura operacional")
            st.write(f"- Violência (X): **{x:.2f}**")
            st.write(f"- Acolhimento (M1): **{m1:.2f}**")
            st.write(f"- Satisfação (M2): **{m2:.2f}**")
            st.write(f"- Cadeia indireta: **{'ativa' if (usar_m1 and usar_m2 and m1 > 0 and m2 > 0) else 'anulada'}**")

    resultado = calcular_resultado(
        x=x,
        m1=m1,
        m2=m2,
        interceptos=inter,
        usar_m1=usar_m1,
        usar_m2=usar_m2
    )

    incerteza = simular_incerteza_risco(
        x=x,
        m1=m1,
        m2=m2,
        interceptos=inter,
        usar_m1=usar_m1,
        usar_m2=usar_m2,
        n_sim=4000,
        seed=123
    )

    with col_dir:
        st.plotly_chart(
            renderizar_cascata(resultado, usar_m1, usar_m2, incerteza=incerteza),
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

        if modo_exibicao:
            st.markdown("---")
            st.markdown("### Faixa plausível do risco estimado")

            h1, h2, h3 = st.columns(3)

            yb_inf, yb_sup = incerteza["y_basal_hdi"]
            yp_inf, yp_sup = incerteza["y_pos_hdi"]
            yf_inf, yf_sup = incerteza["y_final_hdi"]

            with h1:
                st.metric("Basal: mediana simulada", f"{incerteza['y_basal_med']:.3f}")
                st.caption(f"HDI 95% aprox.: [{yb_inf:.3f}, {yb_sup:.3f}]")

            with h2:
                st.metric("Pós-mediação: mediana simulada", f"{incerteza['y_pos_med']:.3f}")
                st.caption(f"HDI 95% aprox.: [{yp_inf:.3f}, {yp_sup:.3f}]")

            with h3:
                st.metric("Final: mediana simulada", f"{incerteza['y_final_med']:.3f}")
                st.caption(f"HDI 95% aprox.: [{yf_inf:.3f}, {yf_sup:.3f}]")

            st.info(
                f"Faixa plausível do escore final (HDI 95% aproximado): **[{yf_inf:.3f}, {yf_sup:.3f}]** "
                f"| mediana simulada: **{incerteza['y_final_med']:.3f}**"
            )

            st.markdown("---")
            st.markdown("### Leitura substantiva")

            s1, s2, s3 = st.columns(3)
            s1.metric("M2 previsto a partir de M1", f"{resultado.m2_previsto_por_m1:.3f}")
            s2.metric("Faixa da escala simulada", classificar_risco(resultado.y_final))
            s3.metric("β M1→M2 efetivo", f"{resultado.beta_m1m2_efetivo:.3f}")
        else:
            st.markdown("---")
            st.markdown("### Resumo executivo")
            s1, s2 = st.columns(2)
            s1.metric("Faixa da escala simulada", classificar_risco(resultado.y_final))
            s2.metric("Cadeia indireta", "Ativa" if resultado.cadeia_ativa else "Anulada")

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

        if modo_exibicao:
            with st.expander("🔬 Diagrama estrutural do modelo", expanded=False):
                st.plotly_chart(renderizar_diagrama_estrutural(), use_container_width=True)

            exibir_painel_tecnico(inter, resultado, usar_m1, usar_m2, incerteza=incerteza)
        else:
            with st.expander("Resumo gerencial", expanded=True):
                if resultado.cadeia_ativa:
                    st.write(
                        "O cenário atual indica que acolhimento e satisfação estão contribuindo para modificar o risco final. "
                        "O valor exibido já incorpora esses efeitos indiretos."
                    )
                else:
                    st.write(
                        "Neste cenário, os efeitos indiretos estão anulados. O risco final reflete apenas o efeito direto da violência."
                    )
                st.write(f"**Escore final esperado:** {resultado.y_final:.3f}")
                st.write(f"**Classificação na escala simulada:** {classificar_risco(resultado.y_final)}")


if __name__ == "__main__":
    main()
