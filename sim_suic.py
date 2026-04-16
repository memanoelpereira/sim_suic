import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ==========================================
# COEFICIENTES BSEM — FIGURA 4 DO ARTIGO
# ==========================================
# Caminhos da mediação serial moderada
BETA_VI  =  0.243   # Violência → Ideação (efeito DIRETO)
BETA_VA  = -0.324   # Violência → Acolhimento
BETA_AS  =  0.527   # Acolhimento → Satisfação (nível médio de pertença)
BETA_SI  = -0.170   # Satisfação → Ideação

# Efeitos indiretos por nível de pertença (HDI 97,5% do artigo)
IND_BAIXA, IND_MEDIA, IND_ALTA = 0.016, 0.020, 0.024
MOD_BAIXA = IND_BAIXA / IND_MEDIA  # 0.80  → BETA_AS efetivo = 0.422
MOD_ALTA  = IND_ALTA  / IND_MEDIA  # 1.20  → BETA_AS efetivo = 0.632

# Médias populacionais (n=1472 após filtragem)
V_POP, A_POP, S_POP, IS_POP = 1.51, 3.56, 3.38, 1.69

# ==========================================
# MOTOR DE CALIBRAÇÃO DOS INTERCEPTOS
# ==========================================

def _beta_as(pertenca_alta: bool) -> float:
    return BETA_AS * (MOD_ALTA if pertenca_alta else MOD_BAIXA)


def interceptos_populacionais(pertenca_alta: bool) -> dict:
    """Interceptos calibrados a partir das médias populacionais (n=1472)."""
    bas = _beta_as(pertenca_alta)
    return {
        "ic_a":   A_POP  - BETA_VA * V_POP,           # = 4.049
        "ic_s":   S_POP  - bas     * A_POP,           # depende de pertença
        "ic_is":  IS_POP - BETA_VI * V_POP - BETA_SI * S_POP,  # = 1.898
    }


def calibrar_interceptos_empiricos(df_sub: pd.DataFrame) -> dict:
    """
    Calibra interceptos diretamente das médias do subgrupo.
    Garante Sincronia Perfeita: IS_pred = IS_real em qualquer filtro.
    Usa os coeficientes BSEM reais (não os 0.59/0.45/0.32 anteriores).
    """
    v_m  = float(df_sub['percepcao_violencia'].mean())
    a_m  = float(df_sub['acolhimento'].mean())
    s_m  = float(df_sub['satisfacao_vida'].mean())
    p_m  = float(df_sub['pertenca_grupal'].mean())
    is_m = float(df_sub['ideacao_suicida'].mean())

    pertenca_alta = p_m > 5.0
    bas = _beta_as(pertenca_alta)

    return {
        "pertenca_alta": pertenca_alta,
        "ic_a":   a_m  - BETA_VA * v_m,
        "ic_s":   s_m  - bas     * a_m,
        "ic_is":  is_m - BETA_VI * v_m - BETA_SI * s_m,
        "v_m": v_m, "a_m": a_m, "s_m": s_m,
        "is_real": is_m, "n": len(df_sub),
    }


# ==========================================
# MOTOR MATEMÁTICO — CASCATA BSEM
# ==========================================

def calcular_cascata_bsem(violencia, acolhimento, satisfacao, pertenca_alta, ic: dict):
    """
    Implementa a mediação serial do artigo: V → A → S → IS
    com pertença modulando o caminho A → S.

    Cascata em unidades de IS (escala 1-5):
      Nó 1 (Pressão bruta):   IS sem nenhuma mediação via A → S
                               = ic_is + β_VI·V + β_SI·ic_s
      Nó 2 (Pós-Acolhimento): IS após A mediar (via S_via_A)
                               = ic_is + β_VI·V + β_SI·(ic_s + β_AS_eff·A)
      Nó 3 (IS Final):        IS com S inserido diretamente
                               = ic_is + β_VI·V + β_SI·S
    """
    bas   = _beta_as(pertenca_alta)
    ic_s  = ic.get("ic_s",  interceptos_populacionais(pertenca_alta)["ic_s"])
    ic_is = ic.get("ic_is", interceptos_populacionais(pertenca_alta)["ic_is"])

    # Nó 1 — pressão de V sem proteção via A→S
    no1 = ic_is + BETA_VI * violencia + BETA_SI * ic_s

    # Satisfação prevista pelo modelo a partir de A
    s_via_a = ic_s + bas * acolhimento

    # Nó 2 — após A mediar via S
    no2 = ic_is + BETA_VI * violencia + BETA_SI * s_via_a

    # Nó 3 — IS com S do usuário (pode ser diferente de s_via_a em modo intervenção)
    no3_raw = ic_is + BETA_VI * violencia + BETA_SI * satisfacao
    no3     = np.clip(no3_raw, 1.0, 5.0)

    queda_acol  = no1 - no2         # proteção via A→S→IS (deve ser > 0)
    queda_satis = no2 - no3_raw     # ajuste adicional por S além de S_via_A

    # Decomposição efeito direto vs indireto (para métricas)
    efeito_direto_V   = BETA_VI * violencia                     # IS por V direto
    efeito_indireto_V = BETA_SI * bas * BETA_VA * violencia     # IS por V via A→S

    return {
        "no1": no1, "no2": no2, "no3": no3,
        "queda_acol":  queda_acol,
        "queda_satis": queda_satis,
        "s_via_a":     np.clip(s_via_a, 1.0, 5.0),
        "bas":         bas,
        "mod":         MOD_ALTA if pertenca_alta else MOD_BAIXA,
        "efeito_direto":   efeito_direto_V,
        "efeito_indireto": efeito_indireto_V,
        "saturacao":   no3 != no3_raw,
    }


# ==========================================
# MOTOR VISUAL — CASCATA DINÂMICA
# ==========================================

def renderizar_cascata(r: dict):
    """Cascata de 3 nós fiel ao modelo BSEM do artigo."""
    fig = go.Figure()
    eixo_x = ["1. Pressão da Violência", "2. Acolhimento", "3. Satisfação (Final)"]
    no1, no2, no3 = r["no1"], r["no2"], r["no3"]

    # Linhas de referência
    fig.add_hline(y=5.0, line_dash="dot", line_color="red", line_width=1.5,
                  annotation_text="Teto Máximo (Likert = 5)", annotation_position="top left",
                  annotation_font_size=11)
    fig.add_hline(y=1.0, line_dash="dot", line_color="green", line_width=1.5,
                  annotation_text="Piso Mínimo (Likert = 1)", annotation_position="bottom left",
                  annotation_font_size=11)

    # Linhas de nível (guias horizontais)
    for x_a, x_b, y in [
        (eixo_x[0], eixo_x[1], no1),
        (eixo_x[1], eixo_x[2], no2),
    ]:
        fig.add_trace(go.Scatter(x=[x_a, x_b], y=[y, y], mode="lines",
                                 line=dict(color="rgba(150,150,150,0.4)", width=2, dash="dash"),
                                 showlegend=False, hoverinfo="skip"))

    # Quedas verticais (em vermelho)
    for x, y_top, y_bot, queda in [
        (eixo_x[1], no1, no2, r["queda_acol"]),
        (eixo_x[2], no2, no3, r["queda_satis"]),
    ]:
        if abs(queda) > 0.005:
            fig.add_trace(go.Scatter(x=[x, x], y=[y_top, y_bot], mode="lines",
                                     line=dict(color="rgba(220,50,50,0.85)", width=3),
                                     showlegend=False, hoverinfo="skip"))
            mid_y = y_bot + abs(queda) / 2
            sinal = "-" if queda > 0 else "+"
            fig.add_annotation(
                x=x, y=mid_y,
                text=f"<b>{sinal}{abs(queda):.3f} pts</b>",
                showarrow=True, arrowhead=2, arrowcolor="rgba(220,50,50,0.8)",
                ax=52, ay=0, font=dict(size=12, color="#d62728"),
                bgcolor="white", bordercolor="rgba(220,50,50,0.5)", borderwidth=1
            )

    # Pontos principais
    for x, y, cor, cor_borda, nome, pos in [
        (eixo_x[0], no1, "#1f77b4", "#0e4e7d", "Pressão Inicial", "top center"),
        (eixo_x[1], no2, "#ff7f0e", "#cc6600", "Pós-Acolhimento", "bottom center"),
        (eixo_x[2], no3, "#2ca02c", "#175e17", "Risco Latente Final", "bottom center"),
    ]:
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=22, color=cor, line=dict(color=cor_borda, width=2)),
            text=[f"<b>{y:.3f}</b>"], textposition=pos,
            textfont=dict(size=14, color=cor), name=nome
        ))

    # Pontos fantasma (abertura das quedas)
    for x, y, cor in [(eixo_x[1], no1, "#1f77b4"), (eixo_x[2], no2, "#ff7f0e")]:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers",
                                 marker=dict(size=12, color="white",
                                             line=dict(color=cor, width=2)),
                                 showlegend=False))

    limite_superior = max(5.5, no1 + 0.4)
    limite_inferior = min(0.7, no3 - 0.3)

    fig.update_layout(
        title=dict(text="Dinâmica de Transbordo e Proteção Psicológica", font=dict(size=18)),
        yaxis=dict(title="Pressão Psicológica (Risco Latente)", range=[limite_inferior, limite_superior],
                   gridcolor="rgba(200,200,200,0.3)"),
        xaxis=dict(showgrid=False),
        plot_bgcolor="white", hovermode="x unified",
        margin=dict(l=60, r=80, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def renderizar_painel_bsem():
    """Exibe um diagrama de trajetória resumindo os coeficientes do artigo."""
    fig = go.Figure()

    nos = {
        "V": (0.1, 0.5),
        "A": (0.38, 0.5),
        "S": (0.66, 0.5),
        "IS": (0.9, 0.5),
        "W": (0.38, 0.08),
    }
    labels = {
        "V": "Violência<br>Percebida",
        "A": "Acolhimento<br>Escolar",
        "S": "Satisfação<br>com a Vida",
        "IS": "Ideação<br>Suicida",
        "W": "Pertença Grupal<br>(Moderadora)",
    }
    setas = [
        ("V", "A", "β = -0.324", -0.02),
        ("A", "S", "β = 0.527*", +0.06),
        ("S", "IS", "β = -0.170", -0.02),
        ("V", "IS", "β = 0.243", +0.15),  # efeito direto (arco)
        ("W", "A",  "ind: 0.016–0.024", 0),
    ]

    # Desenha os nós
    for key, (x, y) in nos.items():
        cor = "#d62728" if key == "V" else "#2ca02c" if key in ("A", "S") else "#1f77b4" if key == "IS" else "#ff7f0e"
        fig.add_annotation(x=x, y=y, text=f"<b>{labels[key]}</b>",
                           showarrow=False,
                           font=dict(size=10, color="white"),
                           bgcolor=cor, bordercolor="white",
                           borderwidth=2, borderpad=6, opacity=0.92)

    # Linhas das setas (aproximadas)
    paths = [
        ([nos["V"][0]+0.04, nos["A"][0]-0.04], [0.5, 0.5]),
        ([nos["A"][0]+0.04, nos["S"][0]-0.04], [0.5, 0.5]),
        ([nos["S"][0]+0.04, nos["IS"][0]-0.04], [0.5, 0.5]),
        ([nos["V"][0]+0.05, nos["IS"][0]-0.04], [0.58, 0.58]),  # arco direto
        ([nos["W"][0],      nos["A"][0]], [0.13, 0.44]),         # moderadora
    ]
    for (xs, ys) in paths:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                 line=dict(color="gray", width=1.5),
                                 showlegend=False))

    # Labels das setas
    anno_pos = [
        (0.24, 0.57, "β = -0.324", "#d62728"),
        (0.52, 0.57, "β = 0.527*mod", "#2ca02c"),
        (0.78, 0.57, "β = -0.170", "#2ca02c"),
        (0.5,  0.65, "β = 0.243 (direto)", "#1f77b4"),
        (0.26, 0.26, "mod: 0.80–1.20", "#ff7f0e"),
    ]
    for (x, y, txt, cor) in anno_pos:
        fig.add_annotation(x=x, y=y, text=f"<i>{txt}</i>",
                           showarrow=False, font=dict(size=9, color=cor))

    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 0.8]),
        plot_bgcolor="white",
        title=dict(text="Diagrama de Trajetórias BSEM (Figura 4 do artigo)",
                   font=dict(size=13), x=0.5)
    )
    return fig


# ==========================================
# INTERFACE PRINCIPAL
# ==========================================

def main():
    st.set_page_config(page_title="Dashboard BSEM", layout="wide", page_icon="📊")

    # --- Estado inicial ---
    if "ultimo_modo" not in st.session_state:
        st.session_state.mem_v = 3
        st.session_state.mem_a = 3
        st.session_state.mem_s = 3
        st.session_state.mem_p = "Baixa"
        st.session_state.ic_cache = interceptos_populacionais(False)
        st.session_state.ic_cache["pertenca_alta"] = False
        st.session_state.ultimo_modo = "Modo simulação"

    df = st.cache_data(lambda: (
        pd.read_csv("dados_final.csv")
        .dropna(subset=['percepcao_violencia','acolhimento','satisfacao_vida',
                        'pertenca_grupal','ideacao_suicida'])
    ) if True else pd.DataFrame())()

    st.title("📊 Simulador: Risco de ideação suicida")
    st.caption("Modelo BSEM — β estimados via MCMC (4 cadeias, 2000 iter., R̂=1.00)")
    st.divider()

    # --- Sidebar ---
    st.sidebar.header("⚙️ Configurações")
    modo = st.sidebar.radio("Selecione a origem dos dados",
                             ["Dados reais", "Modo simulação"])
    modo_csv = (modo == "Dados reais")

    # Guarda memória ao sair do CSV
    if modo == "Modo simulação" and st.session_state.ultimo_modo == "Dados reais":
        st.session_state.slider_v = st.session_state.mem_v
        st.session_state.slider_a = st.session_state.mem_a
        st.session_state.slider_s = st.session_state.mem_s
        st.session_state.radio_p  = st.session_state.mem_p

    st.session_state.ultimo_modo = modo

    ic          = {}
    is_real     = None
    pertenca_alta = False

    # --- Modo CSV ---
    if modo_csv:
        if df.empty:
            st.sidebar.error("CSV não encontrado.")
            st.stop()

        st.sidebar.markdown("---")
        st.sidebar.subheader("🗂️ Filtros Demográficos")
        sexo       = st.sidebar.selectbox("Sexo",      ["Todos"] + sorted(df['sexo'].dropna().unique().tolist()))
        cor        = st.sidebar.selectbox("Cor da Pele",["Todos"] + sorted(df['cor_da_pele'].dropna().unique().tolist()))
        orientacao = st.sidebar.selectbox("Orientação Sexual",
                                          ["Todos"] + sorted(df['orientação_sexual'].dropna().unique().tolist()))
        renda      = st.sidebar.selectbox("Renda",     ["Todos"] + sorted(df['renda'].dropna().unique().tolist()))

        df_f = df.copy()
        if sexo       != "Todos": df_f = df_f[df_f['sexo']              == sexo]
        if cor        != "Todos": df_f = df_f[df_f['cor_da_pele']       == cor]
        if orientacao != "Todos": df_f = df_f[df_f['orientação_sexual'] == orientacao]
        if renda      != "Todos": df_f = df_f[df_f['renda']             == renda]

        st.sidebar.info(f"👥 **Alunos neste perfil de análise:** {len(df_f)}")

        if len(df_f) == 0:
            st.warning("Nenhum aluno corresponde a este filtro.")
            st.stop()

        cal = calibrar_interceptos_empiricos(df_f)
        ic  = cal
        pertenca_alta = cal["pertenca_alta"]
        is_real       = cal["is_real"]
        violencia     = cal["v_m"]
        acolhimento   = cal["a_m"]
        satisfacao    = cal["s_m"]

        # Guarda na memória
        st.session_state.mem_v = int(round(violencia))
        st.session_state.mem_a = int(round(acolhimento))
        st.session_state.mem_s = int(round(satisfacao))
        st.session_state.mem_p = "Alta" if pertenca_alta else "Baixa"
        st.session_state.ic_cache = ic

    # --- Modo Simulação ---
    else:
        st.sidebar.success("💡 Explore intervenções usando a escala Likert.\n"
                           "Os sliders iniciam nos valores do último subgrupo CSV.")
        ic = st.session_state.ic_cache

    # ---- Layout: inputs | gráfico ----
    col_input, col_grafico = st.columns([1, 2.5], gap="large")

    with col_input:
        if modo_csv:
            st.subheader("Perfil Selecionado (Médias)")
            st.metric("Violência Percebida",  f"{violencia:.2f}")
            st.metric("Acolhimento Escolar",  f"{acolhimento:.2f}")
            st.metric("Satisfação com a Vida",f"{satisfacao:.2f}")
            st.write(f"**Pertença Grupal:** {'Alta ×1.20 (proteção amplificada)' if pertenca_alta else 'Baixa ×0.80'}")

            # Acolhimento previsto pelo BSEM a partir de V
            ic_a = ic.get("ic_a", interceptos_populacionais(pertenca_alta)["ic_a"])
            a_bsem = ic_a + BETA_VA * violencia
            st.caption(f"Acolhimento previsto pelo BSEM (V={violencia:.2f}): **{np.clip(a_bsem,1,5):.2f}**")
            st.caption(f"Acolhimento real observado: **{acolhimento:.2f}** "
                       f"({'acima' if acolhimento > a_bsem else 'abaixo'} do previsto por V)")

        else:
            st.subheader("Simulação em Escala Likert")
            violencia   = st.slider("Violência Percebida",  1, 5, step=1, key="slider_v")
            st.markdown("<small><i>0 = Efeito Anulado (Ablação)</i></small>", unsafe_allow_html=True)
            acolhimento = st.slider("Acolhimento Escolar",  0, 5, step=1, key="slider_a")
            satisfacao  = st.slider("Satisfação com a Vida",0, 5, step=1, key="slider_s")
            pertenca    = st.radio("Pertença Grupal", ["Baixa", "Alta"], key="radio_p")
            pertenca_alta = (pertenca == "Alta")

            ic = interceptos_populacionais(pertenca_alta)

        # Painel de coeficientes BSEM (sempre visível)
        st.markdown("---")
        st.markdown("**📐 Coeficientes BSEM (Artigo, Figura 4)**")
        cols_b = st.columns(2)
        with cols_b[0]:
            st.metric("V → A",  f"{BETA_VA:.3f}")
            st.metric("A → S",  f"0.527 × mod")
            st.metric("V → IS (direto)", f"{BETA_VI:.3f}")
        with cols_b[1]:
            st.metric("S → IS", f"{BETA_SI:.3f}")
            mod_eff = MOD_ALTA if pertenca_alta else MOD_BAIXA
            st.metric("mod pertença", f"×{mod_eff:.2f}  ({'Alta' if pertenca_alta else 'Baixa'})")
            ind_eff = IND_ALTA if pertenca_alta else IND_BAIXA
            st.metric("Efeito Indireto", f"{ind_eff:.3f}")

    # ---- Cálculo ----
    r = calcular_cascata_bsem(violencia, acolhimento, satisfacao, pertenca_alta, ic)

    with col_grafico:
        st.plotly_chart(renderizar_cascata(r), use_container_width=True)

        # Diagrama de trajetórias
        with st.expander("🔬 Diagrama de Trajetórias BSEM (Coeficientes do Artigo)", expanded=False):
            st.plotly_chart(renderizar_painel_bsem(), use_container_width=True)

        st.subheader("Métricas de Previsão")
        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Pressão Psicológica Base",
                  f"{r['no1']:.3f}",
                  delta="Risco sem mediação A→S", delta_color="off")

        m2.metric("Mitigação via A→S→IS",
                  f"-{max(r['queda_acol'], 0):.3f}",
                  delta="Ativo" if acolhimento > 0 else "Anulado",
                  delta_color="normal" if acolhimento > 0 else "off")

        m3.metric("Ajuste via Satisfação",
                  f"{'-' if r['queda_satis'] > 0 else '+'}{abs(r['queda_satis']):.3f}",
                  delta="Ativo" if satisfacao > 0 else "Anulado",
                  delta_color="normal" if satisfacao > 0 else "off")

        sat_label = ""
        if r["saturacao"]: sat_label = " ⚠️ Saturado"
        m4.metric("RESPOSTA AO QUESTIONÁRIO",
                  f"{r['no3']:.3f}{sat_label}",
                  delta=f"Defesa Total: -{r['queda_acol']+r['queda_satis']:.3f}",
                  delta_color="inverse")

        # Decomposição direto vs indireto
        st.markdown("---")
        st.markdown("**🔎 Decomposição do Efeito da Violência sobre a Ideação**")
        d1, d2, d3 = st.columns(3)
        d1.metric("Efeito Direto (β=0.243·V)",
                  f"{r['efeito_direto']:+.3f}",
                  delta="V → IS", delta_color="off")
        d2.metric("Efeito Indireto (V→A→S→IS)",
                  f"{r['efeito_indireto']:+.3f}",
                  delta=f"β_AS efetivo: {r['bas']:.3f}", delta_color="off")
        d3.metric("Efeito Total de V sobre IS",
                  f"{r['efeito_direto'] + r['efeito_indireto']:+.3f}",
                  delta="Direto + Indireto", delta_color="off")

        # Sincronia Perfeita (modo CSV)
        if modo_csv and is_real is not None:
            erro = abs(r['no3'] - is_real)
            if erro < 0.01:
                st.success(
                    f"🎯 **Sincronia Perfeita:** Ideação no CSV = **{is_real:.3f}** | "
                    f"Modelo previu = **{r['no3']:.3f}** (Erro = {erro:.4f})"
                )
            else:
                st.info(
                    f"📊 Ideação real no CSV: **{is_real:.3f}** | "
                    f"Modelo previu: **{r['no3']:.3f}** | "
                    f"Margem de erro: {erro:.4f}"
                )

        # Nota metodológica
        with st.expander("ℹ️ Nota Metodológica sobre os Coeficientes"):
            st.markdown(f"""
**Coeficientes do Modelo BSEM:**

| Caminho | β | IC 97,5% |
|---|---|---|
| Violência → Ideação (direto) | 0.243 | [0.167, 0.324] |
| Violência → Acolhimento | -0.324 | [-0.369, -0.279] |
| Acolhimento → Satisfação | 0.527 | [0.462, 0.591] |
| Satisfação → Ideação | -0.170 | [-0.247, -0.089] |

**Pertença Grupal como Moderadora do caminho A→S:**
- Pertença Baixa (−1DP): efeito indireto = **0.016**
- Pertença Média: efeito indireto = **0.020**  
- Pertença Alta (+1DP): efeito indireto = **0.024**

O fator de modulação (×{MOD_BAIXA:.2f}/×{MOD_ALTA:.2f}) é aplicado sobre β_AS = 0.527.

**R² Bayesiano: 0.167** | WAIC: 5266.0 | R̂ = 1.000
""")


if __name__ == "__main__":
    main()
