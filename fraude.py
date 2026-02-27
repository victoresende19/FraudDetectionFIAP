import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Detecção de Fraude", layout="wide", page_icon="🔍")

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3a5f, #2e6da4);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.metric-card h2 { font-size: 2.2rem; margin: 0; font-weight: 700; }
.metric-card p { margin: 4px 0 0 0; font-size: 0.85rem; opacity: 0.85; }
.section-title { font-size: 1.3rem; font-weight: 600; color: #00000; margin: 24px 0 12px 0; border-left: 4px solid #2e6da4; padding-left: 10px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_fraud_dataset.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Month"] = df["Timestamp"].dt.month
    df["Hour"] = df["Timestamp"].dt.hour
    return df

df = load_data()
st.title("🔍 Detecção de Fraudes Bancárias - CAIXA & FIAP")
st.write("Por Gisele Oliveira, Thaisa Guio e Victor Resende.")
tab1, tab2, tab3, tab4 = st.tabs(["📖 Sobre", "📊 Análise Exploratória", "🌟 Modelagem", "💼 Conclusão Executiva"])

# ── TAB SOBRE ─────────────────────────────────────────────────────────────────
with tab1:
    st.title("📖 Contextualização do Projeto")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### O Problema
        Fraudes financeiras representam um dos maiores desafios do setor bancário moderno. Segundo estimativas globais,
        bilhões de dólares são perdidos anualmente em transações fraudulentas, afetando tanto instituições financeiras
        quanto seus clientes. A detecção manual é inviável dado o volume massivo de transações - um grande banco pode
        processar dezenas de milhões de operações por dia.

        ### Detecção automática com Machine Learning
        Modelos de Machine Learning conseguem aprender padrões complexos e não lineares a partir de dados históricos.
        Eles avaliam simultaneamente dezenas de variáveis para calcular em milissegundos a probabilidade de uma transação ser fraudulenta. 
        Isso permite bloqueios automáticos em tempo real, reduzindo perdas e melhorando a experiência do cliente legítimo.

        #### A escolha do modelo - Classificação
        Muitos sistemas antifraude ainda dependem de **regras fixas** (ex.: "bloquear transações acima de U$ 5.000 fora
        do país"), que são facilmente contornadas por fraudadores sofisticados e geram alta taxa de falsos positivos,
        frustrando clientes. Este projeto demonstra como modelos probabilísticos de classificação podem complementar ou substituir regras rígidas, oferecendo:

        - **Maior precisão**: menos falsos positivos, menos clientes frustrados
        - **Adaptabilidade**: o modelo aprende novos padrões de fraude com retreinamento
        - **Interpretabilidade**: é possível entender quais variáveis mais influenciam a decisão
        - **Escalabilidade**: aplicável a qualquer volume de transações em tempo real

        ##### Modelo utilizados 
        Visando a utilização de métodos probabilísticos para a classificação de transações de Fraude com Machine Learning, foram utilizados os modelos:
        - Regressão Logística
        - Naive Bayes

        Ambos modelos foram escolhidos visando a parcimônia, ou seja, atributos como simplicidade e interpretação.
        Além disso, como demonstrado a seguir pelo dicionário de dados, o alvo para realizar o treinamento e comparação dos modelos se da pela variável ```Fraud_Label```.

        ### Sobre os Dados
        O dataset é provinente de um repositório Kaggle, que pode ser acessado [clicando aqui](https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset?resource=download).
        Além disso, os dados contém **50.000 transações sintéticas** com 21 variáveis, incluindo valor da transação, tipo,
        dispositivo, localização, score de risco, histórico de atividade e o rótulo de fraude (`Fraud_Label`), bem como outras variáveis explicadas no dicionário de dados.
        """)
    with col2:
        st.markdown("""
        #### 📌 Dicionário dos dados
        | Variável | Descrição |
        |---|---|
        | Transaction_Amount | Valor da transação |
        | Transaction_Type | Tipo (POS, ATM, Online...) |
        | Account_Balance | Saldo da conta |
        | Device_Type | Dispositivo usado |
        | Location | Cidade da transação |
        | IP_Address_Flag | IP suspeito (0/1) |
        | Previous_Fraudulent_Activity | Histórico de fraude |
        | Risk_Score | Score de risco |
        | Authentication_Method | Método de autenticação |
        | Card_Age | Idade do cartão (dias) |
        | Transaction_Distance | Distância da transação |
        | **Fraud_Label** | **Target (0=legítimo, 1=fraude)** |
        """)
    
    st.markdown("Visualize o dataframe completo abaixo:")
    st.dataframe(df)

# ── TAB ANÁLISE EXPLORATÓRIA ───────────────────────────────────────────────────
with tab2:
    st.title("📊 Análise Exploratória dos Dados")

    fraud_rate = df["Fraud_Label"].mean() * 100
    total_fraud = df["Fraud_Label"].sum()
    avg_fraud_amount = df[df["Fraud_Label"] == 1]["Transaction_Amount"].mean()
    avg_legit_amount = df[df["Fraud_Label"] == 0]["Transaction_Amount"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [f"{len(df):,.0f}".replace(",", "."), f"{total_fraud:,.0f}".replace(",", "."), f"{fraud_rate:.1f}%", f"U$ {avg_fraud_amount:.2f}"],
        ["Total de Transações", "Transações Fraudulentas", "Taxa de Fraude", "Valor Médio (Fraude)"]
    ):
        col.markdown(f'<div class="metric-card"><h2>{val}</h2><p>{label}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Distribuição da target
    st.markdown('<div class="section-title">Distribuição da Variável Target</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        counts = df["Fraud_Label"].value_counts().reset_index()
        counts.columns = ["Classe", "Quantidade"]
        counts["Classe"] = counts["Classe"].map({0: "Legítima", 1: "Fraude"})
        fig = px.pie(counts, names="Classe", values="Quantidade",
                     color_discrete_sequence=["#2e6da4", "#e74c3c"],
                     hole=0.4, title="Proporção de Fraudes")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(counts, x="Classe", y="Quantidade",
                      color="Classe", color_discrete_map={"Legítima": "#2e6da4", "Fraude": "#e74c3c"},
                      title="Volume por Classe", text="Quantidade")
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)
    st.info("O dataset apresenta desbalanceamento moderado (~32% fraudes), o que é comum em dados bancários reais e deve ser considerado na modelagem.")

    # Análise temporal
    st.markdown('<div class="section-title">Análise Temporal</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        hourly = df.groupby(["Hour", "Fraud_Label"]).size().reset_index(name="count")
        hourly["Classe"] = hourly["Fraud_Label"].map({0: "Legítima", 1: "Fraude"})
        fig = px.line(hourly, x="Hour", y="count", color="Classe",
                      color_discrete_map={"Legítima": "#2e6da4", "Fraude": "#e74c3c"},
                      title="Transações por Hora do Dia", labels={"Hour": "Hora", "count": "Qtd"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        monthly = df.groupby(["Month", "Fraud_Label"]).size().reset_index(name="count")
        monthly["Classe"] = monthly["Fraud_Label"].map({0: "Legítima", 1: "Fraude"})
        fig = px.bar(monthly, x="Month", y="count", color="Classe",
                     color_discrete_map={"Legítima": "#2e6da4", "Fraude": "#e74c3c"},
                     title="Transações por Mês", barmode="group",
                     labels={"Month": "Mês", "count": "Qtd"})
        st.plotly_chart(fig, use_container_width=True)
    # st.info("Fraudes são mais frequentes nas madrugadas (0h–5h), sugerindo que autenticações fora do horário comercial merecem atenção redobrada.")
    st.info("""
        Durante o horário comercial foi identificado que os picos de transações legítimas e fraudulentas coincidiram às 11h. 

        Assim, é recomendada a utilização do modelo para triagem das transações, servindo como filtro para uma posterior análise humana, dada a quantidade de transações totais.
        
        Além disso, ao contrário do que é esperado, em horários não comerciais, como às 21h, foi observado o pico de transações legítimas e um vale para transações fraudulentas.
        """)

    # Análise categórica
    st.markdown('<div class="section-title">Análise por Variáveis Categóricas</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fraud_by_type = df.groupby("Transaction_Type")["Fraud_Label"].mean().reset_index()
        fraud_by_type.columns = ["Tipo", "Taxa de Fraude"]
        fraud_by_type = fraud_by_type.sort_values("Taxa de Fraude", ascending=True)
        fig = px.bar(fraud_by_type, x="Taxa de Fraude", y="Tipo", orientation="h",
                     title="Taxa de Fraude por Tipo de Transação",
                     color="Taxa de Fraude", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

        fraud_by_auth = df.groupby("Card_Type")["Fraud_Label"].mean().reset_index()
        fraud_by_auth.columns = ["Método", "Taxa de Fraude"]
        fraud_by_auth = fraud_by_auth.sort_values("Taxa de Fraude", ascending=True)
        fig = px.bar(fraud_by_auth, x="Taxa de Fraude", y="Método", orientation="h",
                     title="Taxa de Fraude por Tipo de Cartão",
                     color="Taxa de Fraude", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fraud_by_auth = df.groupby("Authentication_Method")["Fraud_Label"].mean().reset_index()
        fraud_by_auth.columns = ["Método", "Taxa de Fraude"]
        fraud_by_auth = fraud_by_auth.sort_values("Taxa de Fraude", ascending=True)
        fig = px.bar(fraud_by_auth, x="Taxa de Fraude", y="Método", orientation="h",
                     title="Taxa de Fraude por Método de Autenticação",
                     color="Taxa de Fraude", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        fraud_by_auth = df.groupby("Device_Type")["Fraud_Label"].mean().reset_index()
        fraud_by_auth.columns = ["Método", "Taxa de Fraude"]
        fraud_by_auth = fraud_by_auth.sort_values("Taxa de Fraude", ascending=True)
        fig = px.bar(fraud_by_auth, x="Taxa de Fraude", y="Método", orientation="h",
                     title="Taxa de Fraude por Tipo de Dispositivo",
                     color="Taxa de Fraude", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    # Distribuição valor por classe
    st.markdown('<div class="section-title">Distribuição do Valor das Transações</div>', unsafe_allow_html=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bins = pd.cut(df["Transaction_Amount"], bins=60)
    fraud_rate_by_bin = df.groupby(bins, observed=True).apply(
        lambda x: x["Fraud_Label"].sum() / len(x) * 100
    ).reset_index()
    fraud_rate_by_bin.columns = ["bin", "fraud_rate"]
    fraud_rate_by_bin["bin_mid"] = fraud_rate_by_bin["bin"].apply(lambda x: x.mid)

    for label, color in [("Legítima", "#2e6da4"), ("Fraude", "#e74c3c")]:
        subset = df[df["Fraud_Label"].map({0: "Legítima", 1: "Fraude"}) == label]
        fig.add_trace(go.Histogram(x=subset["Transaction_Amount"], name=label, marker_color=color,
                                opacity=0.7, nbinsx=60), secondary_y=False)

    fig.add_trace(go.Scatter(x=fraud_rate_by_bin["bin_mid"], y=fraud_rate_by_bin["fraud_rate"],
                            name="% Fraude por faixa", mode="lines",
                            line=dict(color="#f39c12", width=2.5, dash="dot")), secondary_y=True)

    fig.update_layout(barmode="overlay", title="Distribuição do Valor por Classe",
                    legend=dict(orientation="h"), template="plotly_dark")
    fig.update_xaxes(title_text="Valor (R$)")
    fig.update_yaxes(title_text="Quantidade de Transações", secondary_y=False)
    fig.update_yaxes(title_text="% de Fraude", secondary_y=True, ticksuffix="%", showgrid=False)

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    É possível identificar que há um número maior de transações com valores mais baixos. Além disso, há grande frequência de transações fraudulentas com valores baixos, ao contrário do que é esperado.
    """)

    # Correlação
    st.markdown('<div class="section-title">Mapa de Correlação</div>', unsafe_allow_html=True)
    num_cols = ["Transaction_Amount", "Account_Balance", "Daily_Transaction_Count",
                "Avg_Transaction_Amount_7d", "Failed_Transaction_Count_7d", "Card_Age",
                "Transaction_Distance", "Risk_Score", "IP_Address_Flag",
                "Previous_Fraudulent_Activity", "Fraud_Label"]
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="Matriz de Correlação", aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    st.info("**Risk_Score** e **Failed_Transaction_Count_7d** apresentam as maiores correlações com **Fraud_Label**, confirmando sua relevância para modelagem.")

# ── TAB MODELAGEM ──────────────────────────────────────────────────────────────
with tab3:
    st.title("🌟 Modelagem Preditiva")

    cat_cols = ["Transaction_Type", "Device_Type", "Location", "Merchant_Category",
                "Card_Type", "Authentication_Method"]
    num_features = ["Transaction_Amount", "Account_Balance", "IP_Address_Flag",
                    "Previous_Fraudulent_Activity", "Daily_Transaction_Count",
                    "Avg_Transaction_Amount_7d", "Failed_Transaction_Count_7d",
                    "Card_Age", "Transaction_Distance", "Risk_Score", "Is_Weekend",
                    "Month", "Hour"]

    df_model = df.copy()
    le = LabelEncoder()
    for c in cat_cols:
        df_model[c + "_enc"] = le.fit_transform(df_model[c])

    enc_features = [c + "_enc" for c in cat_cols]
    all_features = num_features + enc_features

    X_full = df_model[all_features]
    y = df_model["Fraud_Label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    # Feature importance via Logistic Regression with all features
    lr_full = LogisticRegression(max_iter=1000, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    lr_full.fit(X_tr, y_tr)
    importances = np.abs(lr_full.coef_[0])
    feat_imp = pd.DataFrame({"Feature": all_features, "Importance": importances}).sort_values("Importance", ascending=False)

    st.markdown('<div class="section-title">Seleção de Features</div>', unsafe_allow_html=True)
    st.markdown("A importância das variáveis foi avaliada utilizando os coeficientes absolutos da Regressão Logística treinada com todas as features (as variáveis de ID foram descartadas). Selecione o número de features mais importantes para utilizar nos modelos preditivos abaixo.")
    n_features = st.slider("Número de features mais importantes a utilizar:", min_value=1, max_value=len(all_features), value=10)

    top_features = feat_imp["Feature"].head(n_features).tolist()
    col1, col2 = st.columns([1, 1])
    with col1:
        fig = px.bar(feat_imp.head(n_features).sort_values("Importance"),
                     x="Importance", y="Feature", orientation="h",
                     title=f"Top {n_features} Features Mais Importantes",
                     color="Importance", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(feat_imp.head(n_features).reset_index(drop=True))

    X_sel = df_model[top_features]
    X_sel_scaled = StandardScaler().fit_transform(X_sel)
    X_train, X_test, y_train, y_test = train_test_split(X_sel_scaled, y, test_size=0.2, random_state=42, stratify=y)

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    def show_metrics(y_test, y_pred, model_name, color):
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.markdown(f"#### {model_name}")
        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in zip([c1, c2, c3, c4],
                                [f"{acc:.3f}", f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"],
                                ["Acurácia", "Precisão", "Recall", "F1-Score"]):
            col.markdown(f'<div class="metric-card"><h2>{val}</h2><p>{label}</p></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale=[[0, "white"], [1, color]],
                        labels={"x": "Previsto", "y": "Real"},
                        x=["Legítima", "Fraude"], y=["Legítima", "Fraude"],
                        title=f"Matriz de Confusão - {model_name}")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander(f"📋 Ver Classification Report: **{model_name}**"):
            report = classification_report(y_test, y_pred, target_names=["Legítima", "Fraude"])
            st.code(report, language="text")

    st.markdown('<div class="section-title">Resultados dos Modelos</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ Detalhes sobre a modelagem"):
        st.markdown("""
        - **Modelos Utilizados**: foram treinados dois modelos de classificação binária: Regressão Logística e Naive Bayes por serem modelos simples e fáceis de interpretar.
        - **Seleção de Features**: as features mais importantes foram selecionadas com base nos coeficientes absolutos da Regressão Logística treinada com todas as features, garantindo que os modelos fossem treinados apenas com as variáveis mais relevantes para a detecção de fraudes.
        - **Holdout**: foram utilizados 20% dos dados para teste e o restante para treinamento, garantindo que os modelos fossem avaliados em dados não vistos durante o treinamento.
        - **Padronização**: realizada a padronização com o `StandardScaler()` dos dados apenas para o treinamento da Regressão Logística, por ser um algoritmo que sofre com variáveis de escalas diferentes. Já para o algoritmo Naive Bayes, a padronização não é necessária, pois ele é baseado em probabilidades e distribuições de frequência, e não em distâncias ou magnitudes das variáveis. Portanto, os dados foram mantidos em sua escala original para o treinamento do Naive Bayes.
        - **Dummização**: foi realizado o encoding nas variáveis categóricas para que elas pudessem ser utilizadas em ambos os modelos, garantindo uma comparação justa entre eles.            
        - **Balanceamento de Classes**: a Regressão Logística foi treinada com o parâmetro `class_weight='balanced'` para lidar com o desbalanceamento do dataset, enquanto o Naive Bayes, por ser um modelo probabilístico, lida naturalmente com classes desbalanceadas.
    """)
    col1, col2 = st.columns(2)
    with col1:
        show_metrics(y_test, y_pred_lr, "Regressão Logística", "#2e6da4")
    with col2:
        show_metrics(y_test, y_pred_nb, "Naive Bayes", "#e74c3c")

    # Previsão interativa
    st.markdown('<div class="section-title">Fazer Previsão</div>', unsafe_allow_html=True)
    st.markdown("Preencha os campos abaixo para prever se uma transação é fraudulenta:")

    scaler_pred = StandardScaler()
    scaler_pred.fit(df_model[top_features])

    with st.form("prediction_form"):
        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(top_features):
            with cols[i % 3]:
                original_col = feat.replace("_enc", "")
                if feat.endswith("_enc"):
                    options = sorted(df[original_col].unique().tolist())
                    sel = st.selectbox(original_col, options, key=feat)
                    input_data[feat] = le.fit_transform(df[original_col])[df[original_col].tolist().index(sel)] if sel in df[original_col].tolist() else 0
                else:
                    min_v = float(df_model[feat].min())
                    max_v = float(df_model[feat].max())
                    mean_v = float(df_model[feat].mean())
                    input_data[feat] = st.number_input(feat, min_value=min_v, max_value=max_v, value=mean_v, key=feat)

        threshold = st.slider("🎚️ Threshold de classificação", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                          help="Probabilidade mínima para classificar como Fraude. Valores menores aumentam a sensibilidade do modelo.")

        submitted = st.form_submit_button("🔍 Prever")

    if submitted:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler_pred.transform(input_df)

        prob_lr = lr.predict_proba(input_scaled)[0][1]
        prob_nb = nb.predict_proba(input_scaled)[0][1]
        pred_lr = 1 if prob_lr >= threshold else 0
        pred_nb = 1 if prob_nb >= threshold else 0

        st.markdown("### Resultado da Previsão")
        c1, c2 = st.columns(2)
        with c1:
            label = "🚨 FRAUDE" if pred_lr == 1 else "✅ LEGÍTIMA"
            bg = "#e74c3c" if pred_lr == 1 else "#27ae60"
            st.markdown(f'<div class="metric-card" style="background:{bg}"><h2>{label}</h2><h4>Regressão Logística (Probabilidade: {prob_lr:.1%})</h4></div>', unsafe_allow_html=True)
        with c2:
            label2 = "🚨 FRAUDE" if pred_nb == 1 else "✅ LEGÍTIMA"
            bg2 = "#e74c3c" if pred_nb == 1 else "#27ae60"
            st.markdown(f'<div class="metric-card" style="background:{bg2}"><h2>{label2}</h2><h4>Naive Bayes (Probabilidade: {prob_nb:.1%})</h4></div>', unsafe_allow_html=True)

with tab4:
    st.title("💼 Conclusão Executiva")

    avg_fraud_val = df[df["Fraud_Label"] == 1]["Transaction_Amount"].mean()
    fraud_rate_val = df["Fraud_Label"].mean()
    monthly_transactions = 1_000_000
    monthly_frauds = monthly_transactions * fraud_rate_val
    monthly_exposure = monthly_frauds * avg_fraud_val
    recovery_80 = monthly_exposure * 0.80

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#162840; border-radius:12px; padding:24px; border-top: 4px solid #27ae60; height:100%">
            <div style="font-size:1.6rem; margin-bottom:8px">✅</div>
            <div style="color:#7eb8e8; font-weight:700; font-size:1rem; margin-bottom:12px; text-transform:uppercase; letter-spacing:.05em">O modelo resolve o gap?</div>
            <div style="color:#dce6f0; font-size:0.9rem; line-height:1.7">
                Parcialmente sim. Com F1-Score acima de <strong style="color:#7eb8e8">0.80</strong> na Regressão Logística, 
                os modelos detectam padrões sem depender de regras fixas. O gap de rigidez é endereçado, 
                mas a resolução completa exige retreinamento contínuo com dados reais e monitoramento de drift, 
                dado que fraudes evoluem rapidamente.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#162840; border-radius:12px; padding:24px; border-top: 4px solid #e74c3c; height:100%">
            <div style="font-size:1.6rem; margin-bottom:8px">⚠️</div>
            <div style="color:#7eb8e8; font-weight:700; font-size:1rem; margin-bottom:12px; text-transform:uppercase; letter-spacing:.05em">Onde ele falha?</div>
            <div style="color:#dce6f0; font-size:0.9rem; line-height:1.7">
                Alta taxa de <strong style="color:#e74c3c">falsos negativos</strong> em transações de baixo valor, 
                onde fraudes mimetizam comportamento legítimo. O Naive Bayes apresenta probabilidades comprimidas, 
                dificultando separação em casos limítrofes. Ambos os modelos sofrem com 
                <strong style="color:#e74c3c">LabelEncoder em variáveis nominais</strong>, 
                que introduz ordenação artificial inexistente.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#162840; border-radius:12px; padding:24px; border-top: 4px solid #2e6da4; height:100%">
            <div style="font-size:1.6rem; margin-bottom:8px">🚀</div>
            <div style="color:#7eb8e8; font-weight:700; font-size:1rem; margin-bottom:12px; text-transform:uppercase; letter-spacing:.05em">O que falta para produção?</div>
            <div style="color:#dce6f0; font-size:0.9rem; line-height:1.7">
                <span style="color:#7eb8e8">-</span> Pipeline em tempo real para acompanhamento<br>
                <span style="color:#7eb8e8">-</span> Retreinamento automatizado contra data drift<br>
                <span style="color:#7eb8e8">-</span> Explicabilidade com SHAP<br>
                <span style="color:#7eb8e8">-</span> Monitoramento de indicadores, taxas e métricas<br>
                <span style="color:#7eb8e8">-</span> Infraestrutura adequada na nuvem<br>
                <span style="color:#7eb8e8">-</span> Mensageria para comunicação com o cliente<br>
                <span style="color:#7eb8e8">-</span> Filas de revisão humana para casos limítrofes
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">💰 Potencial de Geração de Valor</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#162840; border-radius:12px; padding:24px; margin-bottom:20px; border-left: 4px solid #f39c12">
        <div style="color:#f39c12; font-weight:700; font-size:0.85rem; text-transform:uppercase; letter-spacing:.08em; margin-bottom:10px">📐 Premissas do cálculo</div>
        <div style="color:#dce6f0; font-size:0.9rem; line-height:1.8">
            Base hipotética de <strong style="color:#7eb8e8">1.000.000 transações/mês</strong> &nbsp;-&nbsp;
            Taxa de fraude real do dataset: <strong style="color:#7eb8e8">{fraud_rate_val:.1%}</strong> &nbsp;-&nbsp;
            Valor médio de fraude real: <strong style="color:#7eb8e8">U$ {avg_fraud_val:.2f}</strong> &nbsp;-&nbsp;
            Recall estimado do modelo: <strong style="color:#7eb8e8">80%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in zip(
        [c1, c2, c3, c4],
        [f"{monthly_frauds:,.0f}".replace(",", "."), f"U$ {avg_fraud_val:.2f}", f"U$ {monthly_exposure/1e6:.1f}M", f"U$ {recovery_80/1e6:.1f}M"],
        ["Fraudes esperadas/mês", "Ticket médio de fraude", "Exposição mensal total", "Recuperação potencial (80%)"],
        ["#2e6da4", "#e74c3c", "#c0392b", "#27ae60"]
    ):
        col.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}22,{color}44); border:1px solid {color}88;
                    border-radius:12px; padding:20px; text-align:center">
            <div style="font-size:1.8rem; font-weight:800; color:white; margin-bottom:4px">{val}</div>
            <div style="font-size:0.78rem; color:#dce6f0; opacity:0.85">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#162840; border-radius:12px; padding:20px; margin-top:16px; font-size:0.9rem; color:#dce6f0; line-height:1.8">
        Com <strong style="color:#7eb8e8">{monthly_frauds:,.0f} fraudes esperadas por mês</strong> e ticket médio de 
        <strong style="color:#7eb8e8">U$ {avg_fraud_val:.2f}</strong>, a exposição total chega a 
        <strong style="color:#e74c3c">U$ {monthly_exposure/1e6:.1f}M/mês</strong>. 
        Um modelo com Recall de 80% (próximo ao observado na Regressão Logística) permitiria recuperar 
        <strong style="color:#27ae60">U$ {recovery_80/1e6:.1f}M mensais</strong>, além de reduzir custos operacionais 
        de análise manual e fortalecer a posição regulatória do banco perante o Banco Central.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Riscos e Limitações</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    risks = [
        ("⚙️", "Riscos Técnicos", "#e74c3c", [
            "Dataset sintético: padrões podem não refletir fraudes reais",
            "Para o modelo Naive Bayes, verificar a normalidade das features",
            # "LabelEncoder introduz ordem artificial em variáveis nominais",
            "Ausência de validação temporal no split treino/teste",
            "Modelos lineares limitados para interações complexas",
            "Sem tratamento explícito de outliers nos coeficientes",
        ]),
        ("⚖️", "Riscos de Viés", "#f39c12", [
            "Discriminação por localização ou dispositivo (proxies socioeconômicos)",
            "Features como Location podem ser proxies de variáveis protegidas",
            "Desbalanceamento residual pode ignorar subgrupos minoritários",
            "Threshold único ignora heterogeneidade comportamental dos clientes",
        ]),
        ("📋", "Riscos Regulatórios", "#2e6da4", [
            "LGPD exige base legal para uso de dados em decisões automatizadas",
            "Banco Central exige explicabilidade (Resolução BCB nº 85)",
            "Bloqueios automáticos podem gerar passivo no Procon",
            "Auditoria periódica obrigatória para ausência de discriminação algorítmica",
        ]),
    ]
    for col, (icon, title, color, items) in zip([c1, c2, c3], risks):
        items_html = "".join([f'<div style="display:flex;gap:8px;margin-bottom:8px"><span style="color:{color};margin-top:2px">▸</span><span>{item}</span></div>' for item in items])
        col.markdown(f"""
        <div style="background:#162840; border-radius:12px; padding:24px; border-top:4px solid {color}; height:100%">
            <div style="font-size:1.5rem; margin-bottom:8px">{icon}</div>
            <div style="color:{color}; font-weight:700; font-size:0.95rem; margin-bottom:16px; text-transform:uppercase; letter-spacing:.05em">{title}</div>
            <div style="color:#dce6f0; font-size:0.85rem; line-height:1.6">{items_html}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.write("Imersão CaixaVerso Especialista em IA 2026 - Caixa & FIAP.")


