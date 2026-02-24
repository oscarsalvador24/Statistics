import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import pingouin as pg
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import io

st.set_page_config(layout="wide", page_title="Sistema Estadistico")

@st.cache_data
def cargar_datos(archivo):
    return pd.read_excel(archivo).dropna()

def calcular_ic_95(data):
    mean = np.mean(data)
    sem = stats.sem(data)
    ic = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
return mean, ic

st.title("Plataforma de Analisis e Inferencia en la Nube")

st.warning("Recordatorio de Privacidad y Seguridad Legal: Asegurese de que el archivo Excel esta completamente anonimizado antes de subirlo.")

archivo_subido = st.file_uploader("Carga de datos en formato Excel", type=["xlsx", "xls"])

if archivo_subido:
    df = cargar_datos(archivo_subido)

    palabras_clave_privadas = ["nombre", "apellido", "direccion", "postal", "correo", "email", "telefono", "tlf", "movil", "dni", "nif", "nie", "pasaporte", "historia", "hc", "nhc", "identificacion", "paciente"]

    columnas_detectadas = [col for col in df.columns if any(palabra in str(col).lower() for palabra in palabras_clave_privadas)]

if len(columnas_detectadas) > 0:
    st.warning("Notificacion: Se han detectado columnas con posible informacion privada (" + ", ".join(columnas_detectadas) + "). Se recomienda precaucion con el manejo de estos datos.")

    cols_num = df.select_dtypes(include=np.number).columns.tolist()
    cols_cat = df.select_dtypes(exclude=np.number).columns.tolist()

    tab_desc, tab_inf, tab_mod = st.tabs(["Descriptiva", "Inferencia", "Modelos"])

with tab_desc:
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        var_estudio = st.selectbox("Variable principal", cols_num, key="d1")
        var_agrupar = st.selectbox("Agrupar por", [None] + cols_cat, key="d2")
    with c2:
        stats_df = df.groupby(var_agrupar)[var_estudio].describe() if var_agrupar else df[var_estudio].describe()
        st.dataframe(stats_df)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            stats_df.to_excel(writer, sheet_name='Descriptiva')
            st.download_button("Descargar en Excel", data=buffer.getvalue(), file_name="estadistica_descriptiva.xlsx")
    with c3:
        fig_desc = px.box(df, y=var_estudio, x=var_agrupar, points="all")
        st.plotly_chart(fig_desc, use_container_width=True)
        
with tab_inf:
    c_cfg, c_res = st.columns([1, 2])
    with c_cfg:
        var_dep = st.selectbox("Variable Dependiente", cols_num, key="i1")
        var_indep = st.selectbox("Variable Independiente", cols_cat, key="i2")
        es_apareada = st.checkbox("Medidas apareadas")
        
        grupos = df[var_indep].unique()
        normalidad_cumplida = True
        
        for g in grupos:
            data_g = df[df[var_indep] == g][var_dep]
            if len(data_g) >= 3:
                stat, p_shap = stats.shapiro(data_g)
                if p_shap <= 0.05: 
                    normalidad_cumplida = False
        
            n_grupos = len(grupos)
        if n_grupos == 2:
            if normalidad_cumplida:
                test_sugerido = "t-Student" if not es_apareada else "t-Student Apareada"
            else:
                test_sugerido = "Mann-Whitney" if not es_apareada else "Wilcoxon"
        else:
            test_sugerido = "ANOVA" if normalidad_cumplida else "Kruskal-Wallis"
        
        st.write("Sugerencia por Shapiro-Wilk: " + test_sugerido)
        opciones_test = ["t-Student", "Mann-Whitney", "Wilcoxon", "ANOVA", "Kruskal-Wallis"]
        idx_test = opciones_test.index(test_sugerido) if test_sugerido in opciones_test else 3
        test_final = st.selectbox("Confirmar Test", opciones_test, index=idx_test)
        
    with c_res:
        try:
            res_ic = []
            for g in grupos:
                m, ic = calcular_ic_95(df[df[var_indep] == g][var_dep])
                res_ic.append("Grupo " + str(g) + " - Media: " + str(round(m, 2)) + " (IC 95%: " + str(round(ic[0], 2)) + " a " + str(round(ic[1], 2)) + ")")
            
            for item in res_ic: 
                st.write(item)
            
            if test_final in ["t-Student", "Mann-Whitney", "Wilcoxon"]:
                d1 = df[df[var_indep]==grupos[0]][var_dep]
                d2 = df[df[var_indep]==grupos[1]][var_dep]
                
                if test_final == "t-Student": 
                    res = pg.ttest(d1, d2, paired=es_apareada)
                elif test_final == "Mann-Whitney": 
                    res = pg.mwu(d1, d2)
                else: 
                    res = pg.wilcoxon(d1, d2)
                
                p_val = res['p-val'].values[0]
            
            elif test_final == "ANOVA":
                res = pg.anova(data=df, dv=var_dep, between=var_indep)
                p_val = res['p-unc'].values[0]
            else:
                res = pg.kruskal(data=df, dv=var_dep, between=var_indep)
                p_val = res['p-unc'].values[0]
                
            st.write("P-Valor Final: " + str(round(p_val, 4)))
            
            fig_inf = px.violin(df, x=var_indep, y=var_dep, color=var_indep, box=True, points="all")
            st.plotly_chart(fig_inf, use_container_width=True)
            
            significancia = "existe diferencia significativa" if p_val < 0.05 else "no existe diferencia significativa"
            distribucion = "normal" if normalidad_cumplida else "no normal"
            
            informe = "Para la variable " + var_dep + ", la prueba de Shapiro-Wilk sugiere una distribucion " + distribucion + ". Se aplico la prueba " + test_final + " obteniendo un p-valor de " + str(round(p_val, 4)) + ". La interpretacion es que " + significancia + " entre los grupos analizados."
            st.text_area("Reporte de conclusiones:", value=informe, height=120)
            
        except Exception as e:
            st.write("Esperando configuracion valida o revision de datos: " + str(e))
            
with tab_mod:
    c_m1, c_m2 = st.columns([1, 2])
    with c_m1:
        tipo_mod = st.selectbox("Algoritmo", ["Lineal", "Logistica", "GLM Poisson"])
        target = st.selectbox("Variable Dependiente Y", cols_num, key="m1")
        features = st.multiselect("Variables Independientes X", cols_num + cols_cat, key="m2")
    with c_m2:
        if target and len(features) > 0:
            try:
                formula = target + " ~ " + " + ".join(features)
                if tipo_mod == "Lineal":
                    modelo = smf.ols(formula, data=df).fit()
                elif tipo_mod == "Logistica":
                    modelo = smf.logit(formula, data=df).fit()
                else:
                    modelo = smf.glm(formula, data=df, family=sm.families.Poisson()).fit()
                    
                st.text(modelo.summary().as_text())
            except Exception as e:
                st.write("Aviso en la construccion del modelo: " + str(e))
else:
st.write("Sube un documento para inicializar el sistema.")
