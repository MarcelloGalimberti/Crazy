#!/usr/bin/env python
# coding: utf-8

# Importa librerie
import streamlit as st
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from PIL import Image # serve per l'immagine?
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
st.set_option('deprecation.showPyplotGlobalUse', False)
from io import BytesIO
import io
from io import StringIO
import math
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
st.set_page_config(layout="wide")

# Layout iniziale e caricamento dati

url_immagine = 'https://github.com/MarcelloGalimberti/Crazy/blob/main/logo_crazy.png?raw=true'

col_1, col_2 = st.columns([2, 3])

with col_1:
    st.image(url_immagine, width=500)

with col_2:
    st.title('Organizzazione linea')

uploaded_input = st.file_uploader("Carica file input")
if not uploaded_input:
    st.stop()
df_input=pd.read_excel(uploaded_input)

st.header('File input', divider='violet')
st.dataframe(df_input, width=1500)


#df_input = pd.read_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Crazy Idea/Pianificazione CVL/Balancing/Tempi_ciclo.xlsx')

# ### Grafo iniziale per calcolo distanze

G = nx.from_pandas_edgelist(df_input, source = 'Source', target = 'Target')

# ### Funzione principale

def processa_raw(df_input, takt):
    df_raw = df_input.copy()
    df_raw['Takt_Time'] = takt
    df_raw['Risorse'] = df_raw['TC']/df_raw['Takt_Time']
    df_raw['N_macchine'] = df_raw['Risorse'].apply(lambda x: math.ceil(x))
    df_raw['Saturazione']=df_raw['Risorse']/df_raw['N_macchine']
    lista_source = list(df_raw['Source'])
    comb = combinations(lista_source, 2)
    results = [x for x in comb]
    df_abbinamenti = pd.DataFrame(results, columns=['Op_1','Op_2'])
    df_saturazione_op_1 = df_abbinamenti.merge(df_raw, how = 'left', left_on=['Op_1'], right_on=['Source'])
    df_saturazione_op_1=df_saturazione_op_1[['Op_1','Op_2','Saturazione']]
    df_saturazione_op_1.rename(columns={'Saturazione': 'Saturazione_Op_1'}, inplace=True)
    df_saturazione_op_2 = df_saturazione_op_1.merge(df_raw, how = 'left', left_on=['Op_2'], right_on=['Source'])
    df_saturazione_op_2 = df_saturazione_op_2[['Op_1','Op_2','Saturazione_Op_1','Saturazione']]
    df_saturazione_op_2.rename(columns={'Saturazione': 'Saturazione_Op_2'}, inplace=True)
    df_saturazione_op_2['Saturazione_abbinamento']= df_saturazione_op_2['Saturazione_Op_1']+df_saturazione_op_2['Saturazione_Op_2']
    df_saturazione_op_2['N_archi'] = ''
    for i in range (len (df_saturazione_op_2)):
        df_saturazione_op_2.loc[i,'N_archi'] = len(nx.bidirectional_shortest_path(G, df_saturazione_op_2.loc[i,'Op_1'],
                                                                              df_saturazione_op_2.loc[i,'Op_2']))-1
    df_saturazione_op_2['Saturazione_abbinamento'] = df_saturazione_op_2['Saturazione_abbinamento']+0.025*df_saturazione_op_2['N_archi']
    df_saturazione_op_2.drop(columns=['N_archi'], inplace=True)
    df_saturazioni_fattibili = df_saturazione_op_2[df_saturazione_op_2['Saturazione_abbinamento'] <= 1]
    df_saturazioni_fattibili.reset_index(drop = True, inplace=True)
    df_saturazioni_fattibili.sort_values(by='Saturazione_abbinamento',ascending=False,inplace=True)
    df_saturazioni_fattibili.reset_index(drop = True, inplace=True)
    df_saturazioni_fattibili['Abbinabili'] = True
    abbinamenti = pd.DataFrame(columns = df_saturazioni_fattibili.columns)
    for i in range (len(df_saturazioni_fattibili)):
        if df_saturazioni_fattibili.loc[i,'Abbinabili'] == False:
            pass
        else:
            best_op_1 = df_saturazioni_fattibili.loc[i,'Op_1']
            best_op_2 = df_saturazioni_fattibili.loc[i,'Op_2']
            l = len(abbinamenti)
            abbinamenti.loc[l+1] = df_saturazioni_fattibili.iloc[i,:]
            df_saturazioni_fattibili['Abbinabili'] = ((df_saturazioni_fattibili['Op_1']!=best_op_1) & 
                                                            (df_saturazioni_fattibili['Op_2']!=best_op_2)
                                                   & (df_saturazioni_fattibili['Op_1']!=best_op_2) & 
                                                            (df_saturazioni_fattibili['Op_2']!=best_op_1) 
                                                 & (df_saturazioni_fattibili['Abbinabili'] == True))
    abbinamenti.drop(columns='Abbinabili', inplace=True)
    numero_minimo_op = df_raw['TC'].sum()/takt
    numero_reale_op = (df_raw['N_macchine'].sum()-len(abbinamenti))
    metrica = numero_minimo_op/(df_raw['N_macchine'].sum()-len(abbinamenti))
    return metrica, numero_minimo_op, abbinamenti, numero_reale_op, df_raw


# ### Calcolo performance

performance = pd.DataFrame(columns=['Takt Time','Metrica','Op_min','Op_reale','Capi-giorno'])
for i in np.arange(0.2,4.1, 0.1):
    metrica_out, op_out , abbinamenti, numero_reale_op, df_raw = processa_raw(df_input, i)
    performance.loc[len(performance)+1,'Takt Time'] = i
    performance.loc[len(performance),'Metrica'] = metrica_out
    performance.loc[len(performance),'Op_min'] = op_out
    performance.loc[len(performance),'Op_reale'] = numero_reale_op
    performance.loc[len(performance),'Capi-giorno'] = 450/i



max_op = st.number_input('Numero massimo operatori', step =1)



# #### Bar chart performance

performance_chart = performance[(performance['Op_reale'] <= max_op)]

fig_bar = px.bar(performance_chart, x='Takt Time', y='Metrica', color='Op_reale', text = 'Op_reale', hover_name='Capi-giorno',
                      template='plotly_dark')
st.subheader('Grafico performance')
st.plotly_chart(fig_bar,use_container_width=True)

# Calcolo performance filtrato
min_op = st.number_input('Numero minimo operatori', step =1)

performance = performance[(performance['Op_reale'] <= max_op) & (performance['Op_reale'] >= min_op) ].astype('float')
performance.sort_values('Metrica', ascending=False, inplace=True)

# #### Top n soluzioni

st.subheader('Scegliere numero di soluzioni')
n = st.number_input('Numero di soluzioni', step =1)

performance = performance[:n]
performance.reset_index(drop = True, inplace=True)

st.write('Le performance delle soluzioni sono:')
st.write (performance)


# #### Top n scenari

lista_abbinamenti = []
lista_df_raw = []
for i in range (len (performance)):
    metrica_out, op_out , abbinamenti, op_out_reale, df_raw_out = processa_raw(df_raw, performance.loc[i,'Takt Time'])
    globals()[f'metrica_out_{i}'] = metrica_out
    globals()[f'op_out_{i}'] = op_out
    globals()[f'abbinamenti_{i}'] = abbinamenti
    globals()[f'op_out_reale_{i}'] = op_out_reale
    globals()[f'df_raw_out_{i}'] = df_raw_out
    lista_abbinamenti.append(globals()[f'abbinamenti_{i}'])
    lista_df_raw.append(globals()[f'df_raw_out_{i}'])

# ### Ciclo costruzione grafi degli n scenari

for i in range (n):
    db_macchine = lista_df_raw[i][['Source','Macchina','N_macchine']]
    db_macchine.loc[len(db_macchine)] = [999,'CQ',1]
    db_macchine['Risorsa'] = db_macchine['Source'].astype(str) + ' | ' + db_macchine['Macchina'] + ' | ' + db_macchine['N_macchine'].astype(str)
    df_grafo_1 = lista_df_raw[i].merge(db_macchine, how = 'left', left_on='Source', right_on='Source')
    df_grafo_2 = df_grafo_1.merge(db_macchine, how = 'left', left_on='Target', right_on='Source')
    df_grafo_2 = df_grafo_2[['Risorsa_x','Risorsa_y']]
    H = nx.from_pandas_edgelist(df_grafo_2, source = 'Risorsa_x', target = 'Risorsa_y')  
    globals()[f'H_{i}'] = H

st.header ('Scenari', divider='violet')

for i in range (n):
    st.subheader (f'Scenario {i+1}')
    st.write('Summary linea')
    st.dataframe(lista_df_raw[i])
    st.write ('Abbinamenti')
    st.dataframe(lista_abbinamenti[i])
    st.write('Grafo linea')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    nx.draw_kamada_kawai(globals()[f'H_{i}'], with_labels=True,  node_color='#8772B3', node_size=400, font_size=6, edge_color='white',
                     width=0.5, font_color='white')
    ax.set_facecolor('black')
    ax.axis('off')
    fig.set_facecolor('black')
    st.pyplot(fig)





