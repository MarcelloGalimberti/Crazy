#!/usr/bin/env python
# coding: utf-8


# versione 2: aggiunta opzione di abbinare solo macchine uguali. la soluzione "macchine uguali" comparirà insieme alle top n e verrà confrontata
# con quelle più efficienti

# mostrare macchine negli abbinamenti

# totale macchine necessarie per tipo

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
    st.title('Organizarea liniei | Organizzazione linea')

uploaded_input = st.file_uploader("Încărcați fișierele de sincronizare și secvență | Carica file input")
if not uploaded_input:
    st.stop()
df_input=pd.read_excel(uploaded_input)

st.header('Fișier încărcat | File caricato', divider='violet')
st.dataframe(df_input, width=1500)

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
    df_saturazione_op_1=df_saturazione_op_1[['Op_1','Op_2','Saturazione','Macchina']]
    df_saturazione_op_1.rename(columns={'Saturazione': 'Saturazione_Op_1','Macchina':'Macchina_Op_1'}, inplace=True)
    
    df_saturazione_op_2 = df_saturazione_op_1.merge(df_raw, how = 'left', left_on=['Op_2'], right_on=['Source'])
    df_saturazione_op_2 = df_saturazione_op_2[['Op_1','Op_2','Saturazione_Op_1','Macchina_Op_1','Saturazione','Macchina']]
    df_saturazione_op_2.rename(columns={'Saturazione': 'Saturazione_Op_2','Macchina':'Macchina_Op_2'}, inplace=True)
    
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
    
    df_saturazioni_fattibili_omo =df_saturazioni_fattibili[(df_saturazioni_fattibili['Macchina_Op_1'])==(df_saturazioni_fattibili['Macchina_Op_2'])]
    df_saturazioni_fattibili_omo.reset_index(drop = True, inplace = True)
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
    
    abbinamenti_omo = pd.DataFrame(columns = df_saturazioni_fattibili_omo.columns)
    
    for i in range (len(df_saturazioni_fattibili_omo)):
        if df_saturazioni_fattibili_omo.loc[i,'Abbinabili'] == False:
            pass
        else:
            best_op_1 = df_saturazioni_fattibili_omo.loc[i,'Op_1']
            best_op_2 = df_saturazioni_fattibili_omo.loc[i,'Op_2']
            l = len(abbinamenti_omo)
            abbinamenti_omo.loc[l+1] = df_saturazioni_fattibili_omo.iloc[i,:]
            df_saturazioni_fattibili_omo['Abbinabili'] = ((df_saturazioni_fattibili_omo['Op_1']!=best_op_1) & 
                                                            (df_saturazioni_fattibili_omo['Op_2']!=best_op_2)
                                                   & (df_saturazioni_fattibili_omo['Op_1']!=best_op_2) & 
                                                            (df_saturazioni_fattibili_omo['Op_2']!=best_op_1) 
                                                 & (df_saturazioni_fattibili_omo['Abbinabili'] == True))
    abbinamenti_omo.drop(columns='Abbinabili', inplace=True)
    numero_minimo_op_omo = df_raw['TC'].sum()/takt
    numero_reale_op_omo = (df_raw['N_macchine'].sum()-len(abbinamenti_omo))
    metrica_omo = numero_minimo_op_omo/(df_raw['N_macchine'].sum()-len(abbinamenti_omo))
    
    
    
    return metrica, numero_minimo_op, abbinamenti, numero_reale_op, df_raw, metrica_omo, numero_minimo_op_omo, abbinamenti_omo, numero_reale_op_omo

# ### Calcolo performance

performance = pd.DataFrame(columns=['Takt Time','Metrica','Op_min','Op_reale','Capi-giorno'])
performance_omo = pd.DataFrame(columns=['Takt Time','Metrica','Op_min','Op_reale','Capi-giorno'])
for i in np.arange(0.2,4.1, 0.1):
    metrica_out, op_out , abbinamenti, numero_reale_op, df_raw, metrica_out_omo, op_out_omo, abbinamenti_omo, numero_reale_op_omo = processa_raw(df_input, i)
    performance.loc[len(performance)+1,'Takt Time'] = i
    performance.loc[len(performance),'Metrica'] = metrica_out
    performance.loc[len(performance),'Op_min'] = op_out
    performance.loc[len(performance),'Op_reale'] = numero_reale_op
    performance.loc[len(performance),'Capi-giorno'] = 450/i
    
    performance_omo.loc[len(performance_omo)+1,'Takt Time'] = i
    performance_omo.loc[len(performance_omo),'Metrica'] = metrica_out_omo
    performance_omo.loc[len(performance_omo),'Op_min'] = op_out_omo
    performance_omo.loc[len(performance_omo),'Op_reale'] = numero_reale_op_omo
    performance_omo.loc[len(performance_omo),'Capi-giorno'] = 450/i
    

max_op = st.number_input('Număr maxim de operatori | Numero massimo operatori', step =1)
if not max_op:
    st.stop()

min_op = st.number_input('Număr minim de operatori | Numero minimo operatori', step =1)
if not min_op:
    st.stop()

# #### Bar chart performance

performance_chart = performance[(performance['Op_reale'] <= max_op) & (performance['Op_reale'] >= min_op)]
fig_bar = px.bar(performance_chart, x='Takt Time', y='Metrica', color='Op_reale', text = 'Op_reale', hover_name='Capi-giorno',
                      template='plotly_dark')

performance_chart_omo = performance_omo[(performance_omo['Op_reale'] <= max_op) & (performance_omo['Op_reale'] >= min_op)]
fig_bar_omo = px.bar(performance_chart_omo, x='Takt Time', y='Metrica', color='Op_reale', text = 'Op_reale', hover_name='Capi-giorno',
                      template='plotly_dark')


# Calcolo performance filtrato

performance = performance[(performance['Op_reale'] <= max_op) & (performance['Op_reale'] >= min_op) ].astype('float')
performance.sort_values('Metrica', ascending=False, inplace=True)
performance.reset_index(drop = True, inplace = True)
performance.index.names = ['Soluzione']

performance_omo = performance_omo[(performance_omo['Op_reale'] <= max_op) & (performance_omo['Op_reale'] >= min_op) ].astype('float')
performance_omo.sort_values('Metrica', ascending=False, inplace=True)
performance_omo.reset_index(drop = True, inplace = True)
performance_omo.index.names = ['Soluzione']


st.subheader('Performanța liniei | Performance linea', divider='violet')
col_3, col_4 = st.columns([1, 1])
with col_3:
    st.write('Operatori combinați cu diferite mașini | Operatori abbinati a macchine diverse')
    st.dataframe(performance, width=750)

with col_4:
    st.write ('Performance vs Takt Time vs num. op.')
    st.plotly_chart(fig_bar,use_container_width=True)

#st.subheader('Performance omo', divider='violet')
col_7, col_8 = st.columns([1, 1])
with col_7:
    st.write ('Operatori combinați cu mașini identice | Operatori abbinati a macchine uguali')
    st.dataframe(performance_omo, width=750)

with col_8:
    st.write ('Performance vs Takt Time vs num. op.')
    st.plotly_chart(fig_bar_omo,use_container_width=True)

# sceglie performance o performance_omo

option = st.radio(
    'Alegeți tipul de configurație | Scegliere il tipo di configurazione',
    ('Operatori combinați cu diferite mașini | Operatori abbinati a macchine diverse', 'Operatori combinați cu mașini identice | Operatori abbinati a macchine uguali'))
if not option:
    st.stop()

#st.write('Hai selezionato: ', option)

if option == 'Operatori combinați cu diferite mașini | Operatori abbinati a macchine diverse':
    performance = performance
else:
    performance = performance_omo


# #### Scegliere la soluzione

st.subheader('Alegeți soluția de analizat | Scegliere la soluzione da analizzare')
n = st.number_input('Numărul soluției | Numero soluzione', step =1)

######## 

st.write (f'Caracteristicile soluției {n} | Caratteristiche della soluzione {n}')

performance = performance.loc[n]

st.write (performance)

st.header ('Scenariul propus cu soluția aleasă | Scenario proposto con la soluzione scelta', divider='violet')


metrica_out, op_out , abbinamenti, numero_reale_op, df_raw, metrica_out_omo, op_out_omo, abbinamenti_omo, numero_reale_op_omo  = processa_raw(df_raw, performance.loc['Takt Time'])
if option == 'Operatori combinați cu diferite mașini | Operatori abbinati a macchine diverse':
    metrica_out = metrica_out
    op_out = op_out
    abbinamenti = abbinamenti
    op_out_reale = numero_reale_op
else: 
    metrica_out = metrica_out_omo
    op_out = op_out_omo
    abbinamenti = abbinamenti_omo
    op_out_reale = numero_reale_op_omo
    
df_raw_out = df_raw

abbinamenti.reset_index(drop=True, inplace=True)

# costruisce tabella operatori step 1
df_operatori = pd.DataFrame(columns=['Fase','Macchina','Operatore','Saturazione'])
for i in range (len(abbinamenti)):
    df_operatori.loc[i,'Fase'] = abbinamenti.loc[i,'Op_1']
    df_operatori.loc[i,'Macchina'] = abbinamenti.loc[i,'Macchina_Op_1']
    df_operatori.loc[i,'Operatore'] = i+1
    df_operatori.loc[i,'Saturazione'] = abbinamenti.loc[i,'Saturazione_abbinamento']
k = len(abbinamenti)
for i in range (len(abbinamenti)):
    df_operatori.loc[i+k,'Fase'] = abbinamenti.loc[i,'Op_2']
    df_operatori.loc[i+k,'Macchina'] = abbinamenti.loc[i,'Macchina_Op_2']
    df_operatori.loc[i+k,'Operatore'] = i+1
    df_operatori.loc[i+k,'Saturazione'] = abbinamenti.loc[i,'Saturazione_abbinamento']


df_operatori.sort_values(by='Operatore',inplace=True)
df_operatori.reset_index(drop = True, inplace=True)

# costruisce tabella operatori step 2
lista_fasi = list(df_raw_out.Source)
df_2_macchine = df_raw_out[df_raw_out['N_macchine']==2]
lista_fasi_2_macchine = list(df_2_macchine['Source'])
for macchine in lista_fasi_2_macchine:
    lista_fasi.append(macchine)

lista_fasi_abbinate = list(df_operatori.Fase)
for i in range (len(lista_fasi_abbinate)):
    lista_fasi.remove(lista_fasi_abbinate[i])

for i in range (len(lista_fasi)):
    idx =  df_raw_out[df_raw_out['Source']==lista_fasi[i]].index.values
    df_operatori.loc[len(df_operatori)+1,'Fase'] = df_raw_out.at[idx[0],'Source']
    df_operatori.loc[len(df_operatori),'Macchina'] = df_raw_out.at[idx[0],'Macchina']
    df_operatori.loc[len(df_operatori),'Operatore'] = df_operatori.Operatore.max()+1
    df_operatori.loc[len(df_operatori),'Saturazione'] = df_raw_out.at[idx[0],'Saturazione']

df_op_fasi = df_operatori.merge(df_raw_out, how='left', left_on='Fase', right_on='Source')
df_op_fasi.drop(columns=['Source','Macchina_y','TC','Target','Risorse','N_macchine','Saturazione_y','Takt_Time'], inplace=True)
df_op_fasi.rename(columns={'Macchina_x':'Macchina','Saturazione_x':'Saturazione'}, inplace=True)

elenco_macchine = df_op_fasi.groupby(by = 'Macchina').agg('count')
elenco_macchine.drop(columns=['Fase','Operatore','Operazione'], inplace=True)
elenco_macchine.rename(columns={'Saturazione':'Numero macchine'}, inplace=True)

col_5, col_6 = st.columns([1, 1])

with col_5:
    st.write('Împerecherile operator/mașină | Abbinamenti operatore / macchina')
    st.dataframe (df_op_fasi)

with col_6:
    st.write('Mașini necesare | Macchine necessarie')
    st.dataframe(elenco_macchine)
    st.write ('Total mașini | Totale macchine', elenco_macchine['Numero macchine'].sum())

# ### Costruzione grafo

db_macchine = df_raw_out[['Source','Macchina','N_macchine']]
db_macchine.loc[len(db_macchine)] = [999,'CQ',1]
db_macchine['Risorsa'] = db_macchine['Source'].astype(str) + ' | ' + db_macchine['Macchina'] + ' | ' + db_macchine['N_macchine'].astype(str)
df_grafo_1 = df_raw_out.merge(db_macchine, how = 'left', left_on='Source', right_on='Source')
df_grafo_2 = df_grafo_1.merge(db_macchine, how = 'left', left_on='Target', right_on='Source')
df_grafo_2 = df_grafo_2[['Risorsa_x','Risorsa_y']]
H = nx.from_pandas_edgelist(df_grafo_2, source = 'Risorsa_x', target = 'Risorsa_y')  

st.header ('Grafic cu linii | Grafo linea', divider='violet')

st.write('Faza | mașină | numărul de mașini')
st.write('Fase | macchina | numero macchine')
fig, ax = plt.subplots(figsize=(12, 8), dpi=600)
nx.draw_kamada_kawai(H, with_labels=True,  node_color='#8772B3', node_size=300, font_size=6, edge_color='white',  #spring con random seed?
                     width=0.5, font_color='white', alpha = 0.8)
ax.set_facecolor('black')
ax.axis('off')
fig.set_facecolor('black')
st.pyplot(fig)





