
import pandas
import mysql.connector
import pandas 
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

# Connection à la base

conn = mysql.connector.connect(
    host="192.168.71.42",
    database="myImagiX",
    user="root",
    password="fskfes" )

def read_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute( query )
        names = [ x[0] for x in cursor.description]
        rows = cursor.fetchall()
        return pandas.DataFrame( rows, columns=names)
    finally:
        if cursor is not None:
            cursor.close()

"""# Piscine d'Amaury"""

qr = "SELECT date,state,stateLabel FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE box_events.box_id = 182 AND box_events.date >= '2022-05-01 00:00:00' and event_types.id = box_events.eventId AND idLabel LIKE 'Changement d’état de la filtration' ORDER BY `box_events`.`date` ASC "

dff=read_query(conn,qr)
dff.head(6)

#normalisation de la colonne state
def transform(x,a,b):
    
    return ((x - np.min(x))/(np.max(x)-np.min(x))) * (b-a) + a

dff['normalised_state']= transform(dff['state'],0,50)

dff.head(6)

qry0 = "SELECT * FROM `box_stats` WHERE type LIKE 'dirtying' and box_id= (SELECT id FROM boxes WHERE serialNumber LIKE '%37cf0') ORDER BY `box_stats`.`date` ASC "
df = read_query(conn,qry0)
df0 =read_query(conn,qry0)

fig = px.line(df0, x="date", y="value", color="type" )
fig.add_scatter(x=dff.date, y=dff['normalised_state'], mode='lines',name="filtration",marker_color='rgba(0, 0, 0, .9)')
#fig.add_scatter(x=df_dirtying.date, y=df_dirtying['threshold'], mode='lines',name="seuil",marker_color='rgba(255, 0, 0, .9)')
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
         
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True  
        ),
        type="date"
    )
) 

fig.show()

fig.write_html('encrassement.html')

from tracemalloc import take_snapshot
#define threshold

THRESHOLD = 10
qry0 = "SELECT * FROM `box_stats` WHERE type IN ('dirtying') and box_id= (SELECT id FROM boxes WHERE serialNumber LIKE '%37cf0') ORDER BY `box_stats`.`date` ASC "
df = read_query(conn,qry0)
df0 =read_query(conn,qry0)
df_dirtying = df.iloc[:,3:5]

df_dirtying['threshold'] = THRESHOLD
df_dirtying['event'] = df_dirtying.value < df_dirtying.threshold
df_dirtying['event'] = np.where(df_dirtying['event']==True, 1 ,0)
#df_dirtying = df_dirtying.set_index('date')
df_dirtying

import plotly.express as px


fig = px.bar(df_dirtying, x=df_dirtying.date, y="value")
fig.add_scatter(x=df_dirtying.date, y=df_dirtying['value'], mode='lines',name="values",marker_color='rgba(0, 0, 0, .9)')
fig.add_scatter(x=df_dirtying.date, y=df_dirtying['threshold'], mode='lines',name="seuil",marker_color='rgba(255, 0, 0, .9)')
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
         
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

fig.show()

from doctest import master
from email.errors import MissingHeaderBodySeparatorDefect
from logging import lastResort


df_dirtying_true = df_dirtying[df_dirtying.event == 1]
df_dirtying_true.head(5)

import plotly.express as px

fig = px.line(df_dirtying, x=df_dirtying.date, y=df_dirtying.value) 
fig.add_scatter(x=df_dirtying_true.date, y=df_dirtying_true.value, mode='markers',name="nettoyage de cartouche")
fig.add_scatter(x=df_dirtying.date, y=df_dirtying['threshold'], mode='lines',name="seuil",marker_color='rgba(20, 200, 0, .9)')
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
         
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

fig.show()

