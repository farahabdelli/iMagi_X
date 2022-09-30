
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

"""# Piscine Dominique SCHORR"""

qry0 = "SELECT * FROM box_stats where box_id = 9"
df = read_query(conn,qry0)
#df.iloc[170780:]
df
df0 =read_query(conn,qry0)

fig = px.line(df0.iloc[169491:], x="date", y="value", color="type" )

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
#baignades
shape_dict1= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-06-01 16:00:00', 'x1':'2022-06-01 17:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict2= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-06-02 16:00:00', 'x1':'2022-06-02 17:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict3= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-06-04 10:00:00', 'x1':'2022-06-04 11:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict4= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-06-05 09:30:00', 'x1':'2022-06-05 10:30:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict5= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-06-06 11:00:00', 'x1':'2022-06-06 12:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict6= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-06-11 10:00:00', 'x1':'2022-06-11 11:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict7= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-06-12 15:30:00', 'x1':'2022-06-12 16:30:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict8= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-06-14 16:00:00', 'x1':'2022-06-14 17:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict9= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-07-01 16:00:00', 'x1':'2022-07-01 17:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict10= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-07-09 16:00:00', 'x1':'2022-07-09 17:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict11= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-07-10 11:30:00', 'x1':'2022-07-10 12:30:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict12= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-07-11 16:00:00', 'x1':'2022-07-11 17:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict13= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-07-11 19:00:00', 'x1':'2022-07-11 19:30:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict14= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-07-14 11:30:00', 'x1':'2022-07-14 12:30:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict15= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-07-14 15:00:00', 'x1':'2022-07-14 17:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict16= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-07-15 15:00:00', 'x1':'2022-07-15 19:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict17= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-07-17 10:00:00', 'x1':'2022-07-17 13:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict18= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-07-17 16:00:00', 'x1':'2022-07-17 18:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict19= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-07-18 12:00:00', 'x1':'2022-07-18 13:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict20= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-07-18 17:30:00', 'x1':'2022-07-18 18:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}
shape_dict21= {'type':'rect', 'xref':'x', 'yref':'paper', 'x0':'2022-07-19 14:00:00', 'x1':'2022-07-19 19:00:00', 'y0':0, 'y1':1, 'fillcolor': 'blue', 'layer': 'below', 'opacity': 0.75, 'line_width': 0}


#fig.show()
fig.update_layout(shapes=[shape_dict1,shape_dict2,shape_dict3,shape_dict4,shape_dict5,shape_dict6,shape_dict7,shape_dict8,shape_dict9,shape_dict10,shape_dict11,shape_dict12,shape_dict13,shape_dict14,shape_dict15,shape_dict16,shape_dict17,shape_dict18,shape_dict19,shape_dict20,shape_dict21])

fig.show()

fig = px.line(df0, x="date", y="value", color="type" )

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

# Type of measurement
qry3 = "Select distinct type from box_stats where box_id=9" 
read_query(conn,qry3)

qr_airTemp = "Select date , value as airTemp from box_stats where box_id=9 and type='airTemp' and  `date` >= '2022-06-01 07:00:00' AND `date` <= '2022-07-20 23:00:00'" 
df_airTemp=read_query(conn,qr_airTemp)
df_airTemp=df_airTemp.set_index('date')

qr_waterTemp = "Select date , value as waterTemp from box_stats where box_id=9 and type='waterTemp' and  `date` >= '2022-06-01 07:00:00' AND `date` <= '2022-07-20 23:00:00'" 
df_waterTemp=read_query(conn,qr_waterTemp)
df_waterTemp=df_waterTemp.set_index('date')

qr_ph = "Select date , value as ph from box_stats where box_id=9 and type='ph' and  `date` >= '2022-06-01 07:00:00' AND `date` <= '2022-07-20 23:00:00'" 
df_ph=read_query(conn,qr_ph)
df_ph=df_ph.set_index('date')

qr_treatmentPh = "Select date , value as treatmentPh from box_stats where box_id=9 and type='treatmentPh' and  `date` >= '2022-06-01 07:00:00' AND `date` <= '2022-07-20 23:00:00'" 
df_treatmentPh=read_query(conn,qr_treatmentPh)
df_treatmentPh=df_treatmentPh.set_index('date')

qr_orp = "Select date , value as orp from box_stats where box_id=9 and type='orp' and  `date` >= '2022-06-01 07:00:00' AND `date` <= '2022-07-20 23:00:00'" 
df_orp=read_query(conn,qr_orp)
df_orp=df_orp.set_index('date')

#contacténation des tables
df_box_stats = pandas.concat([df_airTemp,df_waterTemp,df_ph,df_orp,df_treatmentPh],axis=1,join='outer')
df_box_stats['treatmentPh'] =df_box_stats['treatmentPh'].fillna(0)
df_box_stats = df_box_stats.reset_index()
#df_box_stats.to_csv('box_stat_9.csv',sep=';',index=None)
df_box_stats

event_qry = "SELECT * FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC " 
read_query(conn,event_qry)

#type events :
event_qry2 = "SELECT distinct idLabel FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC " 
read_query(conn,event_qry2)

qr_1 = "SELECT date,state,stateLabel as CE_trait_PH  FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état du traitement PH' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label1=read_query(conn,qr_1)
df_label1=df_label1.set_index('date')

qr_2 = "SELECT date,state,stateLabel as CE_chauffage FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement d’état du système de chauffage' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label2=read_query(conn,qr_2)
df_label2=df_label2.set_index('date')

qr_3 = "SELECT date,state,stateLabel as CE_NCC FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement d’état de la NCC' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label3=read_query(conn,qr_3)
df_label3=df_label3.set_index('date')

qr_4 = "SELECT date,state,stateLabel as CE_filtration FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement d’état de la filtration' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label4=read_query(conn,qr_4)
df_label4=df_label4.set_index('date')

"""qr_5= "SELECT date,stateLabel as CE_PAC `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état de la commande de la PAC' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label5=read_query(conn,qr_5)
df_label5=df_label5.set_index('date')"""


qr_6 = "SELECT date,state,stateLabel as CE_volet FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état du volet' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label6=read_query(conn,qr_6)
df_label6=df_label6.set_index('date')

qr_7 = "SELECT date,state,stateLabel as CE_hivernage FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état de hivernage' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label7=read_query(conn,qr_7)
df_label7=df_label7.set_index('date')

qr_8 = "SELECT date,state,stateLabel as CE_electrolyseur FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état électrolyseur' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label8=read_query(conn,qr_8)
df_label8=df_label8.set_index('date')

qr_9 = "SELECT date,state,stateLabel as CE_spot FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état des spots' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label9=read_query(conn,qr_9)
df_label9=df_label9.set_index('date')

qr_10 = "SELECT date,state,stateLabel as CE_pompe_chauff FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement d’état de la commande de pompe de chauffage' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label10=read_query(conn,qr_10)
df_label10=df_label10.set_index('date')

qr_11 = "SELECT date,state,stateLabel as CE_acc1 FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état accessoire 1' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label11=read_query(conn,qr_11)
df_label11=df_label11.set_index('date')


qr_12 = "SELECT date,state,stateLabel as CE_acc2 FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état accessoire 2' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label12=read_query(conn,qr_12)
df_label12=df_label12.set_index('date')

qr_13 = "SELECT date,state,stateLabel as CE_acc3 FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état accessoire 3' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label13=read_query(conn,qr_13)
df_label13=df_label13.set_index('date')


"""qr_14 = "SELECT date, idLabel,stateLabel FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = '...' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label14=read_query(conn,qr_14)
df_label14=df_label14.set_index('date')

qr_15 = "SELECT date, idLabel,stateLabel FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `boxId` = 'cd65cef9c6104fc8bb6037c3dada89a4' AND idLabel = '... ' AND `date` >= '2022-06-01 00:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label15=read_query(conn,qr_15)
df_label15=df_label15.set_index('date')
"""



df_events = pandas.concat([df_label1,df_label2,df_label3,df_label4,df_label6  ,df_label8 ,df_label9 ,df_label10 ],axis=1,join='outer')
"""df_label5,df_label7,df_label11 ,df_label12 ,df_label13"""
#df_events['treatmentPh'] =df_events['treatmentPh'].fillna(0)
df_events = df_events.reset_index()
#df_events.to_csv('df_events_9.csv',sep=';',index=None)
df_events #6137013

qr_airTemp = "Select date , value as airTemp from box_stats where box_id=9 and type='airTemp' and date >= '2022-06-02 07:00:00' AND date <= '2022-07-20 07:00:00'" 
df_airTemp=read_query(conn,qr_airTemp)
df_airTemp = pandas.DataFrame(df_airTemp.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_airTemp = df_airTemp.fillna(method='ffill')
df_airTemp=df_airTemp.set_index('date')

qr_waterTemp = "Select date , value as waterTemp from box_stats where box_id=9 and type='waterTemp' and date >= '2022-06-02 07:00:00' AND date <= '2022-07-20 07:00:00'" 
df_waterTemp=read_query(conn,qr_waterTemp)
df_waterTemp = pandas.DataFrame(df_waterTemp.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_waterTemp = df_waterTemp.fillna(method='ffill')
df_waterTemp=df_waterTemp.set_index('date')

df_waterTemp

"""# summer

one hot encoding
"""

qr_1 = "SELECT date,stateLabel as CE_trait_PH  FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état du traitement PH' AND `date` >= '2022-06-02 07:00:00' AND `date` <= '2022-07-20 07:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label1=read_query(conn,qr_1)
df_label1 = pandas.DataFrame(df_label1.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label1 = df_label1.fillna(method='ffill')
df_label1 = pandas.get_dummies(df_label1, columns = ['CE_trait_PH'])
df_label1=df_label1.set_index('date')

qr_2 = "SELECT date,stateLabel as CE_chauffage FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement d’état du système de chauffage' AND `date` >= '2022-06-02 07:00:00' AND `date` <= '2022-07-20 07:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label2=read_query(conn,qr_2)
df_label2 = pandas.DataFrame(df_label2.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label2 = df_label2.fillna(method='ffill')
df_label2 = pandas.get_dummies(df_label2, columns = ['CE_chauffage'])
df_label2=df_label2.set_index('date')


qr_3 = "SELECT date,stateLabel as CE_NCC FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement d’état de la NCC' AND `date` >= '2022-06-02 07:00:00' AND `date` <= '2022-07-20 07:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label3=read_query(conn,qr_3)
df_label3 = pandas.DataFrame(df_label3.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label3 = df_label3.fillna(method='ffill')
df_label3 = pandas.get_dummies(df_label3, columns = ['CE_NCC'])
df_label3=df_label3.set_index('date')

qr_4 = "SELECT date,stateLabel as CE_filtration FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement d’état de la filtration' AND `date` >= '2022-06-02 07:00:00' AND `date` <= '2022-07-20 07:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label4=read_query(conn,qr_4)
df_label4 = pandas.DataFrame(df_label4.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label4 = df_label4.fillna(method='ffill')
df_label4 = pandas.get_dummies(df_label4, columns = ['CE_filtration'])
df_label4=df_label4.set_index('date')


qr_6 = "SELECT date,stateLabel as CE_volet FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état du volet' AND `date` >= '2022-06-02 07:00:00' AND `date` <= '2022-07-20 07:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label6=read_query(conn,qr_6)
df_label6 = pandas.DataFrame(df_label6.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label6 = df_label6.fillna(method='ffill')
df_label6 = pandas.get_dummies(df_label6, columns = ['CE_volet'])
df_label6=df_label6.set_index('date')


qr_8 = "SELECT date,stateLabel as CE_electrolyseur FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état électrolyseur' AND `date` >= '2022-06-02 07:00:00' AND `date` <= '2022-07-20 07:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label8=read_query(conn,qr_8)
df_label8 = pandas.DataFrame(df_label8.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label8 = df_label8.fillna(method='ffill')
df_label8 = pandas.get_dummies(df_label8, columns = ['CE_electrolyseur'])
df_label8=df_label8.set_index('date')

qr_9 = "SELECT date,stateLabel as CE_spot FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état des spots' AND `date` >= '2022-06-02 07:00:00' AND `date` <= '2022-07-20 07:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label9=read_query(conn,qr_9)
df_label9 = pandas.DataFrame(df_label9.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label9 = df_label9.fillna(method='ffill')
df_label9 = pandas.get_dummies(df_label9, columns = ['CE_spot'])
df_label9=df_label9.set_index('date')

qr_10 = "SELECT date,stateLabel as CE_pompe_chauff FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement d’état de la commande de pompe de chauffage' AND `date` >= '2022-06-02 07:00:00' AND `date` <= '2022-07-20 07:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label10=read_query(conn,qr_10)
df_label10 = pandas.DataFrame(df_label10.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label10 = df_label10.fillna(method='ffill')
df_label10 = pandas.get_dummies(df_label10, columns = ['CE_pompe_chauff'])
df_label10=df_label10.set_index('date')

df_events = pandas.concat([df_label1,df_label2,df_label3,df_label4,df_label6  ,df_label8  ,df_label10 ],axis=1,join='outer')

df_events #6137013

df_events.drop('CE_chauffage_-7 Arrêt filtration off different de AUTO', inplace=True, axis=1)
df_events.drop('CE_electrolyseur_-3 Arrêt pas de pompe allumée', inplace=True, axis=1)
df_events.drop('CE_trait_PH_-3 Arrêt pas de pompe allumée', inplace=True, axis=1)
df_events['CE_filtration_5 Marche hivernage'] = df_events['CE_volet_-1 Position inconnue']
df_events['CE_volet_1 position fermé'] = df_events['CE_volet_-1 Position inconnue']
df_events['CE_hivernage_2 Actif activé'] = df_events['CE_volet_-1 Position inconnue']
df_events['CE_hivernage_0 Arret'] = df_events['CE_volet_-1 Position inconnue']
df_events.rename(columns={"CE_volet_-1 Position inconnue": "CE_volet_0 Position ouvert"},inplace=True)
df_events

df_test = pandas.concat([df_events,df_waterTemp,df_airTemp ],axis=1,join='outer')
df_test

df_test.isna().sum()

df_test=df_test.set_index('date')
df_test = df_test.sort_index(axis=1)
df_test = df_test.reset_index()

df_test.to_csv('data/features_summer.csv',sep=';',index=False)

"""# winter"""

qr_airTemp = "Select date , value as airTemp from box_stats where box_id=9 and type='airTemp' and `date` >= '2021-12-01 09:00:00' and `date` <= '2022-03-18 05:00:00'" 
df_airTemp=read_query(conn,qr_airTemp)
df_airTemp = pandas.DataFrame(df_airTemp.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_airTemp = df_airTemp.fillna(method='ffill')
df_airTemp=df_airTemp.set_index('date')

qr_waterTemp = "Select date , value as waterTemp from box_stats where box_id=9 and type='waterTemp' and `date` >= '2021-12-01 09:00:00' and `date` <= '2022-03-18 05:00:00'" 
df_waterTemp=read_query(conn,qr_waterTemp)
df_waterTemp = pandas.DataFrame(df_waterTemp.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_waterTemp = df_waterTemp.fillna(method='ffill')
df_waterTemp=df_waterTemp.set_index('date')

df_airTemp

qr_1 = "SELECT date,stateLabel as CE_trait_PH  FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état du traitement PH' AND `date` >= '2021-12-01 07:00:00' and `date` <= '2022-03-18 05:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label1=read_query(conn,qr_1)
df_label1 = pandas.DataFrame(df_label1.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label1 = df_label1.fillna(method='ffill')
df_label1 = pandas.get_dummies(df_label1, columns = ['CE_trait_PH'])
df_label1=df_label1.set_index('date')

qr_2 = "SELECT date,stateLabel as CE_chauffage FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement d’état du système de chauffage' AND `date` >= '2021-12-01 07:00:00' and `date` <= '2022-03-18 05:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label2=read_query(conn,qr_2)
df_label2 = pandas.DataFrame(df_label2.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label2 = df_label2.fillna(method='ffill')
df_label2 = pandas.get_dummies(df_label2, columns = ['CE_chauffage'])
df_label2=df_label2.set_index('date')


qr_3 = "SELECT date,stateLabel as CE_NCC FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement d’état de la NCC' AND `date` >= '2021-12-01 07:00:00' and `date` <= '2022-03-18 05:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label3=read_query(conn,qr_3)
df_label3 = pandas.DataFrame(df_label3.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label3 = df_label3.fillna(method='ffill')
df_label3 = pandas.get_dummies(df_label3, columns = ['CE_NCC'])
df_label3=df_label3.set_index('date')

qr_4 = "SELECT date,stateLabel as CE_filtration FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement d’état de la filtration' AND `date` >= '2021-12-01 07:00:00' and `date` <= '2022-03-18 05:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label4=read_query(conn,qr_4)
df_label4 = pandas.DataFrame(df_label4.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label4 = df_label4.fillna(method='ffill')
df_label4 = pandas.get_dummies(df_label4, columns = ['CE_filtration'])
df_label4=df_label4.set_index('date')

qr_5 = "SELECT date,stateLabel as CE_hivernage FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état de hivernage' AND `date` >= '2021-12-01 07:00:00' and `date` <= '2022-03-18 05:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label5=read_query(conn,qr_5)
df_label5 = pandas.DataFrame(df_label5.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label5 = df_label5.fillna(method='ffill')
df_label5 = pandas.get_dummies(df_label5, columns = ['CE_hivernage'])
df_label5=df_label5.set_index('date')

qr_6 = "SELECT date,stateLabel as CE_volet FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état du volet' AND `date` >= '2021-12-01 07:00:00' and `date` <= '2022-03-18 05:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label6=read_query(conn,qr_6)
df_label6 = pandas.DataFrame(df_label6.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label6 = df_label6.fillna(method='ffill')
df_label6 = pandas.get_dummies(df_label6, columns = ['CE_volet'])
df_label6=df_label6.set_index('date')


qr_8 = "SELECT date,stateLabel as CE_electrolyseur FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement état électrolyseur' AND `date` >= '2021-12-01 07:00:00' and `date` <= '2022-03-18 05:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label8=read_query(conn,qr_8)
df_label8 = pandas.DataFrame(df_label8.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label8 = df_label8.fillna(method='ffill')
df_label8 = pandas.get_dummies(df_label8, columns = ['CE_electrolyseur'])
df_label8=df_label8.set_index('date')


qr_10 = "SELECT date,stateLabel as CE_pompe_chauff FROM `box_events` LEFT JOIN event_types ON event_types.state = box_events.paramNew WHERE `box_id` = 9 AND idLabel = 'Changement d’état de la commande de pompe de chauffage' AND `date` >= '2021-12-01 07:00:00' and `date` <= '2022-03-18 05:00:00' and event_types.id = box_events.eventId ORDER BY `box_events`.`date` ASC" 
df_label10=read_query(conn,qr_10)
df_label10 = pandas.DataFrame(df_label10.groupby([pandas.Grouper(key='date', freq='2Min')]).max()).reset_index()
df_label10 = df_label10.fillna(method='ffill')
df_label10 = pandas.get_dummies(df_label10, columns = ['CE_pompe_chauff'])
df_label10=df_label10.set_index('date')


df_events_winter = pandas.concat([df_label1,df_label2,df_label3,df_label4,df_label5,df_label6  ,df_label8  ,df_label10 ],axis=1,join='outer')

#drop unnecessary columns
df_events_winter.drop('CE_trait_PH_-3 Arrêt pas de pompe allumée', inplace=True, axis=1)
df_events_winter.drop('CE_trait_PH_-6 Arrêt température trop basse', inplace=True, axis=1)
df_events_winter.drop('CE_chauffage_-6 Arrêt hivernage en cours', inplace=True, axis=1)
df_events_winter.drop('CE_NCC_-1 Arrêt securite', inplace=True, axis=1)
df_events_winter.drop('CE_NCC_-2 inconnu', inplace=True, axis=1)
df_events_winter.drop('CE_filtration_-1 Arrêt Sécurité', inplace=True, axis=1)
df_events_winter.drop('CE_filtration_2 Marche manuel ON', inplace=True, axis=1)
df_events_winter.drop('CE_hivernage_3 inconnu', inplace=True, axis=1)
df_events_winter.drop('CE_hivernage_4 inconnu', inplace=True, axis=1)
df_events_winter.drop('CE_volet_-1 Position inconnue', inplace=True, axis=1)
df_events_winter.drop('CE_electrolyseur_-1 Arrêt manuel', inplace=True, axis=1)
df_events_winter.drop('CE_pompe_chauff_-1 Arrêt Sécurité', inplace=True, axis=1)
df_events_winter.isna().sum()

df_events_winter

df_test_winter = pandas.concat([df_events_winter,df_waterTemp,df_airTemp ],axis=1,join='outer')
df_test_winter.isna().sum()

df_test_winter

#data to csv

df_test_winter = df_test_winter.sort_index(axis=1)
df_test_winter = df_test_winter.reset_index()
df_test_winter

df_test_winter.to_csv('data/features_winter.csv',sep=';',index=False)

"""# winter + summer"""

df_test_winter = df_test_winter.set_index('date')
df_test = df_test.set_index('date')

data = pandas.concat([df_test_winter,df_test],axis=0,ignore_index=False)
data = data.drop(data.columns[1],axis=1)
data['CE_NCC_1 Marche '] = data['CE_NCC_1 Marche '].fillna(0).astype(int)
data = data.sort_index(axis=1)
data = data.reset_index()
data

data.isna().sum()

data.to_csv("all_features.csv",sep=";",index=False)