# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 00:23:58 2021

@author: M Shoaib
"""



import math
import warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from wordcloud import WordCloud
from scipy import signal
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
terror_data = pd.read_csv('globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')

terror_data.head()
terror_data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','provstate':'state',
                       'region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed',
                       'nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type',
                       'weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
terror_data=terror_data[['Year','Month','Day','Country','state','Region','city','latitude','longitude','AttackType','Killed',
               'Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]

terror_data.head()
terror_data.isnull().sum()
print("Country with the most attacks:",terror_data['Country'].value_counts().idxmax())
print("City with the most attacks:",terror_data['city'].value_counts().index[1]) #as first entry is 'unknown'
print("Region with the most attacks:",terror_data['Region'].value_counts().idxmax())
print("Year with the most attacks:",terror_data['Year'].value_counts().idxmax())
print("Month with the most attacks:",terror_data['Month'].value_counts().idxmax())
print("Group with the most attacks:",terror_data['Group'].value_counts().index[1])
print("Most Attack Types:",terror_data['AttackType'].value_counts().idxmax())

cities = terror_data.state.dropna(False)
plt.subplots(figsize=(10,10))
wordcloud = WordCloud(background_color = 'white',
                     width = 512,
                     height = 384).generate(' '.join(cities))
plt.axis('off')
plt.imshow(wordcloud)
plt.show()

print(terror_data['Year'].value_counts(dropna = False).sort_index())
x_year = terror_data['Year'].unique()
y_count_years = terror_data['Year'].value_counts(dropna = False).sort_index()
plt.figure(figsize = (18,10))
sns.barplot(x = x_year,
           y = y_count_years,
           palette = 'rocket')
plt.xticks(rotation = 45)
plt.xlabel('Attack Year')
plt.ylabel('Number of Attacks each year')
plt.title('Attack_of_Years')
plt.show()

plt.subplots(figsize=(15,6))
sns.countplot('Year',data=terror_data,palette='RdYlGn_r',edgecolor=sns.color_palette("YlOrBr", 10))
plt.xticks(rotation=45)
plt.title('Number Of terror_dataist Activities Each Year')
plt.show()
pd.crosstab(terror_data.Year, terror_data.Region).plot(kind='area',figsize=(15,6))
plt.title('terror_dataist Activities by Region in each Year')
plt.ylabel('Number of Attacks')
plt.show()
'Values are sorted by the top 40 worst terror_data attacks as to keep the heatmap simple and easy to visualize'
terror_data1 = terror_data.sort_values(by='Killed',ascending=False)[:40]
heat=terror_data1.pivot_table(index='Country',columns='Year',values='Killed')
heat.fillna(0,inplace=True)

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
colorscale = [[0, '#edf8fb'], [.3, '#00BFFF'],  [.6, '#8856a7'],  [1, '#810f7c']]
heatmap = go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale=colorscale)
data = [heatmap]
layout = go.Layout(
    title='Top 40 Worst terror_data Attacks in History from 1982 to 2016',
    xaxis = dict(ticks='', nticks=20),
    yaxis = dict(ticks='')
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='heatmap',show_link=False)


terror_data.Country.value_counts()[:15]
'Top Countries affected by terror_data Attacks'
plt.subplots(figsize=(15,6))
sns.barplot(terror_data['Country'].value_counts()[:15].index,terror_data['Country'].value_counts()[:15].values,palette='Blues_d')
plt.title('Top Countries Affected')
plt.xlabel('Countries')
plt.ylabel('Count')
plt.xticks(rotation= 90)
plt.show()
'look at the terror_dataist acts in the world over a certain year.'

import folium
from folium.plugins import MarkerCluster
filterYear = terror_data['Year'] == 1970
filterData = terror_data[filterYear] # filter data
# filterData.info()
reqFilterData = filterData.loc[:,'city':'longitude'] #We are getting the required fields
reqFilterData = reqFilterData.dropna() # drop NaN values in latitude and longitude
reqFilterDataList = reqFilterData.values.tolist()
# reqFilterDataList

map = folium.Map(location = [0, 30], tiles='CartoDB positron', zoom_start=2)
# clustered marker
markerCluster = folium.plugins.MarkerCluster().add_to(map)
for point in range(0, len(reqFilterDataList)):
    folium.Marker(location=[reqFilterDataList[point][1],reqFilterDataList[point][2]],
                  popup = reqFilterDataList[point][0]).add_to(markerCluster)
print(map)
'''
84% of the terror_dataist attacks in 1970 were carried out on
 the American continent. In 1970, the Middle East and North
 Africa, currently the center of wars and terror_dataist attacks,
 faced only one terror_dataist attack.**
Now let us check out which terror_dataist organizations have carr
ied out their operations in each country. A value count would give us the terror_dataist organizations that have carried out the most attacks. we have indexed from 1 as to negate the value of
'''
terror_data.Group.value_counts()[1:15]
test = terror_data[terror_data.Group.isin(['Shining Path (SL)','Taliban','Islamic State of Iraq and the Levant (ISIL)'])]
test.Country.unique()
terror_data_df_group = terror_data.dropna(subset=['latitude','longitude'])
terror_data_df_group = terror_data_df_group.drop_duplicates(subset=['Country','Group'])

terror_dataist_groups = terror_data.Group.value_counts()[1:8].index.tolist()
terror_data_df_group = terror_data_df_group.loc[terror_data_df_group.Group.isin(terror_dataist_groups)]
print(terror_data_df_group.Group.unique())
map = folium.Map(location=[20, 0], tiles="CartoDB positron", zoom_start=2)
markerCluster = folium.plugins.MarkerCluster().add_to(map)
for i in range(0,len(terror_data_df_group)):
    folium.Marker([terror_data_df_group.iloc[i]['latitude'],terror_data_df_group.iloc[i]['longitude']], 
                  popup='Group:{}<br>Country:{}'.format(terror_data_df_group.iloc[i]['Group'], 
                  terror_data_df_group.iloc[i]['Country'])).add_to(map)
'''
The Above map looks untidy even though it can be zoomed in to view 
the Country in question. Hence in the next chart, I have used Folium
's Marker Cluster to cluster these icons. This makes it 
visually pleasing and highly interactive
'''
m1 = folium.Map(location=[20, 0], tiles="CartoDB positron", zoom_start=2)
marker_cluster = MarkerCluster(
    name='clustered icons',
    overlay=True,
    control=False,
    icon_create_function=None
)
for i in range(0,len(terror_data_df_group)):
    marker=folium.Marker([terror_data_df_group.iloc[i]['latitude'],terror_data_df_group.iloc[i]['longitude']]) 
    popup='Group:{}<br>Country:{}'.format(terror_data_df_group.iloc[i]['Group'],
                                          terror_data_df_group.iloc[i]['Country'])
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)
marker_cluster.add_to(m1)
folium.TileLayer('openstreetmap').add_to(m1)
folium.TileLayer('Stamen Terrain').add_to(m1)
folium.TileLayer('cartodbdark_matter').add_to(m1)
folium.TileLayer('stamentoner').add_to(m1)
folium.LayerControl().add_to(m1)
print(m1)
terror_data.head()
killData = terror_data.loc[:,'Killed']
print('Number of people killed by terror_data attack:', int(sum(killData.dropna())))# drop the NaN values

# Let's look at what types of attacks these deaths were made of.
attackData = terror_data.loc[:,'AttackType']
# attackData
typeKillData = pd.concat([attackData, killData], axis=1)

typeKillData.head()

typeKillFormatData = typeKillData.pivot_table(columns='AttackType', values='Killed', aggfunc='sum')
typeKillFormatData
labels = typeKillFormatData.columns.tolist() # convert line to list
transpoze = typeKillFormatData.T # transpoze
values = transpoze.values.tolist()
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(aspect="equal"))
plt.pie(values, startangle=90, autopct='%.2f%%')
plt.title('Types of terror_dataist attacks that cause deaths')
plt.legend(labels, loc='upper right', bbox_to_anchor = (1.3, 0.9), fontsize=15) # location legend
plt.show()
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=25
fig_size[1]=25
plt.rcParams["figure.figsize"] = fig_size
countryData = terror_data.loc[:,'Country']
# countyData
countryKillData = pd.concat([countryData, killData], axis=1)
countryKillFormatData = countryKillData.pivot_table(columns='Country', values='Killed', aggfunc='sum')
countryKillFormatData
labels = countryKillFormatData.columns.tolist()
labels = labels[:50] #50 bar provides nice view
index = np.arange(len(labels))
transpoze = countryKillFormatData.T
values = transpoze.values.tolist()
values = values[:50]
values = [int(i[0]) for i in values] # convert float to int
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'brown', 'black', 'gray', 'magenta', 'orange'] # color list for bar chart bar color 
fig, ax = plt.subplots(1, 1)
ax.yaxis.grid(True)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=25
fig_size[1]=25
plt.rcParams["figure.figsize"] = fig_size
plt.bar(index, values, color = colors, width = 0.9)
plt.ylabel('Killed People', fontsize=20)
plt.xlabel('Countries', fontsize = 20)
plt.xticks(index, labels, fontsize=18, rotation=90)
plt.title('Number of people killed by countries', fontsize = 20)
# print(fig_size)
plt.show()

labels = countryKillFormatData.columns.tolist()
labels = labels[50:101]
index = np.arange(len(labels))
transpoze = countryKillFormatData.T
values = transpoze.values.tolist()
values = values[50:101]
values = [int(i[0]) for i in values]
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'brown', 'black', 'gray', 'magenta', 'orange']
fig, ax = plt.subplots(1, 1)
ax.yaxis.grid(True)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=20
fig_size[1]=20
plt.rcParams["figure.figsize"] = fig_size
plt.bar(index, values, color = colors, width = 0.9)
plt.ylabel('Killed People', fontsize=20)
plt.xlabel('Countries', fontsize = 20)
plt.xticks(index, labels, fontsize=18, rotation=90)
plt.title('Number of people killed by countries', fontsize = 20)
plt.show()
labels = countryKillFormatData.columns.tolist()
labels = labels[152:206]
index = np.arange(len(labels))
transpoze = countryKillFormatData.T
values = transpoze.values.tolist()
values = values[152:206]
values = [int(i[0]) for i in values]
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'brown', 'black', 'gray', 'magenta', 'orange']
fig, ax = plt.subplots(1, 1)
ax.yaxis.grid(True)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=25
fig_size[1]=25
plt.rcParams["figure.figsize"] = fig_size
plt.bar(index, values, color = colors, width = 0.9)
plt.ylabel('Killed People', fontsize=20)
plt.xlabel('Countries', fontsize = 20)
plt.xticks(index, labels, fontsize=18, rotation=90)
plt.title('Number of people killed by countries', fontsize = 20)
plt.show()
'''

terror_dataist acts in the Middle East and northern Africa have been 
seen to have fatal consequences. The Middle East and North Africa
 are seen to be the places of serious terror_dataist attacks. In addition,
 even though there is a perception that Muslims are supporters of terror_dataism,
 Muslims are the people who are most damaged by terror_dataist attacks. If you look
 at the graphics, it appears that Iraq, Afghanistan and Pakistan are the most
 damaged countries. All of these countries are Muslim countries
 '''
 