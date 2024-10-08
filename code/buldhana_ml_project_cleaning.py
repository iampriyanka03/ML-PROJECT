# -*- coding: utf-8 -*-
"""BULDHANA ML PROJECT CLEANING.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wRzg0VMInIueZOh6xlyKoJJATa1NdJdO
"""

import pandas as pd
data=pd.read_csv("/content/FINAL BULDHANA1.csv")

data.info()

data.shape

data.isnull().sum()

data["Experimental weight"].unique()

data["Crop condition"].unique()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data['Crop condition'] = label_encoder.fit_transform(data['Crop condition'])

data['Any_Damage'] = label_encoder.fit_transform(data['Any_Damage'])

data['Weeds'] = label_encoder.fit_transform(data['Weeds'])

data["2fnJul22_FCover"].unique()

import numpy as np
data.replace('Nodata', np.nan, inplace=True)

data.isnull().sum()

data['2fnJul22_FCover'] = pd.to_numeric(data['2fnJul22_FCover'], errors='coerce')

data["2fnJul22_FCover"].mean()

data["2fnJul22_FCover"].fillna(24,inplace=True)

data['1fnAug22_FCover'].unique()

data['1fnAug22_FCover'] = pd.to_numeric(data['1fnAug22_FCover'], errors='coerce')

data['1fnAug22_FCover'].mean()

data['1fnAug22_FCover'].fillna(24,inplace=True)

data['1fnAug22_FCover'].unique()

data["2fnAug22_FCover"].unique()

data['2fnAug22_FCover'] = pd.to_numeric(data['2fnAug22_FCover'], errors='coerce')

data['2fnAug22_FCover'].mean()

data['2fnAug22_FCover'].unique()

data['2fnAug22_FCover'].fillna(155.96666666666667,inplace=True)

data['2fnAug22_FCover'].unique()

data["1fnSep22_FCover"].unique()

data['1fnSep22_FCover'] = pd.to_numeric(data['1fnSep22_FCover'], errors='coerce')

data["1fnSep22_FCover"].mean()

data["1fnSep22_FCover"].fillna(145.66666666666666,inplace=True)

data["1fnSep22_FCover"].unique()

data["2fnSep22_FCover"].unique()

data['2fnSep22_FCover'] = pd.to_numeric(data['2fnSep22_FCover'], errors='coerce')

data["2fnSep22_FCover"].mean()

data["2fnSep22_FCover"].fillna(129.04166666666666,inplace=True)

data["2fnSep22_FCover"].unique()

data['1fnOct22_FCover'].unique()

data['1fnOct22_FCover'] = pd.to_numeric(data['1fnOct22_FCover'], errors='coerce')

data["1fnOct22_FCover"].mean()

data["1fnOct22_FCover"].fillna(116.58333333333333,inplace=True)

data["2fnOct22_FCover"].unique()

data['2fnOct22_FCover'] = pd.to_numeric(data['2fnOct22_FCover'], errors='coerce')

data["2fnOct22_FCover"].mean()

data["2fnOct22_FCover"].fillna(104.21355932203389,inplace=True)

data.info()

data['FAPAR_2fnJuly'].unique()

data['FAPAR_1fnAug'].unique()

data['FAPAR_2fnAug'].unique()

data['FAPAR_1fnSept'].unique()

data['FAPAR_2fnSept'].unique()

data['FAPAR_1fnOct'].unique()

data['FAPAR_2fnOct'].unique()

data['FAPAR_2fnJuly'].mean()

data['FAPAR_1fnAug'].mean()

data['FAPAR_2fnAug'].mean()

data['FAPAR_1fnSept'].mean()

data['FAPAR_2fnSept'].mean()

data['FAPAR_1fnOct'].mean()

data['FAPAR_2fnOct'].mean()

data['FAPAR_2fnOct'] = pd.to_numeric(data['FAPAR_2fnOct'], errors='coerce')

data["FAPAR_2fnJuly"].fillna(0.13222222222222224,inplace=True)

data["FAPAR_1fnAug"].fillna(0.13444444444444448,inplace=True)

data["FAPAR_2fnAug"].fillna(0.6017073170731707,inplace=True)

data["FAPAR_1fnSept"].fillna(0.5829629629629629,inplace=True)

data["FAPAR_2fnSept"].fillna(0.5613157894736842,inplace=True)

data["FAPAR_1fnOct"].fillna(0.4936842105263157,inplace=True)

data["FAPAR_2fnOct"].fillna(0.4827397260273973,inplace=True)

data.info()

data['2fnJul22_DryMatter(Biomass)'].unique()

data['2fnJul22_DryMatter(Biomass)'].mean()

data["2fnJul22_DryMatter(Biomass)"].fillna(-0.02,inplace=True)

data['1fnAug22_DryMatter(Biomass)'].unique()

data['1fnAug22_DryMatter(Biomass)'].mean()

data["1fnAug22_DryMatter(Biomass)"].fillna(-0.02,inplace=True)

data['2fnAug22_DryMatter(Biomass)'].unique()

data['2fnAug22_DryMatter(Biomass)'].mean()

data["2fnAug22_DryMatter(Biomass)"].fillna(66.75435897435898,inplace=True)

data['1fnSep22_DryMatter(Biomass)'].unique()

data['1fnSep22_DryMatter(Biomass)'].mean()

data["1fnSep22_DryMatter(Biomass)"].fillna(50.66230769230769,inplace=True)

data['2fnSep22_DryMatter(Biomass)'].unique()

data['2fnSep22_DryMatter(Biomass)'].mean()

data["2fnSep22_DryMatter(Biomass)"].fillna(57.11972972972973,inplace=True)

data['1fnOct22_DryMatter(Biomass)'].unique()

data['1fnOct22_DryMatter(Biomass)'].mean()

data["1fnOct22_DryMatter(Biomass)"].fillna(56.05796874999999,inplace=True)

data['2fnOct22_DryMatter(Biomass)'].unique()

data['2fnOct22_DryMatter(Biomass)'].mean()

data["2fnOct22_DryMatter(Biomass)"].fillna(57.2091836734694,inplace=True)

data.to_csv("CLEANED_FINAL_BULDHANA.csv")

data.isnull().sum()

data.info()

import pandas as pd
data=pd.read_csv("/content/CLEANED_FINAL_BULDHANA.csv")
data.info()

one_hot_encoded = pd.get_dummies(data['Crop condition'], prefix='Crop condition')

df_encoded = pd.concat([data, one_hot_encoded], axis=1)

print(df_encoded)

one_hot_encoded = pd.get_dummies(data['Weeds'], prefix='Weeds')
df_encoded = pd.concat([data, one_hot_encoded], axis=1)
print(df_encoded)

data1 = {'Weeds_0': df_encoded['Weeds_0']}

df = pd.DataFrame(data1)
df.to_csv("weeds0.csv")

data2 = {'Weeds_1': df_encoded['Weeds_1']}
df = pd.DataFrame(data2)
df.to_csv("weeds1.csv")

data3 = {'Weeds_2': df_encoded['Weeds_2']}
df = pd.DataFrame(data3)
df.to_csv("weeds2.csv")

data4
= {'Any_Damage': df_encoded['Any_Damage']}
df = pd.DataFrame(data1)
df.to_csv("anydamage0.csv")

data2 = {'Weeds_1': df_encoded['Weeds_1']}
df = pd.DataFrame(data1)
df.to_csv("weeds1.csv")

data2 = {'Weeds_1': df_encoded['Weeds_1']}
df = pd.DataFrame(data1)
df.to_csv("weeds1.csv")

one_hot_encoded = pd.get_dummies(data['Any_Damage'], prefix='Any_Damage')

df_encoded = pd.concat([data, one_hot_encoded], axis=1)

print(df_encoded)

data.info()

data.head()

data["Crop condition"]

data.to_csv("CLEANED_FINAL_BULDHANA1.csv")

