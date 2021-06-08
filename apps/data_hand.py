import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle as pk

def app():
    st.header('Data Loading')
    df = []
    uploaded_file = st.file_uploader("Upload Files", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name,
                        "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)
        df = pd.read_csv(uploaded_file)
        st.subheader('Data Frame')
        st.dataframe(df.head(10))

        st.subheader('Columns to be dropped')
        opt = st.multiselect('   ', tuple(df.columns))

        df.drop(opt, inplace=True, axis=1)

        cat_col = [i for i in df.columns if type(df[i].iloc[0]) == type('a')]
        with open('./pickles/cat.pk', 'wb') as f:
            pk.dump(cat_col, f)

        st.header('Data Pre-Processing')

        st.subheader('Missing Values')
        fig, ax = plt.subplots()
        ax = sns.heatmap(df.isna())
        st.pyplot(fig)

        st.subheader('Select columns to impute')
        options = st.multiselect(label='',
                                 options=list(df.columns))

        st.dataframe(df[options].head(10))

        if options:
            st.subheader('Choose the method to impute')
            imp = st.radio(
                '',
                ('Mean', 'Median',
                         'Custom Input - [For single column]')
            )
            if imp == 'Mean':
                opt = df[options].copy()
                for i, j in zip(list(np.mean(opt[options])), options):
                   opt[j].fillna(value=i, inplace=True)

            elif imp == 'Median':
                opt = df[options].copy()
                for i, j in zip(list(np.median(opt[options])), options):
                    opt[j].fillna(value=i, inplace=True)

            elif imp == 'Custom Input - [For single column]':
                opt = df[options].copy()
                st.subheader('Custom Input type')
                inp_type = st.radio(
                    '',
                    ('Text', 'Number')
                )

                if inp_type == 'Text':
                    val = st.text_input('', '')
                    opt.fillna(value=val, inplace=True)

                elif inp_type == 'Number':
                    val = st.text_input('')
                    opt.fillna(val, inplace=True)

            df[options] = opt
        with open('./pickles/imputed.pk', 'wb') as f:
            pk.dump(df, f)

        st.dataframe(df.head(10))

        st.subheader('Do you wish to encode categorical data')
        yes_enc = st.radio(
            '',
            ('Yes', 'No')
        )
        if yes_enc == 'Yes':
            with open('./pickles/imputed.pk', 'rb') as f:
                data = pk.load(f)
            data[cat_col] = data[cat_col].apply(LabelEncoder().fit_transform)

            st.subheader('Encoded Dataframe')
            df = data.copy()
            st.dataframe(df.head(10))
            with open('./pickles/encoded.pk', 'wb') as f:
                pk.dump(df, f)

        st.subheader('Do you wish to normalize data')
        yes_nor = st.radio(
            ' ',
            ('Yes', 'No')
        )
        with open('./pickles/encoded.pk', 'rb') as f:
            data = pk.load(f)
        if yes_nor == 'Yes':
            st.subheader('Choose the method to normalize')
            nor = st.radio(
                '',
                ('Min-Max Scaler', 'Standard Scaler')
            )
            non_cat_col = [i for i in df.columns if i not in cat_col]
            opt = st.multiselect('          ', tuple(non_cat_col))
            if nor == 'Min-Max Scaler':
                mmx = MinMaxScaler()
                for i in opt:
                    data[i] = mmx.fit_transform(
                        np.array(data[i]).reshape(data.shape[0], -1))

            elif nor == 'Standard Scaler':
                sx = StandardScaler()
                for i in opt:
                    data[i] = sx.fit_transform(
                        np.array(data[i]).reshape(data.shape[0], -1))

            with open('./pickles/scaled.pk', 'wb') as f:
                pk.dump(data, f)

            st.subheader('Scaled Dataframe')
            st.dataframe(data.head(10))
