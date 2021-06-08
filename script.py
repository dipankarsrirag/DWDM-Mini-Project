import streamlit as st
from multiapp import MultiApp
from apps import home, data_hand, modeling

app = MultiApp()

app.add_app("Home", home.app)
app.add_app("Data Handling", data_hand.app)    
app.add_app("Modelling", modeling.app)

app.run()