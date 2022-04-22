import streamlit as st
import plotly.express as px
import numpy as np
import pickle
import time
from predict import predict_flower

st.title("Iris Species Predictor")
st.header("Let's predict Iris species")
st.subheader("Cool")

#save time because only loads data once
@st.cache()
def read_data():
    return px.data.iris()

show_df = st.checkbox('See data?')
#show_df

df_iris = px.data.iris()

hist = px.histogram(df_iris, x='sepal_length')
hist

if show_df:
    df_iris

col1, col2, col3 = st.columns(3)

with col1:
    sl = st.number_input("Sepal Length (cm)", 0.0, 10.0)

with col2:
    sw = st.number_input("Sepal Width (cm)", 0.0, 100.0)

with col3:
    pl = st.number_input("Petal Length (cm)", 0.0, 100.0)
    pw = st.number_input("Petal Width (cm)", 0.0, 100.0)



#sl = st.number_input("Sepal Length (cm)", 0, 10)
#sw = st.number_input("Sepal Width (cm)", 0, 100)
#pl = st.number_input("Petal Length (cm)", 0, 100)
#pw = st.number_input("Petal Width (cm)", 0, 100)

f"Sepal length is {sl}"

user_input = np.array([[sl, sw, pl, pw]])

with open("saved-iris-model.pkl", "rb") as f:
    classifier = pickle.load(f)

with st.spinner("predicting..."):
    time.sleep(1)
    #prediction = classifier.predict(user_input)
    prediction = predict_flower(classifier, user_input)

prediction

if prediction[0] == "setosa":
    st.image("https://static.streamlit.io/examples/dog.jpg")

#st.balloons()