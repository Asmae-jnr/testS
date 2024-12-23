# Import the required packages
import streamlit as st
import pandas as pd
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Iris Classification",
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.themes.enable("dark")

# Load data
df = pd.read_csv('iris.csv', delimiter=',')

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar Navigation
with st.sidebar:
    st.title('Iris Classification')
    st.subheader("Navigation")

    # Page Button Navigation
    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        pass
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        pass
    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        pass
    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        pass
    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)):
        pass
    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)):
        pass
    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        pass

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of a training two classification models using the Iris flower dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")
    st.markdown("by: [Zeraphim](https://jcdiamante.com)")

# Display Content Based on Page Selection
if st.session_state.page_selection == 'about':
    st.title("About")
    st.write("Cette page contient une introduction au projet d'exploration et de classification de données.")

elif st.session_state.page_selection == 'dataset':
    st.title("Dataset")
    st.subheader("Description des données")
    st.write(df.head())

elif st.session_state.page_selection == 'eda':
    st.title("Exploration des Données (EDA)")
    st.subheader("Analyse exploratoire")
    chart = alt.Chart(df).mark_point().encode(
        x='petal_length',
        y='petal_width',
        color="species"
    )
    st.altair_chart(chart, use_container_width=True)

elif st.session_state.page_selection == 'data_cleaning':
    st.title("Data Cleaning / Pre-processing")
    st.write("Nettoyage et préparation des données pour le Machine Learning.")

elif st.session_state.page_selection == 'machine_learning':
    st.title("Machine Learning")
    st.write("Construction et entraînement des modèles de classification.")

elif st.session_state.page_selection == 'prediction':
    st.title("Prediction")
    st.write("Tester des prédictions sur le modèle entraîné.")

elif st.session_state.page_selection == 'conclusion':
    st.title("Conclusion")
    st.write("Résumé des résultats et prochaines étapes.")