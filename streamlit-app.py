import streamlit as st
import pandas as pd
import altair as alt
import joblib
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Iris Classification",
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark mode for Altair charts
alt.themes.enable("dark")

# Load dataset
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
    st.markdown("A Streamlit dashboard highlighting the results of training two classification models using the Iris flower dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")
    st.markdown("by: [Zeraphim](https://jcdiamante.com)")

# Load your trained model (replace with your actual model path)
try:
    if os.path.exists('model.pkl'):  # Check if the model file exists
        model = joblib.load('model.pkl')
    else:
        st.error("Le fichier 'model.pkl' est introuvable. Assurez-vous qu'il est présent dans le répertoire du projet.")
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")

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
    st.write("Utilisez les curseurs pour tester des prédictions sur le modèle entraîné.")

    # Create sliders for each feature
    petal_length = st.slider("Petal Length (cm)", min_value=0.0, max_value=7.0, value=1.5, step=0.1)
    petal_width = st.slider("Petal Width (cm)", min_value=0.0, max_value=3.0, value=0.2, step=0.1)
    sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.0, step=0.1)
    sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.0, step=0.1)

    # Prepare the input data as a 2D array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    try:
        # Predict the class using the trained model
        prediction = model.predict(input_data)
        # Display the prediction
        st.write(f"Prediction: *{prediction[0]}*")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

elif st.session_state.page_selection == 'conclusion':
    st.title("Conclusion")
    st.write("Résumé des résultats et prochaines étapes.")