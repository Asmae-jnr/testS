# Import the required packages
import streamlit as st
import pandas as pd
import altair as alt
import os

# Page configuration
st.set_page_config(
    page_title="Iris Classification", 
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Activer le thème sombre pour Altair
alt.themes.enable("dark")

# -------------------------
# Sidebar

# Initialiser la sélection de page dans session_state
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Page par défaut

# Fonction pour mettre à jour la sélection de page
def set_page_selection(page):
    st.session_state.page_selection = page

# Barre latérale
with st.sidebar:
    st.title('Iris Classification')

    # Navigation entre les pages
    st.subheader("Pages")
    st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',))
    st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',))
    st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',))
    st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',))
    st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',))
    st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',))
    st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',))

    # Détails du projet
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of two classification models using the Iris flower dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")
    st.markdown("by: [`Zeraphim`](https://jcdiamante.com)")

# -------------------------
# Page Content

# Charger les données
try:
    df = pd.read_csv('iris.csv', delimiter=',')
except FileNotFoundError:
    st.error("Le fichier 'iris.csv' est introuvable. Assurez-vous qu'il est présent dans le répertoire de l'application.")

# Navigation des pages
if st.session_state.page_selection == 'about':
    st.title('About the Iris Classification App')
    st.write("""
    This app demonstrates data exploration, preprocessing, and machine learning classification models
    using the Iris dataset.
    """)
    image_path = os.path.join('assets', 'icon', 'icon.png')
    if os.path.exists(image_path):
        st.image(image_path, caption='Iris Dataset Example', use_container_width=True)
    else:
        st.warning("L'image 'icon.png' est introuvable. Vérifiez le chemin du fichier.")

elif st.session_state.page_selection == 'dataset':
    st.title('Dataset')
    st.write("### Aperçu des premières lignes du Dataset")
    st.write(df.head())
    st.write("### Description du Dataset")
    st.write(df.describe())

elif st.session_state.page_selection == 'eda':
    st.title('Exploratory Data Analysis')
    st.subheader('Exploration Visuelle')

    chart = alt.Chart(df).mark_point().encode(
        x='petal_length',
        y='petal_width',
        color="species"
    )
    st.write(chart)
    
    chart2 = alt.Chart(df).mark_circle(size=60).encode(
        x='sepal_length',
        y='sepal_width',
        color='species',
        tooltip=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    ).interactive()
    st.write(chart2)

elif st.session_state.page_selection == 'data_cleaning':
    st.title('Data Cleaning / Pre-processing')
    st.write("### Étapes de Prétraitement")
    st.write("""
    - Gestion des valeurs manquantes  
    - Mise à l'échelle des caractéristiques  
    - Encodage des variables catégorielles  
    """)

elif st.session_state.page_selection == 'machine_learning':
    st.title('Machine Learning')
    st.write("### Entraînement et Évaluation du Modèle")
    st.write("""
    - Entraînement d'un classifieur  
    - Évaluation des performances du modèle  
    """)

elif st.session_state.page_selection == 'prediction':
    st.title('Prediction')
    st.write("### Prédictions sur de nouvelles données")
    user_input = st.text_input("Entrez les dimensions des pétales et sépales (séparées par des virgules)", "")
    if user_input:
        try:
            dimensions = list(map(float, user_input.split(',')))
            st.write(f"Classe prédite pour {dimensions}: Iris-setosa (Exemple)")
        except ValueError:
            st.warning("Entrée invalide. Veuillez entrer quatre valeurs numériques séparées par des virgules.")

elif st.session_state.page_selection == 'conclusion':
    st.title('Conclusion')
    st.write("### Récapitulatif des observations")
    st.write("""
    - Précision du modèle  
    - Principales observations  
    """)

# Section supplémentaire
if st.button("About App"):
    st.subheader("App d'exploration des données des Iris")
    st.text("Construite avec Streamlit")
    st.text("Merci à l'équipe Streamlit pour leur excellent travail")

if st.checkbox("By"):
    st.text("Stéphane C. K. Tékouabou")
    st.text("ctekouaboukoumetio@gmail.com")
