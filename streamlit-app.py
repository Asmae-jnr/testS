# Import the required packages
import streamlit as st
import pandas as pd
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Iris Classification", 
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

# -------------------------
# Sidebar

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:
    st.title('Iris Classification')

    # Page Button Navigation
    st.subheader("Pages")

    st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',))
    st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',))
    st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',))
    st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',))
    st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',))
    st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',))
    st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',))

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of a training two classification models using the Iris flower dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")
    st.markdown("by: [`Zeraphim`](https://jcdiamante.com)")

# -------------------------
# Page Content

# Load data
df = pd.read_csv('iris.csv', delimiter=',')

# Page Navigation
if st.session_state.page_selection == 'about':
    st.title('About the Iris Classification App')
    st.write("""
    This app demonstrates data exploration, preprocessing, and machine learning classification models
    using the Iris dataset.
    """)
    st.image('assets/icon/icon.png', caption='Iris Dataset Example', use_column_width=True)

elif st.session_state.page_selection == 'dataset':
    st.title('Dataset')
    st.write("### First Rows of the Dataset")
    st.write(df.head())
    st.write("### Dataset Description")
    st.write(df.describe())

elif st.session_state.page_selection == 'eda':
    st.title('Exploratory Data Analysis')
    st.subheader('Visual Exploration')
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
    st.write("### Preprocessing Steps")
    st.write("""
    - Handling missing values
    - Feature scaling
    - Encoding categorical variables
    """)

elif st.session_state.page_selection == 'machine_learning':
    st.title('Machine Learning')
    st.write("### Model Training and Evaluation")
    st.write("""
    - Train a classifier
    - Evaluate performance
    """)

elif st.session_state.page_selection == 'prediction':
    st.title('Prediction')
    st.write("### Make Predictions on New Data")
    user_input = st.text_input("Enter Petal and Sepal dimensions (comma-separated)", "")
    if user_input:
        try:
            dimensions = list(map(float, user_input.split(',')))
            st.write(f"Predicted Class for {dimensions}: Iris-setosa (Example)")
        except ValueError:
            st.warning("Invalid input. Please enter four numeric values separated by commas.")

elif st.session_state.page_selection == 'conclusion':
    st.title('Conclusion')
    st.write("### Summary of Insights")
    st.write("""
    - Model Accuracy
    - Key Observations
    """)

# Additional About Section
if st.button("About App"):
    st.subheader("App d'exploration des données des Iris")
    st.text("Construite avec Streamlit")
    st.text("Thanks to the Streamlit Team Amazing Work")

if st.checkbox("By"):
    st.text("Stéphane C. K. Tékouabou")
    st.text("ctekouaboukoumetio@gmail.com")
