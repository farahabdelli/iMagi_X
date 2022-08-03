import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

PAGES = ["Accueil", "Prédiction"]

with st.sidebar:
    st_lottie(load_lottieurl('https://assets8.lottiefiles.com/packages/lf20_jjojhxyb.json'), height=150)
choix_page = st.sidebar.radio(label="", options=PAGES)


############# Page 1 #############
if choix_page == "Accueil":
    st.markdown('<p class="first_titre">Piscines Magiline </p>', unsafe_allow_html=True)
    st.write("---")
    c1, c2 = st.columns((3, 2))
    with c2:
        st.write("##")
        st.write("##")
        st.image("logo/background.png")
    st.write("##")
    with c1:
        st.write("##")
        st.markdown(
            '<p class="intro">Bienvenue sur la <b>no-code AI platform</b> ! Déposez vos datasets csv ou excel ou choisissez en un parmi ceux proposés et commencez votre analyse dès maintenant ! Cherchez les variables les plus intéressantes, visualisez vos données, et créez vos modèles de Machine Learning en toute simplicité.' +
            ' Si vous choisissez de travailler avec votre dataset et que vous voulez effectuez des modifications sur celui-ci, il faudra le télécharger une fois les modifications faites pour pouvoir l\'utiliser sur les autres pages. </p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p class="intro">Un tutoriel sur l\'utilisation de ce site est disponible sur le repo Github. En cas de bug ou d\'erreur veuillez m\'en informer par mail ou sur Discord.</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p class="intro"><b>Commencez par choisir un dataset dans la section Dataset !</b></p>',
            unsafe_allow_html=True)
    c1, _, c2, _, _, _ = st.columns(6)
    with c1:
        st.subheader("Liens")
        st.write(
            "• [Mon profil GitHub](https://github.com/antonin-lfv/Online_preprocessing_for_ML/blob/master/README.md)")
        st.write("• [Mon site](https://antonin-lfv.github.io)")
    with c2:
        lottie_accueil = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_xRmNN8.json')
        st_lottie(lottie_accueil, height=200)
#*******************************************

############# Classification #############



elif choix_page == "Prédiction":
    PAGES_Prédiction = ["RF", "RL", "SVM"]
    st.sidebar.title('Classifications  :brain:')
    st.sidebar.radio(label="", options=PAGES_Prédiction, key="choix_page_classification")

    if st.session_state.choix_page_classification == "RF":
        st.markdown('<p class="grand_titre">RF : Random Forest</p>', unsafe_allow_html=True)
        st.write("##")
        exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
        with exp2:
            with st.expander("Principe de l'algorithme RF"):
                st.write("""
                * 1ère étape : Choix du nombre de voisins k
                * 2ème étape : Calcul de la distance entre le point non classifié et tous les autre
                * 3ème étape : Sélection des k plus proches voisins
                * 4ème étape : On compte le nombre de voisins dans chaque classe
                * 5ème étape : Attribution de la classe la plus présente à notre point 
                """)


        if 'data' in st.session_state:
            st.sidebar.header('User Input Parameters')

            def user_input_features():
                sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
                sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
                petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
                petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
                data = {'sepal_length': sepal_length,
                        'sepal_width': sepal_width,
                        'petal_length': petal_length,
                        'petal_width': petal_width}
                features = pd.DataFrame(data, index=[0])
                return features   

            df = user_input_features()

            st.subheader('User Input parameters')
            st.write(df)

            iris = datasets.load_iris()
            X = iris.data
            Y = iris.target

            clf = RandomForestClassifier()
            clf.fit(X, Y)

            prediction = clf.predict(df)
            prediction_proba = clf.predict_proba(df)

            st.subheader('Class labels and their corresponding index number')
            st.write(iris.target_names)

            st.subheader('Prediction')
            st.write(iris.target_names[prediction])
            #st.write(prediction)

            st.subheader('Prediction Probability')
            st.write(prediction_proba)


   
############# Fin Classification #############



st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")



