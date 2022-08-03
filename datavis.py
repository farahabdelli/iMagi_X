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
    st.write("---")
    c1, c2 = st.columns((3, 2))
    with c2:
        st.write("##")
        st.write("##")
        st.image("images/magi.png")



    with c1:
        st.write("""
# Comment peut-on prédire l'usage d'une piscine Magiline grâce à la data ?
""")
        st.write("##")
        st.markdown(
            '<p class="intro">Afin de valider ma thèse professionnelle, j''ai choisi de travailler sur un sujet bien particulier qui est l''étude des comportements des différents équipements d''une piscine Magiline à travers les différents capteurs existants. Cette étude permet d''identifier quels sont les paramètres permettant de prédire la présence dans le bassin.</p>',
            unsafe_allow_html=True)

        st.markdown(
            '<p class="intro">En vu de réaliser tout ce qui précède, j''ai mis en place une infrastructure Big Data permettant à la fois de collecter, stocker, analyser et enfin, visualiser les données traitées. Accompagnée d''un traitement en temps réel des flux de données générées par la solution domotique i-MAGI-X et de plusieurs algorithmes d''apprentissage automatiques. Ce traitement est basé sur l''analyses des sondes (pH, Redox), des pompes (filtration, chauffage) et des différents équipements en options tels que la nage à contre courant , les robots de nettoyages et les dispositifs de sécurité ,etc.</p>',
            unsafe_allow_html=True)

        st.markdown(
            '<p class="intro">Cette application WEB est la dernière étape du processus. Il s''agit de mettre le modèle résultant en production. L''objectif est de mettre les connaissances acquises grâce à la modélisation dans le processus de prise de décision sous une forme appropriée.</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p class="intro">Pour déployer ma solution, on sert des outils suivants :</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p class="intro">- Github : il stocke le code de l’application, le code de la modélisation, le modèle enregistré, les données et un fichier requirements.txt qui contient toutes les librairies dont l’application a besoin pour fonctionner </p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p class="intro">- Streamlit Cloud : il construit et déploie l’application web à partir du code stocké sur Github et héberge la solution sur son serveur.</p>',
            unsafe_allow_html=True)
        
        st.markdown(
            '<p class="intro"><b>Passez à la page suivante pour voir les résultats. Bonne lecture !</b></p>',
            unsafe_allow_html=True)
    c1, _, c2, _, _, _ = st.columns(6)
    with c1:
        st.subheader("Pour visiter le site officiel :")

        st.write("• [Le site](https://www.piscines-magiline.fr)")
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
        st.write("""# Random Forest""")
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

    elif st.session_state.choix_page_classification == "SVM":
        st.write("""# Support Vector Machine""")
        st.write("##")
        exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
        with exp2:
            with st.expander("Principe de l'algorithme SVM"):
                st.write("""
                * 1ère étape : Choix du nombre de voisins k
                * 2ème étape : Calcul de la distance entre le point non classifié et tous les autre
                * 3ème étape : Sélection des k plus proches voisins
                * 4ème étape : On compte le nombre de voisins dans chaque classe
                * 5ème étape : Attribution de la classe la plus présente à notre point 
                """)

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





