import pandas as pd
import requests
import plotly.express as px
import os
import joblib
import plotly as plt
import matplotlib.pyplot as pyplt
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image

import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

#from prediction import train_test

def train_test(df):

    
    train = df.iloc[70000:]
    test =  df.iloc[106952:]
    
    return train, test


class_names = ['Absence', 'Présence']

def plot_metrics(metrics_list):
    if 'Confusion Matrix' in metrics_list:
        st.write('##')
        st.write('##')
        st.write('##')
        st.subheader(" Matrice de confusion")
        st.write('##')
        st.write('##')
        st.write('##')
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.write('##')
        st.write('##')
        st.write('##')
        st.subheader("ROC Curve")
        st.write('##')
        st.write('##')
        st.write('##')
        plot_roc_curve(model, x_test, y_test)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    

#paths
ABS_DATAPATH = os.path.abspath('data/')
Saved_model_DATAPATH = os.path.abspath('saved_models/')
PREP_DATA = 'all_features.csv' 
LABEL_DATA = 'descriptif_hiver_ete.csv' 
DESC_DATA = 'descriptif.csv' 
RLOGIST = 'reg_logist2.joblib'
RF = 'random_forest2.joblib'
Dtree = 'decision_tree2.joblib'

# load data
data_model = pd.read_csv(os.path.join(ABS_DATAPATH, PREP_DATA), sep=';')
target = pd.read_csv(os.path.join(ABS_DATAPATH, LABEL_DATA), sep=';') 
target_ete = pd.read_csv(os.path.join(ABS_DATAPATH, DESC_DATA), sep=';') 

####### html/css config ########
st.set_page_config(layout="wide", page_title="Magiline data prediction", menu_items={
    'About': "Magiline data prediction - réalisé par Farah"
})

st.markdown("""
<style>
.first_titre {
    font-size:75px !important;
    font-weight: bold;
    box-sizing: border-box;
    text-align: center;
    width: 100%;
}
.intro{
    text-align: justify;
    font-size:20px !important;
}
.grand_titre {
    font-size:30px !important;
    font-weight: bold;
    text-align: center;
    text-decoration: underline;
    text-decoration-color: #4976E4;
    text-decoration-thickness: 5px;
}
.section{
    font-size:20px !important;
    font-weight: bold;
    text-align: center;
    text-decoration: underline;
    text-decoration-color: #111111;
    text-decoration-thickness: 3px;
}
.petite_section{
    font-size:16px !important;
    font-weight: bold;
}
.nom_colonne_page3{
    font-size:17px !important;
    text-decoration: underline;
    text-decoration-color: #000;
    text-decoration-thickness: 1px;
}
</style>
""", unsafe_allow_html=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#side bar 

PAGES = ["Accueil", "Description des données", "Prédiction","Meilleur modèle"]

with st.sidebar:
    #st_lottie(load_lottieurl('https://assets8.lottiefiles.com/packages/lf20_jjojhxyb.json'), height=150)
    image = Image.open('images/magi.png')

    st.image(image,width=260)
choix_page = st.sidebar.radio(label="", options=PAGES)

#



############# Page 1 #############
if choix_page == "Accueil":
    st.image("images/magiline.png",use_column_width=None)
   # st.write("---")
    st.write("##")


    st.write("""
# Comment peut-on prédire l'usage d'une piscine Magiline grâce à la data ?
""")
    st.write("##")
    st.markdown(
    "<p class='intro'>Afin de valider ma thèse professionnelle, j'ai choisi de travailler sur un sujet bien particulier qui est l'étude des comportements des différents équipements d'une piscine Magiline à travers les différents capteurs existants. Cette étude permet d'identifier quels sont les paramètres permettant de prédire la présence dans le bassin.</p>",
    unsafe_allow_html=True)

    st.markdown(
    "<p class='intro'>En vu de réaliser tout ce qui précède, j'ai mis en place une infrastructure Big Data permettant à la fois de collecter, stocker, analyser et enfin, visualiser les données traitées. Accompagnée d'un traitement en temps réel des flux de données générées par la solution domotique i-MAGI-X et de plusieurs algorithmes d'apprentissage automatiques. Ce traitement est basé sur l'analyses des sondes (pH, Redox), des pompes (filtration, chauffage) et des différents équipements en options tels que la nage à contre courant , les robots de nettoyages et les dispositifs de sécurité ,etc.</p>",
    unsafe_allow_html=True)

    st.markdown(
    "<p class='intro'>Cette application WEB est la dernière étape du processus. Il s'agit de mettre le modèle résultant en production. L'objectif est de mettre les connaissances acquises grâce à la modélisation dans le processus de prise de décision sous une forme appropriée.</p>",
    unsafe_allow_html=True)
    st.markdown(
    "<p class='intro'>Pour déployer ma solution, on sert des outils suivants :</p>",
    unsafe_allow_html=True)
    st.markdown(
    "<p class='intro'>- Github : il stocke le code de l’application, le code de la modélisation, le modèle enregistré, les données et un fichier requirements.txt qui contient toutes les librairies dont l’application a besoin pour fonctionner </p>",
    unsafe_allow_html=True)
    st.markdown(
    "<p class='intro'>- Streamlit Cloud : il construit et déploie l’application web à partir du code stocké sur Github et héberge la solution sur son serveur.</p>",
    unsafe_allow_html=True)

    st.markdown(
    "<p class='intro'><b>Passez à la page suivante pour voir les résultats. Bonne lecture !</b></p>",
    unsafe_allow_html=True)


    st.subheader("Pour visiter le site officiel :")

    st.write("• [Le site](https://www.piscines-magiline.fr)")

    lottie_accueil = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_xRmNN8.json')
    st_lottie(lottie_accueil, height=200)
#******************Description*************************

elif choix_page == "Description des données":

    st.markdown('<p class="grand_titre">Chargement du dataset</p>', unsafe_allow_html=True)
    st.write('##')
    col1_1, b_1, col2_1 = st.columns((1, 0.1, 1))
    
    with col2_1:
        st_lottie(load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_zidar9jm.json'), height=200)
    with col1_1:
        dataset_choix = st.selectbox("Dataset",
                                     ["-- Choisissez une option --", "Evènements","Mesures de température","Dataset final (Features)",
                                      "Descriptifs (Label)"], )
        message_ = st.empty()

    if 'choix_dataset' in st.session_state:
        with col1_1:
            message_.success(st.session_state.choix_dataset)




    noms_fichiers =  ["Evènements","Mesures de température","Dataset final (Features)", "Descriptifs (Label)"]
    path_fichiers = ['data/all_data.csv','data/temp_data.csv','data/all_features.csv', 'data/descriptif.csv']

    for i, j in zip(noms_fichiers, path_fichiers):
        if dataset_choix == i:
            if i in  ["Evènements","Mesures de température", "Descriptifs (Label)"]:
                col1, b, col2 = st.columns((1.2, 0.1, 1.4))
                st.session_state.data = pd.read_csv(j,sep=";")
                st.session_state.choix_dataset = "Le fichier chargé est le dataset " + i
                with col1_1:
                    message_.success(st.session_state.choix_dataset)

                with col1:
                    st.write("##")
                    st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
                    st.dataframe(st.session_state.data.head(100),width=750)
                    st.write("##")

                with col2:
                    st.write("##")
                    st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
                    st.write(' - Taille:', st.session_state.data.shape)
                    st.write(' - Nombre de valeurs:', st.session_state.data.shape[0] * st.session_state.data.shape[1])
                    st.write(' - Pourcentage de valeurs manquantes:', round(
                        sum(pd.DataFrame(st.session_state.data).isna().sum(axis=1).tolist()) * 100 / (
                                st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                            '%')

                    fig = pyplt.figure(figsize=(10, 4))
                    
                    fig=px.histogram(st.session_state.data,x=st.session_state.data.columns[1],color=st.session_state.data.columns[1], width=800, height=400)
                    st.plotly_chart(fig)
            else :
                col1, b, col2 = st.columns((1.5, 0.2, 1.4))
                st.session_state.data = pd.read_csv(j,sep=";")
                st.session_state.choix_dataset = "Le fichier chargé est le dataset " + i
                with col1_1:
                    message_.success(st.session_state.choix_dataset)

                with col1:
                    st.write("##")
                    st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
                    st.write(st.session_state.data.head(100))
                    st.write("##")

                with col2:
                    st.write("##")
                    st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
                    st.write(' - Taille:', st.session_state.data.shape)
                    st.write(' - Nombre de valeurs:', st.session_state.data.shape[0] * st.session_state.data.shape[1])
                    st.write(' - Pourcentage de valeurs manquantes:', round(
                        sum(pd.DataFrame(st.session_state.data).isna().sum(axis=1).tolist()) * 100 / (
                                st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                            '%')

                


############# Classification #############



elif choix_page == "Prédiction":
    PAGES_Prédiction = ["RL", "RF", "Decision Tree"]
    st.write("##")
    st.sidebar.title('Choisissez un modèle  ')
    st.sidebar.radio(label="", options=PAGES_Prédiction, key="choix_page_classification")
    st.markdown('<p class="grand_titre">Les résultats de la prédiction</p>', unsafe_allow_html=True)
           

    if st.session_state.choix_page_classification == "RF":
        st.sidebar.title("Choisissez une métrique d'évaluation ")
        metrics = st.sidebar.multiselect("", ('Confusion Matrix', 'ROC Curve'))

        st.write("""# Modèle des forêts aléatoires """)
        st.write("##")
        exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
        with exp2:
            with st.expander("Principe de l'algorithme des forêts aléatoires :"):
                st.write("""
                * 1ère étape : Sélectionner K points de données aléatoires dans l'ensemble d'apprentissage.
                * 2ème étape : Construire les arbres de décision associés aux points de données sélectionnés (sous-ensembles).
                * 3ème étape : Choisir le nombre N pour les arbres de décision à créer
                * 4ème étape : Répéter les étapes 1 et 2
                * 5ème étape : Pour les nouveaux points de données, rechercher les prédictions de chaque arbre de décision 
                    et attribuer les nouvea
                    ux points de données à la catégorie qui remporte la majorité des votes.
                """)
                st.write("##")
                st.image("images/rf.jpg",use_column_width=None)
        st.write("##")
        st.write("##")
         

        # load model 
        model = joblib.load(os.path.join(Saved_model_DATAPATH, RF))

        #train test data
        x_train, x_test = train_test(data_model.iloc[:,1:])
        train_target,test_target = train_test(target)
        y_test = test_target['baignade']
        y_train = train_target['baignade']
        # predict
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        # metrics on train
        accur_train = accuracy_score(y_train, y_pred_train)
        precis_train = precision_score(y_train, y_pred_train, average='micro')
        rappel_train = recall_score(y_train, y_pred_train, average='micro')
        F1_train = f1_score(y_train, y_pred_train, average='micro')

        # metrics on test
        accur_test = accuracy_score(y_test, y_pred_test)
        precis_test = precision_score(y_test, y_pred_test)
        rappel_test = recall_score(y_test, y_pred_test)
        F1_test = f1_score(y_test, y_pred_test)
        _, col1_dt, _ = st.columns((0.5, 1, 0.1))
        _, col1_eval_modele, col2_eval_modele, col3_eval_modele, col4_eval_modele, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
        # Affichage métriques




        if st.sidebar.button("Voir les résultats", key='Voir les résultats'):
            

            with col1_dt:
                st.write("##")
                st.write("##")
                st.subheader('Évaluation par rapport au train set')
                st.write("##")
                st.write("##")
                st.write("##")
            with col1_eval_modele:
                st.metric(label="Precision", value=round(precis_test, 3),
                            delta=round(precis_test - precis_train, 3))
            with col2_eval_modele:
                st.metric(label="Recall", value=round(rappel_test, 3),
                            delta=round(rappel_test - rappel_train, 3))
            with col3_eval_modele:
                st.metric(label="F1 score", value=round(F1_test, 3),
                            delta=round(F1_test - F1_train, 3))
            with col4_eval_modele:
                st.metric(label="Accuracy", value=round(accur_test, 3),
                            delta=round(accur_test - accur_train, 3))  
                            
            
            plot_metrics(metrics)

    elif st.session_state.choix_page_classification == "Decision Tree":
        st.sidebar.title("Choisissez une métrique d'évaluation ")
        metrics = st.sidebar.multiselect("", ('Confusion Matrix', 'ROC Curve'))
        st.write("##")       
        st.write("""# Modèle des arbres de décision""")
        st.write("##")
        exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
        with exp2:
            with st.expander("Principe de l'algorithme decision tree"):
                st.write("""
                * Le principe des SVM consiste à ramener un problème de classification ou de discrimination à un hyperplan (feature space) dans lequel les données sont séparées en plusieurs classes dont la frontière est la plus éloignée possible des points de données (ou "marge maximale") 
                """)
                st.write("##")
                st.image("images/svm.png",use_column_width=None)

        st.write("##")
        st.write("##")


      # load model 
        model = joblib.load(os.path.join(Saved_model_DATAPATH, Dtree))

        #train test data
        x_train, x_test = train_test(data_model.iloc[:,1:])
        train_target,test_target = train_test(target)
        y_test = test_target['baignade']
        y_train = train_target['baignade']
        # predict
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        # metrics on train
        accur_train = accuracy_score(y_train, y_pred_train)
        precis_train = precision_score(y_train, y_pred_train, average='micro')
        rappel_train = recall_score(y_train, y_pred_train, average='micro')
        F1_train = f1_score(y_train, y_pred_train, average='micro')

        # metrics on test
        accur_test = accuracy_score(y_test, y_pred_test)
        precis_test = precision_score(y_test, y_pred_test)
        rappel_test = recall_score(y_test, y_pred_test)
        F1_test = f1_score(y_test, y_pred_test)
        _, col1_dt, _ = st.columns((0.5, 1, 0.1))
        _, col1_eval_modele, col2_eval_modele, col3_eval_modele, col4_eval_modele, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
        # Affichage métriques

        if st.sidebar.button("Voir les résultats", key='Voir les résultats'):
            

            with col1_dt:

                st.write("##")
                st.write("##")
                st.write("##")
                st.subheader('Évaluation par rapport au train set')
                st.write("##")
                st.write("##")
                st.write("##")
            with col1_eval_modele:
                st.metric(label="Precision", value=round(precis_test, 3),
                            delta=round(precis_test - precis_train, 3))
            with col2_eval_modele:
                st.metric(label="Recall", value=round(rappel_test, 3),
                            delta=round(rappel_test - rappel_train, 3))
            with col3_eval_modele:
                st.metric(label="F1 score", value=round(F1_test, 3),
                            delta=round(F1_test - F1_train, 3))
            with col4_eval_modele:
                st.metric(label="Accuracy", value=round(accur_test, 3),
                            delta=round(accur_test - accur_train, 3))  
                            
            
            plot_metrics(metrics)

    elif st.session_state.choix_page_classification == "RL":
        st.sidebar.title("Choisissez une métrique d'évaluation ")
        metrics = st.sidebar.multiselect("", ('Confusion Matrix', 'ROC Curve'))
        st.write("##")        
        st.write("""# Modèle de la régression logistique""")
        st.write("##")
        exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
        with exp2:
            with st.expander("Principe de l'algorithme RL"):
                st.write("""
                * La régression logistique est un type d'apprentissage automatique supervisé utilisé pour prédire la probabilité d'une variable cible. Il est utilisé pour estimer la relation entre une variable dépendante (cible) et une ou plusieurs variables indépendantes. La sortie de la variable dépendante est représentée en valeurs discrètes telles que 0 et 1.
                """)
                st.write("##")
                st.image("images/rl.png",use_column_width=None)  

        
        st.write("##")


      # load model 
        model = joblib.load(os.path.join(Saved_model_DATAPATH, RLOGIST))

        #train test data
        x_train, x_test = train_test(data_model.iloc[:,1:])
        train_target,test_target = train_test(target)
        y_test = test_target['baignade']
        y_train = train_target['baignade']

        #scaler = StandardScaler()
        #x_train.iloc[:,0:25] = scaler.fit_transform(x_train.iloc[:,0:25])
        #x_test.iloc[:,0:25] = scaler.fit_transform(x_test.iloc[:,0:25])
        # predict
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        # metrics on train
        accur_train = accuracy_score(y_train, y_pred_train)
        precis_train = precision_score(y_train, y_pred_train, average='micro')
        rappel_train = recall_score(y_train, y_pred_train, average='micro')
        F1_train = f1_score(y_train, y_pred_train, average='micro')

        # metrics on test
        accur_test = accuracy_score(y_test, y_pred_test)
        precis_test = precision_score(y_test, y_pred_test)
        rappel_test = recall_score(y_test, y_pred_test)
        F1_test = f1_score(y_test, y_pred_test)
        _, col1_dt, _ = st.columns((0.5, 1, 0.1))
        _, col1_eval_modele, col2_eval_modele, col3_eval_modele, col4_eval_modele, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
        # Affichage métriques

        if st.sidebar.button("Voir les résultats", key='Voir les résultats'):
            

            with col1_dt:

                st.write("##")
                st.write("##")
                st.write("##")
                st.subheader('Évaluation par rapport au train set')
                st.write("##")
                st.write("##")
                st.write("##")
            with col1_eval_modele:
                st.metric(label="Precision", value=round(precis_test, 3),
                            delta=round(precis_test - precis_train, 3))
            with col2_eval_modele:
                st.metric(label="Recall", value=round(rappel_test, 3),
                            delta=round(rappel_test - rappel_train, 3))
            with col3_eval_modele:
                st.metric(label="F1 score", value=round(F1_test, 3),
                            delta=round(F1_test - F1_train, 3))
            with col4_eval_modele:
                st.metric(label="Accuracy", value=round(accur_test, 3),
                            delta=round(accur_test - accur_train, 3))  
                            
            st.write("##")  
            st.write("##")  
            plot_metrics(metrics)

   
############# Fin Classification #############

############# Best model #############



elif choix_page == "Meilleur modèle":
    st.markdown('<p class="grand_titre">Résultat du meilleur modèle</p>', unsafe_allow_html=True)
    st.write("##")    
    st.write("##")
    st.write("##")
    st.subheader('Évaluation du modèle sur les données de validation') 
    # load model 
    model = joblib.load(os.path.join(Saved_model_DATAPATH, RF))

    #train test data
    x_train, x_test = train_test(data_model.iloc[:,1:])
    train_target,test_target = train_test(target)
    y_test = test_target['baignade']
    y_train = train_target['baignade']
    # predict
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)


    # metrics on test
    accur_test = accuracy_score(y_test, y_pred_test)
    precis_test = precision_score(y_test, y_pred_test, average='micro')
    rappel_test = recall_score(y_test, y_pred_test, average='micro')
    F1_test = f1_score(y_test, y_pred_test, average='micro')
    _, col1_dt, _ = st.columns((0.1, 1, 0.1))
    _, col1_eval_modele, col2_eval_modele, col3_eval_modele, col4_eval_modele, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
    # Affichage métriques


     
    with col1_eval_modele:
        st.metric(label="Precision", value=round(precis_test, 3))
    with col2_eval_modele:
        st.metric(label="Recall", value=round(rappel_test, 3))
    with col3_eval_modele:
        st.metric(label="F1 score", value=round(F1_test, 3))
    with col4_eval_modele:
        st.metric(label="Accuracy", value=round(accur_test, 3))  
                    
    metrics = ['Confusion Matrix', 'ROC Curve']
    plot_metrics(metrics)



