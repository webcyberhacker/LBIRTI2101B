import streamlit as st
from inference import BirdClassifier
import os

# Configuration
st.set_page_config(page_title="Bird Identifier", page_icon="🐦", layout="centered")
CHEMIN_MODELE = "src/best_model.pth"

# Picture
BIRD_IMAGES = {
    "Merle noir": "https://upload.wikimedia.org/wikipedia/commons/9/9d/Turdus_Merula_in_Saint_Sernin_Croped.jpg",
    "Mésange charbonnière": "https://lejardindesoiseaux.fr/wp-content/uploads/2022/12/51803190217_cb9089c7ff_k.jpg",
    "Mésange bleue": "https://focusingonwildlife.com/news/wp-content/uploads/Blue-Tit-Parus-caeruleus-15.jpg",
    "Rouge-gorge familier": "https://upload.wikimedia.org/wikipedia/commons/f/f3/Erithacus_rubecula_with_cocked_head.jpg",
    "Moineau domestique": "https://image.jimcdn.com/app/cms/image/transf/dimension=1480x10000:format=jpg/path/s5dde8bff85c81b2f/image/idee729741ace7b0a/version/1559635729/fiche-oiseaux-animaux-moineau-domestique-house-sparrow-animal-facts-bird.jpg",
    "Pigeon ramier": "https://www.wnve.nl/images/C_palumbus.png",
    "Étourneau sansonnet": "https://www.oiseaux.net/photos/gerard.fauvet/images/etourneau.sansonnet.gefa.2g.jpg",
    "Pinson des arbres": "https://vigienature.openkeys.science/oiseaux/res/Pinson_des_arbres_2.jpg",
    "Tourterelle turque": "https://www.parc-auxois.fr/wp-content/uploads/2019/06/tourterelle-turque-1.jpg",
    "Geai des chênes": "https://www.animaleco.com/wp-content/uploads/2023/05/167370786_m_normal_none.jpg"
}
IMAGE_DEFAUT = "https://tse1.mm.bing.net/th/id/OIP.uNWfLzqs-dweZw2S9oDBzAHaID?pid=Api"


# load the model
@st.cache_resource
def charger_ia():
    # On crée une instance de notre classe 'BirdClassifier'
    classifier = BirdClassifier(model_path=CHEMIN_MODELE, labels=None)
    return classifier


try:
    ia = charger_ia()
except Exception as e:
    st.error(f"Impossible to charge the model : {e}")
    st.stop()

# Interface
st.title("Bird species identifier")
st.write("""
This identification tool is capable of differentiating 
the 10 most common bird species in Belgium (Source: Natagora).
""")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("""
    **Espèces supportées (1-5) :** 1. Merle noir (*Turdus merula*)  
    2. Mésange charbonnière (*Parus major*)  
    3. Mésange bleue (*Cyanistes caeruleus*)  
    4. Rouge-gorge familier (*Erithacus rubecula*)  
    5. Moineau domestique (*Passer domesticus*)  
    """)

with right_col:
    st.markdown("""
    **Espèces supportées (6-10) :**
    6. Pigeon ramier (*Columba palumbus*)  
    7. Étourneau sansonnet (*Sturnus vulgaris*)  
    8. Pinson des arbres (*Fringilla coelebs*)  
    9. Tourterelle turque (*Streptopelia decaocto*)
    10. Geai des chênes (*Garrulus glandarius*)
    """)

st.markdown("---")  # Séparateur visuel

fichier = st.file_uploader("Audio file (.wav or mp3)", type=["wav", "mp3"])

if fichier and st.button("Analyse"):
    try:
        # Sauvegarde temporaire
        with open("temp.wav", "wb") as f:
            f.write(fichier.getbuffer())  # <--- CORRIGÉ ICI (Décalé vers la droite)

        # Prédiction
        resultat = ia.predict("temp.wav")
        specie_name = resultat['label']

        # Result affichage
        st.markdown("### Result")

        col_res_texte, col_res_image = st.columns([1, 1])

        with col_res_texte:
            st.success(f"Prediction : **{specie_name}**")
            st.info(f"Confidence : {resultat['confidence'] * 100:.2f}%")

        with col_res_image:
            url_img = BIRD_IMAGES.get(specie_name, IMAGE_DEFAUT)
            st.image(url_img, caption=f"Illustration : {specie_name}", use_column_width=True)

    except Exception as e:  # <--- CORRIGÉ ICI (Aligné avec le try)
        st.error(f"An error occurred during the analysis: {e}")

    finally:  # <--- CORRIGÉ ICI (Aligné avec le try)
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: grey;'>
       Project completed by Morgane Clicque, Bastien d'Argembeau, Gabriel Deligne
       Matthias Dewé, Valérian Vermeeren and Alix Wagner as part of the course LBIRTI2101B.  
        The model was trained using the xenocanto database.
    </div>
    """, unsafe_allow_html=True)

