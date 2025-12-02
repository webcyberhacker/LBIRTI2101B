import streamlit as st
from inference import BirdClassifier
import os

# Configuration
st.set_page_config(page_title="Bird Identifier", page_icon="🐦", layout="centered")
CHEMIN_MODELE = "src/best_model.pth"

# Picture

LATIN_TO_FRENCH = {
    "Turdus_merula": "Merle noir",
    "Parus_major": "Mésange charbonnière",
    "Cyanistes_caeruleus": "Mésange bleue",
    "Erithacus_rubecula": "Rouge-gorge familier",
    "Passer_domesticus": "Moineau domestique",
    "Columba_palumbus": "Pigeon ramier",
    "Sturnus_vulgaris": "Étourneau sansonnet",
    "Fringilla_coelebs": "Pinson des arbres",
    "Streptopelia_decaocto": "Tourterelle turque",
    "Garrulus_glandarius": "Geai des chênes"
}

BIRD_IMAGES = {
    "Merle noir": "src/assets/merle_noir.jpg",
    "Mésange charbonnière": "src/assets/mesange_charbo.jpg",
    "Mésange bleue": "src/assets/mesange_bleue.jpg",
    "Rouge-gorge familier": "src/assets/rouge_gorge_familier.jpg",
    "Moineau domestique": "src/assets/moineau_domestique.jpg",
    "Pigeon ramier": "src/assets/pigeon_ramier.png",
    "Étourneau sansonnet": "src/assets/etourneau_sansonnet.jpg",
    "Pinson des arbres": "src/assets/Pinson_des_arbres.jpg",
    "Tourterelle turque": "src/assets/tourterelle_turque.jpg",
    "Geai des chênes": "src/assets/geai_des_chenes.jpg"
}
IMAGE_DEFAUT = "src/assets/silhouete.jpg"


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
    1. Merle noir (*Turdus merula*)  
    2. Mésange charbonnière (*Parus major*)  
    3. Mésange bleue (*Cyanistes caeruleus*)  
    4. Rouge-gorge familier (*Erithacus rubecula*)  
    5. Moineau domestique (*Passer domesticus*)  
    """)

with right_col:
    st.markdown("""
    6. Pigeon ramier (*Columba palumbus*)  
    7. Étourneau sansonnet (*Sturnus vulgaris*)  
    8. Pinson des arbres (*Fringilla coelebs*)  
    9. Tourterelle turque (*Streptopelia decaocto*)  
    10. Geai des chênes (*Garrulus glandarius*)
    """)
# Analyse
st.markdown("---")
fichier = st.file_uploader("Audio file (.wav or mp3)", type=["wav", "mp3"])

button = False
if fichier:
    button = st.button("Analyse")

st.markdown("### Result")
col_res_texte, col_res_image = st.columns([1, 1])
with col_res_texte:
    placeholder_texte = st.empty()
with col_res_image:
    placeholder_image = st.empty()

if fichier and button:
    try:
        # Sauvegarde temporaire
        with open("temp.wav", "wb") as f:
            f.write(fichier.getbuffer())

        # Prédiction
        resultat = ia.predict("temp.wav")
        latin = resultat['label']
        confidence = resultat['confidence']
        specie_name = LATIN_TO_FRENCH.get(latin, latin)

        # Result affichage
        placeholder_texte.empty()
        with col_res_texte:

            if confidence < 0.70:
                st.error("⚠️ Unknown species")
                st.write("This song cannot be identified with a sufficient degree of confidence.")
                st.caption(f"Best match : {specie_name} ({confidence * 100:.1f}%), but that's not enough.")

                url_img = IMAGE_DEFAUT
                caption_img = "Unknown"

            elif 0.70 <= confidence < 0.90:
                st.warning(f"Uncertain identification : **{specie_name}**")
                st.write(
                    f"Similarities have been detected ({confidence * 100:.1f}%), but it could be another species.")
                st.caption(f"Scientific name: *{latin}*")

                url_img = BIRD_IMAGES.get(specie_name, IMAGE_DEFAUT)
                caption_img = f"Likely illustration : {specie_name}"

            else:
                st.success(f"Prediction : **{specie_name}**")
                st.info(f"Confidence : {confidence * 100:.2f}%")
                st.caption(f"Scientific name: *{latin}*")

                url_img = BIRD_IMAGES.get(specie_name, IMAGE_DEFAUT)
                caption_img = f"Illustration : {specie_name}"

            placeholder_image.image(url_img, caption=caption_img, width="stretch")

    except Exception as e:
        st.error(f"An error occurred during the analysis: {e}")

    finally:
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")

# Footer
st.markdown("---")
# Création de deux colonnes : Texte (7 parts) | Logo (1 part)
col_footer_text, col_footer_logo = st.columns([7, 1])

with col_footer_text:
    st.markdown("""
    <div style='text-align: right; color: grey; font-size: 12px;'>
       Project completed by Morgane Clicque, Bastien d'Argembeau, Gabriel Deligne,<br>
       Matthias Dewé, Valérian Vermeeren and Alix Wagner.<br>
       <i>Course LBIRTI2101B - 2025-2026>
    </div>
    """, unsafe_allow_html=True)

with col_footer_logo:
    # Logo UCLouvain
    st.image("src/assets/agro.jpeg",
             width="stretch")


