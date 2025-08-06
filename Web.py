import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from image_segmentation import segment_image

# Configure page settings
st.set_page_config(
    page_title="Skin Disease Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .css-1d391kg {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Hardcoded credentials for demonstration
USERNAME = 'user'
PASSWORD = 'pass'

# Initialize session states
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'login_attempts' not in st.session_state:
    st.session_state['login_attempts'] = 0

# Disease information dictionary
disease_info = {
    'Melanocytic nevi': {
        'description': 'Common moles that are usually harmless',
        'risk_level': 'Low',
        'common_locations': 'Can appear anywhere on the body',
        'symptoms': ['Round or oval shape', 'Smooth surface', 'Uniform color']
    },
    'Melanoma': {
        'description': 'A serious form of skin cancer',
        'risk_level': 'High',
        'common_locations': 'Any part of the skin',
        'symptoms': ['Asymmetrical shape', 'Irregular borders', 'Color variations']
    },
    'Benign keratosis': {
        'description': 'Non-cancerous skin growths that appear with age',
        'risk_level': 'Low',
        'common_locations': 'Face, chest, and back',
        'symptoms': ['Waxy appearance', 'Light brown to black color', 'Slightly raised']
    },
    'Basal cell carcinoma': {
        'description': 'Most common type of skin cancer',
        'risk_level': 'Moderate to High',
        'common_locations': 'Sun-exposed areas',
        'symptoms': ['Pearly, waxy bump', 'Flat, flesh-colored lesion', 'Bleeding or scabbing sore']
    },
    'Actinic keratoses': {
        'description': 'Precancerous growths',
        'risk_level': 'Moderate',
        'common_locations': 'Sun-exposed areas',
        'symptoms': ['Rough, scaly patches', 'Pink to red color', 'May be itchy']
    },
    'Vascular lesions': {
        'description': 'Abnormalities of blood vessels',
        'risk_level': 'Low to Moderate',
        'common_locations': 'Any part of the body',
        'symptoms': ['Red or purple color', 'May be raised or flat', 'Can be warm to touch']
    },
    'Dermatofibroma': {
        'description': 'Common benign skin growths',
        'risk_level': 'Low',
        'common_locations': 'Lower legs',
        'symptoms': ['Firm, raised growth', 'Brown to pink color', 'May dimple when pressed']
    }
}

# Function to preprocess the uploaded image
def preprocess_image(image):
    try:
        image = np.array(image)
        image = cv2.resize(image, (64, 64))
        image = image.astype('float32') / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Define a function to make predictions and return probabilities
def predict_proba(images):
    images = np.array([cv2.resize(img, (64, 64)) for img in images])
    images = images.astype('float32') / 255.0
    return model.predict(images)

# Initialize LIME explainer
explainer = lime_image.LimeImageExplainer()

# Generate explanation for an uploaded image
def generate_lime_explanation(image):
    explanation = explainer.explain_instance(
        np.array(image),
        predict_proba,
        top_labels=5,
        hide_color=0,
        num_samples=1000
    )
    return explanation

# Example function to generate a counterfactual image
def generate_counterfactual(image, target_label):
    return image  # Placeholder - returns original image

# Main application logic
if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title('🏥 Medical Login Portal')
        
        with st.container():
            st.markdown("### 👤 User Authentication")
            username = st.text_input('Username', placeholder='Enter your username')
            password = st.text_input('Password', type='password', placeholder='Enter your password')
            
            if st.button('🔐 Login', use_container_width=True):
                if username == USERNAME and password == PASSWORD:
                    st.session_state['logged_in'] = True
                    st.success('🎉 Login successful! Redirecting...')
                    st.experimental_rerun()
                else:
                    st.session_state['login_attempts'] += 1
                    remaining_attempts = 3 - st.session_state['login_attempts']
                    if remaining_attempts > 0:
                        st.error(f'❌ Invalid credentials. {remaining_attempts} attempts remaining.')
                    else:
                        st.error('🚫 Maximum login attempts reached. Please try again later.')
                        st.session_state['login_attempts'] = 0

# Main application UI after login
if st.session_state['logged_in']:
    st.sidebar.title('🔍 Navigation')
    page = st.sidebar.radio('Select a page:', ['🏠 Home', '🔲 Segmentation', '🔬 Skin Disease Prediction'])

    if page == '🔲 Segmentation':
        st.title('🔲 Skin Lesion Segmentation')
        uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'png'])
        if uploaded_file is not None:
            image_path = uploaded_file.name
            with open(image_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            segmented_output = segment_image(image_path)
            st.image(segmented_output, caption='Segmented Output', use_column_width=True)
            
    elif page == '🔬 Skin Disease Prediction':
        st.title("🔬 Skin Disease Prediction with Explainability")
        
        col1, col2 = st.columns([2,3])
        
        with col1:
            selected_disease = st.selectbox(
                'Select a disease to learn more:',
                list(disease_info.keys())
            )
            
            if selected_disease:
                st.markdown(f"### ℹ️ About {selected_disease}")
                st.markdown(f"**Description:** {disease_info[selected_disease]['description']}")
                st.markdown(f"**Risk Level:** {disease_info[selected_disease]['risk_level']}")
                st.markdown(f"**Common Locations:** {disease_info[selected_disease]['common_locations']}")
                st.markdown("**Common Symptoms:**")
                for symptom in disease_info[selected_disease]['symptoms']:
                    st.markdown(f"- {symptom}")
        
        with col2:
            st.markdown("### 📤 Upload Image for Analysis")
            uploaded_file = st.file_uploader("Choose an image...", type="jpg")
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    if st.button('🔍 Analyze Image', use_container_width=True):
                        with st.spinner('Analyzing image...'):
                            # Load model and make prediction
                            model = tf.keras.models.load_model("D:/Dataset/models/skin_lesion_model_3.h5")
                            processed_image = preprocess_image(image)
                            
                            if processed_image is not None:
                                prediction = model.predict(processed_image)
                                predicted_label = list(disease_info.keys())[np.argmax(prediction)]
                                
                                st.markdown("### 📊 Analysis Results")
                                st.success(f"Predicted Condition: **{predicted_label}**")
                                
                                # Generate and display confidence scores
                                st.markdown("### 🎯 Confidence Scores")
                                for i, (disease, prob) in enumerate(zip(disease_info.keys(), prediction[0])):
                                    st.progress(float(prob))
                                    st.markdown(f"{disease}: {prob:.2%}")
                                
                                # Generate and display LIME explanation
                                try:
                                    explanation = generate_lime_explanation(image)
                                    temp, mask = explanation.get_image_and_mask(
                                        np.argmax(prediction),
                                        positive_only=False,
                                        num_features=10,
                                        hide_rest=False
                                    )
                                    
                                    st.markdown("### 🔍 Visual Explanation")
                                    fig, ax = plt.subplots()
                                    ax.imshow(mark_boundaries(temp, mask))
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error generating explanation: {str(e)}")
                                
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    else:  # Home page
        st.title('🏥 Welcome to the Medical Resource App')
        st.markdown("""
        ### 👋 Welcome to our Skin Disease Analysis Platform
        
        This application helps medical professionals analyze skin conditions using advanced AI technology.
        
        #### 🔑 Key Features:
        - 🔍 Image Segmentation
        - 🔬 Disease Prediction
        - 📊 Detailed Analysis
        - 📋 Comprehensive Reports
        
        #### 📝 Getting Started:
        1. Use the sidebar to navigate between different tools
        2. Upload your images for analysis
        3. Review the detailed results and explanations
        
        #### ⚠️ Important Note:
        This tool is meant to assist medical professionals and should not be used as a replacement for professional medical diagnosis.
        """)

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ by Medical AI Team")