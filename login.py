import streamlit as st
import subprocess
import time

def check_credentials(username, password):
    return username == "Rutu" and password == "0206"

def main():
    st.set_page_config(
        page_title="Login - Skin Disease Detection System",
        page_icon="🔒",
        layout="centered"
    )

    # Custom CSS for the login page
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            height: 3rem;
            margin-top: 1rem;
        }
        .login-container {
            max-width: 400px;
            margin: auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: white;
        }
        .title {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state for login status
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Center the form on the page
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Title and Logo
    st.markdown('<h1 class="title">🏥<br>Skin Disease Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### Login")

    # Login Form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if check_credentials(username, password):
            # Show success message
            st.success("Login Successful! Redirecting...")
            
            # Set login status
            st.session_state.logged_in = True
            
            # Add a delay for the success message to be visible
            time.sleep(2)
            
            # Redirect to main application
            subprocess.Popen(["streamlit", "run", "Rag_modified.py"])
            st.stop()
        else:
            st.error("Invalid username or password")

    st.markdown('</div>', unsafe_allow_html=True)

    # Add some helpful text at the bottom
    st.markdown("""
        <div style='text-align: center; margin-top: 2rem; color: #666;'>
            <p>Please login with your credentials to access the system.</p>
            <p>If you need help, contact system administrator.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()