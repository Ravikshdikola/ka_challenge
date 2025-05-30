import streamlit as st                      # Import Streamlit for building web app UI
import requests                            # Import requests to make HTTP API calls

st.title("🧠 Text Classification & Embedding")  # Display the main title of the app

st.subheader("Enter your text below")     # Display a subheader prompt for user input
input_text = st.text_area("Input", "", height=200)  # Multi-line text input box with label "Input" and height 200px

if st.button("Submit"):                    # Create a "Submit" button, trigger when clicked
    if input_text.strip():                 # Check if input text is not empty (ignoring whitespace)
        try:
            # Send input text as JSON payload to API endpoint, wrapped in a list (API expects list of texts)
            response = requests.post("http://localhost:8000/process", json={"texts": [input_text]})
            result = response.json()       # Parse JSON response from API

            st.subheader(f"🔎 Predicted Mathematical Topic")  # Display subheader for prediction result
            # st.subheader("🔎 Prediction")                  # (commented out alternate subheader)
            st.write(f"**Predicted Class:** {result['predictions'][0]}")  # Show predicted class (first item)

            st.subheader("📊 Embedding Vector")               # Display subheader for embeddings
            st.json(result["embeddings"][0])                 # Show embedding vector formatted as JSON (first item)

        except Exception as e:            # Catch any exceptions from API call or JSON parsing
            st.error(f"Error: {e}")        # Display error message in the app

    else:
        st.warning("Please enter some text.")  # Warn user if input was empty or just whitespace
