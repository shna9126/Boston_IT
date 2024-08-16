import streamlit as st
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer ,pipeline
import pandas as pd

# Function to load and apply custom CSS
def load_css(file_path):
    with open(file_path, "r") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# Load custom CSS
load_css("styles.css") 


# Define the sentiment labels with colors and emojis
sentiment_info = {
    0: {"label": "Very Negative", "color": "#FF4C4C", "emoji": "😡"},
    1: {"label": "Negative", "color": "#FF9B9B", "emoji": "😞"},
    2: {"label": "Neutral", "color": "#FFFF00", "emoji": "😐"},
    3: {"label": "Positive", "color": "#9BFF9B", "emoji": "😊"},
    4: {"label": "Very Positive", "color": "#4CFF4C", "emoji": "😍"},
}

# Function to load the model based on selection
def load_model(model_name):
    if model_name == 'DistilBERT':
        return pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
    elif model_name == 'Fine-tuned BERT':
        # Assuming you have the fine-tuned BERT on Kaggle or local machine
        model_path = "/Users/shivam/vscode/bert_model"  # Update this path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        return pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    elif model_name == 'Fine-tuned RoBERTa':
        # Assuming you have the fine-tuned RoBERTa on Kaggle or local machine
        model_path = "/Users/shivam/vscode/roberta_model"  # Update this path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        return pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Load the CSV file
def load_csv():
    return pd.read_csv("/Users/shivam/vscode/model_comparison2.csv",index_col='SNo')



st.title('🔍 Sentiment Analysis Tool For Indian Election - 2024')

# Dropdown for selecting model
model_option = st.selectbox("Choose a model:", ["DistilBERT", "Fine-tuned BERT", "Fine-tuned RoBERTa"])

# Load the selected model
model = load_model(model_option)

st.markdown("### Enter your comment below and click **Analyze** to see the sentiment:")

text_input = st.text_area("Your Comment:")

if st.button('Analyze'):
    if text_input:
        with st.spinner('Analyzing sentiment...'):
            if model=='DistilBERT':
                result = model(text_input)[0]
                label = result['label']
                label_num = int(label.split(' ')[0]) - 1  # Convert '5 stars' to 4 (for zero-based indexing)
            else:
                result = model(text_input)[0]
                label = result['label']  # e.g., "LABEL_1"
                # Extract the label number from "LABEL_X"
                label_num = int(label.split('_')[1])  # Convert "LABEL_1" to 1, "LABEL_0" to 0, etc.
            
            # Get sentiment information
            sentiment_data = sentiment_info.get(label_num, {"label": "Unknown", "color": "#FFFFFF", "emoji": "❓"})
            sentiment = sentiment_data["label"]
            color = sentiment_data["color"]
            emoji = sentiment_data["emoji"]
            
            # Display the sentiment result with color and emoji
            st.markdown(f"<h3 style='color: {color};'>{emoji} <strong>Sentiment:</strong> {sentiment} - Label: {label_num}</h3>", unsafe_allow_html=True)
            
    else:
        st.warning("Please enter a comment to analyze.")

# Display a dataset image
st.image("/Users/shivam/vscode/distri.png", caption="Distribution of Data Sources", use_column_width=True)

# Display comparative analysis CSV file
st.markdown("## 📊 Comparative Analysis")
df = load_csv()
st.dataframe(df)


# Credits Section
st.markdown("""
    ## 🙌 Credits
    **Project by: Shivam Naik**  
    Special Thanks to: Jeevitha Mam and Laxmi Mam  
    """)