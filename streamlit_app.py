import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True
    )

    # Convert inputs to PyTorch tensors
    input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0)

    # Run the model prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get the predicted sentiment label
    sentiment_label = torch.argmax(outputs[0])

    return sentiment_label.item()

def main():
    st.title("Sentiment Analysis")
    st.write("Welcome to the Sentiment Analysis App!")
    st.write("Enter a text and the app will predict its sentiment (positive or negative).")

    # User input for the text
    text = st.text_area("Enter text")

    if st.button("Predict"):
        # Perform sentiment prediction
        sentiment_label = predict_sentiment(text)
        sentiment = "Positive" if sentiment_label == 1 else "Negative"
        st.success(f"The predicted sentiment is: {sentiment}")

if __name__ == '__main__':
    main()
