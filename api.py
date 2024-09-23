from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from transformers import BertTokenizer, BertModel
# from sklearn.linear_model import LogisticRegression
# import torch
# import os


# Initialize Flask app
app = Flask(__name__)

# log_reg_model = LogisticRegression(max_iter=1000)

# Load the dataset from a CSV file
# df = pd.read_csv(os.path.join(os.getcwd(), "profanity_en.csv"))
 
# df_filtered = df[df['category_1'] == 'sexual anatomy / sexual acts']
# df_filtered = df_filtered[['text']]
# df_filtered['label'] = 1  # Binary label for grooming

# df2 = pd.read_csv(os.path.join(os.getcwd(), "Non-grooming set_utf8.csv"))
# df2 = df2[['text', 'label']]  # Assume 'label' column exists

# # Combine the two datasets 
# combined_df = pd.concat([df_filtered, df2], ignore_index=True)

# # Convert 'label' column to numeric, handling non-numeric values
# combined_df['label'] = pd.to_numeric(combined_df['label'], errors='coerce').fillna(0).astype(int)
# # 'coerce' replaces non-numeric values with NaN, which are then filled with 0 and converted to integers.

# combined_df.shape

# combined_df.head(1000)

# # Split the data into training and test sets
# train_texts, test_texts, train_labels, test_labels = train_test_split(
#     combined_df['text'], combined_df['label'], test_size=0.2, random_state=42
# )

# # Load BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# # Function to encode text using BERT
# def encode_texts(texts):
#     encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')
#     with torch.no_grad():
#         outputs = model(**encodings)
#     # Use the [CLS] token embedding for classification
#     return outputs.last_hidden_state[:, 0, :].numpy()
  
# # Encode the training and test texts
# train_embeddings = encode_texts(train_texts)
# test_embeddings = encode_texts(test_texts)

# # Train a Logistic Regression model on the BERT embeddings
# log_reg_model = LogisticRegression(max_iter=1000)
# log_reg_model.fit(train_embeddings, train_labels)

# # Make predictions on the test set
# test_predictions = log_reg_model.predict(test_embeddings)


# Function to determine if a conversation is grooming
def is_grooming_conversation(texts, threshold=0.5):
    embeddings = encode_texts(texts)
    predictions = log_reg_model.predict(embeddings)
    grooming_count = sum(predictions)
    if grooming_count / len(texts) > threshold:
        return True
    return False

# API route to classify grooming conversations
@app.route('/predict', methods=['POST'])
def predict_grooming():
    try:
        # Get the text input from the JSON request
        data = request.get_json()
        texts = data['texts']  # Expecting 'texts' to be a list of strings
        # texts = np.array(["FYI 69"])
        # Predict grooming or non-grooming
        is_grooming = is_grooming_conversation(texts)

        # Prepare the response
        response = {
            "is_grooming": is_grooming,
            "message": "The conversation is classified as grooming." if is_grooming else "The conversation is classified as non-grooming."
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
  return jsonify({"ok" :True})


# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)