import tensorflow as tf
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load model
model = tf.keras.models.load_model('ner_best_model.keras')

# Load tag mapping
with open('tag_map.pkl', 'rb') as f:
    tag_map = pickle.load(f)

tag2idx = tag_map["tag2idx"]
idx2tag = tag_map["idx2tag"]

# FastAPI setup
app = FastAPI(title="NER Model")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_issue(input: TextInput):
    text = [input.text] if isinstance(input.text, str) else input.text

    # Predict
    predicted_response = model.predict(np.array(text), verbose=0)
    pred_idxs = np.argmax(predicted_response, axis=-1)

    output = []
    for i, t in enumerate(text):
        tokens = t.split()[:100]
        tags = [idx2tag.get(int(idx), "PAD") for idx in pred_idxs[i][:len(tokens)]]
        output.append({"token": tokens, "tag": tags})

    return output
