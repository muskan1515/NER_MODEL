import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import re

# ---------------- Config ----------------
DATA_PATH = "/content/sample_data/datasets/ner.csv"
EXPORT_DIR = "/content/ner_model"
VOCAB_SIZE = 20000
MAX_LEN = 200
EMBED_DIM = 100        # GloVe dimension
BATCH_SIZE = 32
EPOCHS = 25
GLOVE_PATH = "/content/sample_data/glove.6B.100d.txt"  # path to glove embeddings
# ----------------------------------------

# 1️⃣ Load dataset
df = pd.read_csv(DATA_PATH)
sentences = df["tokens"].astype(str).tolist()
tag_strs = df["tags"].astype(str).tolist()
tokens_list = [s.split() for s in sentences]
tags_list = [t.split() for t in tag_strs]

# 2️⃣ Build tag vocabulary (PAD=0)
all_tags = sorted({tag for tags in tags_list for tag in tags})
tag2idx = {"PAD": 0}
for i, tg in enumerate(all_tags, start=1):
    tag2idx[tg] = i
idx2tag = {i: t for t, i in tag2idx.items()}
num_tags = len(tag2idx)

# 3️⃣ Convert tags to integers & pad
y_seq = [[tag2idx[tag] for tag in tags] for tags in tags_list]
y_padded = pad_sequences(y_seq, maxlen=MAX_LEN, padding="post", truncating="post", value=0)

# 4️⃣ Text preprocessing
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()
sentences_cleaned = [clean_text(s) for s in sentences]

# 5️⃣ Tokenize words and build word index
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences_cleaned)
X_seq = tokenizer.texts_to_sequences(sentences_cleaned)
X_padded = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post", truncating="post")

word_index = tokenizer.word_index

# 6️⃣ Load GloVe embeddings
embeddings_index = {}
with open(GLOVE_PATH, encoding="utf8") as f:
    for line in f:
        values = line.strip().split()
        if len(values) != EMBED_DIM + 1:  # 1 word + 100 floats
            continue  # skip malformed line
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs


# 7️⃣ Prepare embedding matrix
embedding_matrix = np.zeros((VOCAB_SIZE, EMBED_DIM))
for word, i in word_index.items():
    if i >= VOCAB_SIZE:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 8️⃣ Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_padded, y_padded, test_size=0.1, random_state=42
)

# 9️⃣ Build tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 🔟 Build BiLSTM model with pretrained embeddings
model = models.Sequential([
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False, mask_zero=True),
    layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
    layers.TimeDistributed(layers.Dense(num_tags, activation="softmax"))
])

model.summary()

# 1️⃣1️⃣ Masked Accuracy
def masked_accuracy(y_true, y_pred):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    matches = tf.cast(tf.equal(tf.cast(y_true, tf.int64), y_pred), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.maximum(tf.reduce_sum(mask), 1)

# 1️⃣2️⃣ Compile with gradient clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=[masked_accuracy])

# 1️⃣3️⃣ Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(os.path.join(EXPORT_DIR, "ner_best_model.keras"), save_best_only=True)
]

# 1️⃣4️⃣ Train
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# 1️⃣5️⃣ Evaluate
val_loss, val_acc = model.evaluate(val_ds)
print("Validation loss:", val_loss, "Validation accuracy (token-level):", val_acc)

# 1️⃣6️⃣ Save model & tag map
os.makedirs(EXPORT_DIR, exist_ok=True)
model.save(os.path.join(EXPORT_DIR, "ner_model.keras"))
with open(os.path.join(EXPORT_DIR, "tag_map.pkl"), "wb") as f:
    pickle.dump({"tag2idx": tag2idx, "idx2tag": idx2tag}, f)
with open(os.path.join(EXPORT_DIR, "training_history.json"), "w") as f:
    json.dump(history.history, f)

print("Saved model, tag map and history to:", EXPORT_DIR)

# Example sentence
test_sentence = "John lives in New York City ."
test_sentence_clean = [clean_text(test_sentence)]
seq = tokenizer.texts_to_sequences(test_sentence_clean)
seq_padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
preds = model.predict(seq_padded)
pred_tags = [idx2tag[i] for i in np.argmax(preds[0], axis=-1)]

# Remove PAD tokens
pred_tags = [t for t, w in zip(pred_tags, seq_padded[0]) if w != 0]

print(list(zip(test_sentence.split(), pred_tags)))
