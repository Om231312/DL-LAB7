# 🧪 Lab 7: BERT Fine-Tuning for Text Classification

## 📌 Objective

To fine-tune a pre-trained **BERT model** for text classification tasks.

---

## 🧠 Introduction

BERT (Bidirectional Encoder Representations from Transformers) is a powerful NLP model that understands context from both directions.

---

## ⚙️ Methodology

### 🔄 Steps Performed

1. Load pre-trained BERT
2. Tokenize input text
3. Add classification layer
4. Train model on dataset

---

## 💻 Sample Code

```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = ["I love AI", "I hate bugs"]
labels = [1, 0]

encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=model.compute_loss,
    metrics=['accuracy']
)

model.fit(encodings['input_ids'], labels, epochs=2)
```

---

## 📈 Expected Results

* High accuracy
* Better contextual understanding
* Works well on complex NLP tasks

---

## 🎯 Conclusion

BERT significantly improves NLP performance due to its bidirectional context understanding and pre-trained knowledge.

---

## 🚀 Future Improvements

* Train on larger datasets
* Use different BERT variants
* Fine-tune hyperparameters

---

## 📌 Author

**Om Prakash Kannaujiya**
