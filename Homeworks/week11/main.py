import numpy as np
import time
from transformers import BertTokenizer, TFBertModel
from concurrent.futures import ProcessPoolExecutor

# Cargar el tokenizador y el modelo
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

def process_text(text):
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Convert to numpy array

def generate_bert_embeddings(texts):
    with ProcessPoolExecutor() as executor:
        embeddings = list(executor.map(process_text, texts))
    
    # Convertir la lista de embeddings en un numpy array y transponerlo
    return np.array(embeddings).transpose(0, 2, 1)

# Medir el tiempo de procesamiento
start_time = time.time()

# Asumiendo que 'corpus' es una lista de textos
corpus_bert = generate_bert_embeddings(corpus)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Tiempo de procesamiento: {elapsed_time:.2f} segundos")
print(corpus_bert.shape)
