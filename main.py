import csv
import streamlit as st
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize

st.title('App tìm các công việc liên quan nhau')
input = st.text_input("Viết công việc muốn tìm:")

if len(input) > 0:
    model = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')

    with open('data/jobs.csv', newline='') as f:
        reader = csv.reader(f)
        input_text = list(reader)

    input_text.remove(['\ufeffJobs'])
    sentences = [input]

    for i in input_text:
        for j in i:
            sentences.append(j)


    sentences_tokenizer = [tokenize(sentence) for sentence in sentences]
    embeddings = model.encode(sentences_tokenizer)
    cosine_list = []

    def custom_sort(elem):
        return elem[0]

    # calculate cosine similarity
    for i in range(1,len(embeddings)):
    	cosine_sim = 1-cosine(embeddings[0],embeddings[i])
    	cosine_list.append([cosine_sim,input_text[i-1]])

    cosine_list.sort(key=custom_sort,reverse=True)

    for i in range(len(cosine_list[:10])):
        st.write("Ngành thứ {} phù hợp là: {} (độ phù hợp: {cos:.2f})".format(i+1, cosine_list[i][1][0],cos=cosine_list[i][0]))
else: pass