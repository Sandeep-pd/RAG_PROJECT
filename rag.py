import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline,AutoTokenizer,AutoModelForCausalLM
import google.generativeai as genai
import os
import sqlite3
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.5-flash')




knowledge_base =[
    "High-Level and Readable: Python is designed to be highly readable and has a clean, straightforward syntax that closely resembles plain English, making it exceptionally beginner-friendly.",
    "Interpreted Language: Python code is executed line by line by an interpreter rather than being compiled into machine code beforehand, which simplifies debugging and rapid prototyping.",
    "Dynamically Typed: You do not need to explicitly declare the data type of a variable when creating it; Python determines the type automatically at runtime.",
    "Object-Oriented and Multi-Paradigm: Python supports multiple programming styles, including object-oriented, procedural, and functional programming paradigms.",
    "Extensive Standard Library: It comes with a vast built-in standard library (\"batteries included\") that provides modules for handling tasks like file I/O, regular expressions, math, and internet protocols.",
    "Cross-Platform Compatibility: Python code can run seamlessly across various operating systems—including Windows, macOS, Linux, and Unix—without requiring major modifications.",
    "Massive Ecosystem of Third-Party Packages: The Python Package Index (PyPI) hosts hundreds of thousands of independent packages, expanding its capabilities for web development, data science, automation, and more.",
    "Dominance in Data Science and AI: Python is the undisputed industry standard for data analysis, machine learning, and artificial intelligence, powered by dominant libraries like NumPy, Pandas, TensorFlow, and PyTorch.",
    "Automatic Memory Management: Python handles memory allocation and deallocation automatically using reference counting and a built-in garbage collector, freeing developers from manual memory management.",
    "Strong Community Support: Python boasts one of the largest, most active, and welcoming developer communities in the world, offering endless documentation, tutorials, and open-source contributions.",
    "Python was created by Guido van Rossum, and its first official release (version 0.9.0) was published in 1991 (with development starting in late 1989)."
]
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(knowledge_base)
embeddings_dimension = embeddings.shape[1]
embeddings_dimension
index = faiss.IndexFlatL2(embeddings_dimension)
index.add(embeddings)

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
generator = pipeline(task="text-generation",model=model,tokenizer=tokenizer)


def rag_query_gemini(query,k=2):
  query_embedding =  embedding_model.encode([query])
  distances , indices = index.search(query_embedding,k)
  retrived_docs = [knowledge_base[i] for i in indices[0]]
  context = "\n".join(retrived_docs)

  # Constract an augumented prompt for Gemini model
  # Note - FYI , A good prompt is crucial for efficient RAG with any advanced LLM
  augumented_prompt = (
      f"Based on the following information , please answer the question thoroughly and concisely. also if question is not related to Content then please search in Gemini knowledge base,
      so give priority to below mentioned content but do not limit your search if question has different context than my knowledge base\n"
      f"Content:\n{context}\n"
      f"Question:{query}\n"
      f"Answer:
  ")


  print("\n Augumented prompt ",augumented_prompt)
  print("*"*50)


  response = gemini_model.generate_content(augumented_prompt,
                                           generation_config=genai.GenerationConfig(
                                               temperature=0.2,
                                               max_output_tokens=700
                                           ) )
  print(response)
  return response


