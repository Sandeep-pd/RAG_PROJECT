import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline,AutoTokenizer,AutoModelForCausalLM

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


def rag_query_v2(query,k=2):
  # We need to convert user's query in embedding
  query_embedding =  embedding_model.encode([query])
  distances , indices = index.search(query_embedding,k)
  retrived_docs = [knowledge_base[i] for i in indices[0]]
  context = "\n".join(retrived_docs)
  augmented_prompt = f"Based on the following information, answer the questions : \n content:\n {context} \n and question is {query}"
  print("-"*50)
  llm_response = generator(augmented_prompt,max_new_tokens=100,temperature=0.7,do_sample=True)
  print(llm_response)    



