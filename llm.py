def rag_query_gemini_sqlite(query,k=2):
  query_embedding =  embedding_model.encode([query])
  distances , indices = index.search(query_embedding,k)
  # change this logic and get data from sqlite table retrived_docs = [knowledge_base[i] for i in indices[0]]
  # create content so that we can supply to prompt context = "\n".join(retrived_docs)
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  context = list(cursor.execute(f"SELECT * from {TABLE_NAME}"))
  conn.close()

  # Constract an augumented prompt for Gemini model
  # Note - FYI , A good prompt is crucial for efficient RAG with any advanced LLM
  augumented_prompt = (
      f"Based on the following information , please answer the question thoroughly and concisely.\n"
      f"Content:\n{context}\n"
      f"Question:{query}\n"
      f"Answer:"
  )


  print("\n Augumented prompt ",augumented_prompt)
  print("*"*50)


  response = gemini_model.generate_content(augumented_prompt,
                                           generation_config=genai.GenerationConfig(
                                               temperature=0.2,
                                               max_output_tokens=700
                                           ) )
  return response