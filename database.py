import sqlite3

DB_NAME = "rag_knowledge_base.db"
TABLE_NAME = 'documents_demo'

def setup_database(record):
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE_NAME} (content TEXT not null)""")
  cursor.execute(f"INSERT INTO {TABLE_NAME}  VALUES ('{record}')")
  conn.commit()
  # print(list(cursor.execute(f"SELECT * from {TABLE_NAME}")))
  conn.close()
def cleanup():
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute(f"""DELETE FROM  {TABLE_NAME} """)
  conn.commit()
  conn.close()
def display_records():
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  records = list(cursor.execute(f"SELECT * from {TABLE_NAME}"))
  return records


cleanup()
for i in knowledge_base:
  setup_database(i)