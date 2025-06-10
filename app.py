import os
from dotenv import load_dotenv
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from vanna.flask import VannaFlaskApp

# Load environment variables from .env file
load_dotenv('.env')


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)


# Get the OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize Vanna with proper configuration
vn = MyVanna(config={
    'api_key': openai_api_key,
    'model': 'gpt-4o'
})

if openai_api_key:
    print("✅ OpenAI API key loaded successfully")
else:
    print("❌ OpenAI API key not found. Please check your .env file")

vn.connect_to_postgres(
    host=os.getenv('DB_HOST'),
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    port=int(os.getenv('DB_PORT'))
)

# Uncomment the following lines on the first run only
# df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
# plan = vn.get_training_plan_generic(df_information_schema)
# print(plan)
# vn.train(plan=plan)

app = VannaFlaskApp(vn)
app.run()
