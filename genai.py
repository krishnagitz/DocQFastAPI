# imports
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from data import wikipedia_article_on_curling

openai.api_key = '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'
# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# an example question about the 2022 Olympics
# query = 'Which athletes won the silver medal in curling at the 2022 Winter Olympics?'

# response = openai.ChatCompletion.create(
#     messages=[
#         {'role': 'system', 'content': 'You answer questions about the 2022 Winter Olympics.'},
#         {'role': 'user', 'content': query},
#     ],
#     model=GPT_MODEL,
#     temperature=0,
# )

# print(response['choices'][0]['message']['content'])

query = f"""Use the below article on the 2022 Winter Olympics to answer the subsequent question. 
If the answer cannot be found, write "I don't know."

Article:
\"\"\"
{wikipedia_article_on_curling}
\"\"\"

Question: Did india has won any medals? if not why pls?"""

response = openai.ChatCompletion.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2022 Winter Olympics.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODEL,
    temperature=0,
)

print(response['choices'][0]['message']['content'])