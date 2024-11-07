from rank_bm25 import BM25Okapi
import pandas as pd
from openai import OpenAI
import os
import json
from bs4 import BeautifulSoup
import requests

# Fetch the content from the URL
url = 'https://microsoftfabric.devpost.com/'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract the content of the blog post
blog_post_content = soup.get_text()

# Convert the content to JSON
content = json.dumps({"content": blog_post_content})

print(content)

# Messages to give LLM, to create a short LinkedIn post based on a blog post
system_message = """
You are a social assistant who writes creative content. You will politely decline any other requests from the user not related to creating content. Don't talk about a single VS Code release and don't talk about release dates at all. Instead, only talk about the relevant features. Don't include made up links, but do provide real links to the VS Code release notes for specific features. You format all your responses as Markdown unless otherwise specified. Avoid wrapping your entire response in a markdown code element.
"""
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": f"Create a very short LinkedIn post using the following: {content}"}
]

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

response = client.chat.completions.create(
    messages=messages,
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000,
    model=model_name
)

print(response.choices[0].message.content)

# Messages to give LLM, to extract key topics & features
topic_system_message = """
You are an expert at conducting entity extraction. Generate top topics and functionality based on provided content. Focus on identifying key concepts, themes, and relevant terms related to specific developer tooling, with a particular emphasis on VS Code features. Make sure entities you extract are directly relevant to the developer environment described. Don't mention specific dates or years. Use advanced search techniques, including Boolean operators and keyword variations, to craft precise, optimized queries that yield the most relevant results. Aim for clarity, relevance, and depth to cover all aspects of the topic efficiently. Simply list the phrases without additional explanation or details. Do not list any bullet points or numbered lists or quotation marks.
"""

topic_user_message = "Come up with a list of top 5 developer tooling topics, functionalities, and relevant terms, with a strong focus on VS Code features and integrations based on the following content: "


def extract_key_topics(content, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": topic_system_message},
        {"role": "user", "content": topic_user_message+content}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )

    key_topics = response.choices[0].message.content.split('\n')
    return key_topics


key_topics = extract_key_topics(content)
print("\n".join([topic + "\n" for topic in key_topics]))

# Load & filter data
# load split_docs_contents.json as a dataframe
df = pd.read_json('split_docs_contents.json')
df.head()

# Filter rows based on column: 'content'
df_clean = df[(df['content'].str.contains("2023", regex=False, na=False)) | (
    df['content'].str.contains("2024", regex=False, na=False))]


def bm25_search(df_clean, key_topics, top_n=10):
    # Tokenize the content of each document
    tokenized_corpus = [doc.split(" ") for doc in df_clean['content']]

    # Initialize BM25
    bm25 = BM25Okapi(tokenized_corpus)

    # Combine key topics into a single query string
    query = " ".join(key_topics)
    tokenized_query = query.split(" ")

    # Get BM25 scores for the query
    scores = bm25.get_scores(tokenized_query)

    # Get the indices of the top_n documents
    top_n_indices = scores.argsort()[-top_n:][::-1]

    # Retrieve the top_n documents
    top_n_docs = df_clean.iloc[top_n_indices]

    return top_n_docs


# Perform the search and output the top 10 documents
top_10_docs = bm25_search(df_clean, key_topics, top_n=10)
print(top_10_docs)

# Messages to give LLM, to re-rank the documents based on semantic relevance
rerank_system_message = """
You are tasked with re-ranking a set of documents based on their relevance to given search queries. The documents have already been retrieved based on initial search criteria, but your role is to refine the ranking by considering factors such as semantic similarity to the query, context relevance, and alignment with the user's intent. Focus on documents that provide concise, high-quality information, ensuring that the top-ranked documents answer the query as accurately and completely as possible. If you can't rank them based on semantic relevance, give higher rank to documents with VS Code features that were published most recently. Make sure to return the full name of the feature and URL of each release note document, and format your response as a Markdown list item, with the URL in parentheses. Do not include any additional information or commentary about the documents. List a variety of documents, and give more weight to documents that mention Python and or notebooks features. Only return the top 3 documents and reference them by the feature name, not the release version or date.
"""

rerank_user_message = f"Here are some documents: {top_10_docs.to_json(
    orient='records')}. Re-rank those documents based on these key VS Code functionalities: {key_topics}. Only return the top 3 documents."


def rerank_documents(model="gpt-4o-mini"):
    # Truncate the user message to fit within the token limit
    max_length = 7500  # Adjust this value as needed to fit within the token limit
    truncated_user_message = rerank_user_message[:max_length]

    messages = [
        {"role": "system", "content": rerank_system_message},
        {"role": "user", "content": truncated_user_message}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )

    reranked_documents = response.choices[0].message.content.split('\n')
    return reranked_documents


reranked_documents = rerank_documents()
print("\n".join([doc + "\n" for doc in reranked_documents]))


def generate_llm_answer(content, context, completion_model):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content":  f"Create a very short LinkedIn post using the following content: {
            content}. Also, include the following established VS Code features along with their URLs in your response, so folks seeing the post can try them out: {context}."}
    ]

    response = client.chat.completions.create(
        model=completion_model,
        messages=messages,
        temperature=0.3
    )

    answer = response.choices[0].message.content
    return answer


print(generate_llm_answer(content, reranked_documents, completion_model="gpt-4o-mini"))
print(generate_llm_answer(content, reranked_documents,
      completion_model="Mistral-small"))
print(generate_llm_answer(content, reranked_documents,
      completion_model="meta-llama-3-8b-instruct"))
