import json
import re
from langchain.text_splitter import MarkdownHeaderTextSplitter
import base64
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

REPO_OWNER = 'microsoft'
REPO_NAME = 'vscode-docs'


def get_markdown_files_from_github(path=''):
    url = f'https://api.github.com/repos/{
        REPO_OWNER}/{REPO_NAME}/contents/{path}'
    headers = {
        'Authorization': f'token {os.getenv("GITHUB_TOKEN")}'
    }
    response = requests.get(url, headers=headers)
    markdown_files = []

    if response.status_code == 200:
        files = response.json()

        for file in files:
            if file['type'] == 'dir':
                markdown_files += get_markdown_files_from_github(file['path'])
            elif file['type'] == 'file' and file['name'].endswith('.md'):
                markdown_files.append(file['path'])

        return markdown_files
    else:
        raise Exception(f"Error fetching repo contents: {
                        response.status_code}")


# Test: Print all markdown files in the learn/students folder
print(get_markdown_files_from_github("learn/students"))


def get_file_content(file_path):
    url = f'https://api.github.com/repos/{
        REPO_OWNER}/{REPO_NAME}/contents/{file_path}'
    headers = {
        'Authorization': f'token {os.getenv("GITHUB_TOKEN")}'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        file_content = base64.b64decode(
            response.json()['content']).decode('utf-8')
        title = next(line.replace('# ', '')
                     for line in file_content.split('\n') if line.startswith('#'))
        description = next((line.split(':', 1)[1].strip() for line in file_content.split(
            '\n') if line.lower().startswith('metadescription:')), '')
        content = file_content.split(title, 1)[1].strip()
        content = content.replace('<!-- DOWNLOAD_LINKS_PLACEHOLDER -->', '')
        full_content = f"{description}\n\n{content}"
        return {
            'content': full_content,  # rate limit could be exceeded for free tier
            'url': f'https://github.com/{REPO_OWNER}/{REPO_NAME}/blob/main/{file_path}',
        }
    else:
        raise Exception(f"Error fetching file content: {response.status_code}")


# Test: Print a sample file content
get_file_content("release-notes/v1_94.md")

# Iterate over each file in `release-notes` and get the content
docs_files = get_markdown_files_from_github("release-notes")

docs_contents = []
for file_path in docs_files:
    content = get_file_content(file_path)
    docs_contents.append(content)

# Print the first few contents to verify
for doc in docs_contents[:5]:
    print(doc)
    print('---')

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
]

# Initialize the MarkdownHeaderTextSplitter with the headers to split on
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

# Split each document content and retain the URL
split_docs = []
for doc in docs_contents:
    split_content = markdown_splitter.split_text(doc['content'])
    first_line = doc['content'].split('\n', 1)[0]
    for split in split_content:
        if 'Header 3' in split.metadata:
            header_text = split.metadata['Header 3'].replace(' ', '-').lower()
            new_url = re.sub(r'https://github.com/microsoft/vscode-docs/blob/main/release-notes/(.*)\.md',
                             r'https://code.visualstudio.com/updates/\1', doc['url'])
            split_docs.append({
                'content': f"{first_line}\n\n{split.page_content}",
                'url': f"{new_url}#{header_text}",
            })

# Print the first few split documents to verify
for split_doc in split_docs[:5]:
    print(split_doc)
    print('---')

# Write split_docs_dicts to a JSON file
with open('split_docs_contents.json', 'w') as f:
    json.dump(split_docs, f, indent=4)
