# Core and FastAPI
from typing import Annotated
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import PlainTextResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from desc import solvers_descriptions
import solvers
# File and OS operations
import os
import subprocess
import json
import re
import base64

# Data processing
import numpy as np
from dateutil.parser import parse
import csv

# Database
import sqlite3
import duckdb  # Optional for DuckDB support

# Web and API
import requests
import aiohttp
from bs4 import BeautifulSoup  # For web scraping

# Image processing
from PIL import Image  # For image manipulation


# Markdown processing
import markdown  # For Markdown to HTML conversion

# Git operations
from git import Repo  # For cloning and committing to Git repositories

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
SCRIPT_RUNNER = {
        "type": "function",
        "function": {
            "name": "script_runner",
            "description": "Install a package and run a script url with provided arguments",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_url": {
                        "type": "string",
                        "description": "The url of the script to run.",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of arguments to pass to the script.",
                    },
                },
                "required": ["script_url", "args"],
            },
        },
    }

FORMAT_MARKDOWN =  {
        "type": "function",
        "function": {
            "name": "format_markdown",
            "description": "Format the contents of a markdown file using prettier@3.4.2, updating the file in-place",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the markdown file to format.",
                    },
                },
                "required": ["file_path"],
            },
        },
    }

COUNT_WEDNESDAYS = {
        "type": "function",
        "function": {
            "name": "count_wednesdays",
            "description": "Count the number of Wednesdays in a file which contains one date per line and write only the count to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "The path to the file containing the list of dates.",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the count of Wednesdays will be written.",
                    },
                },
                "required": ["input_file", "output_file"],
            },
        },
    }

SORT_CONTACTS = {
        "type": "function",
        "function": {
            "name": "sort_contacts",
            "description": "Sort the array of contacts by last_name, then first_name, and write the result to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "The path to the file containing the contacts.",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the sorted contacts will be written.",
                    },
                },
                "required": ["input_file", "output_file"],
            },
        },
    }

EXTRACT_FIRST_LINES = {
        "type": "function",
        "function": {
            "name": "extract_first_lines",
            "description": "Write the first line of the 10 most recent .log files to a file, most recent first",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_dir": {
                        "type": "string",
                        "description": "The directory containing the log files.",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the first lines will be written.",
                    },
                },
                "required": ["log_dir", "output_file"],
            },
        },
    }

CREATE_MARKDOWN_INDEX =   {
        "type": "function",
        "function": {
            "name": "create_markdown_index",
            "description": "Create an index file mapping Markdown filenames to their titles",
            "parameters": {
                "type": "object",
                "properties": {
                    "docs_dir": {
                        "type": "string",
                        "description": "The directory containing the Markdown files.",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the index will be written.",
                    },
                },
                "required": ["docs_dir", "output_file"],
            },
        },
    }

EXTRACT_SENDER_EMAIL = {
        "type": "function",
        "function": {
            "name": "extract_sender_email",
            "description": "Extract the sender's email address from an email message and write it to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "The path to the file containing the email message.",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the sender's email will be written.",
                    },
                },
                "required": ["input_file", "output_file"],
            },
        },
    }

EXTRACT_CREDIT_CARD_NUMBER = {
        "type": "function",
        "function": {
            "name": "extract_credit_card_number",
            "description": "Extract the credit card number from an image and write it to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "The path to the image file containing the credit card number.",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the credit card number will be written.",
                    },
                },
                "required": ["input_file", "output_file"],
            },
        },
    }

FIND_SIMILAR_COMMENTS = {
        "type": "function",
        "function": {
            "name": "find_similar_comments",
            "description": "Find the most similar pair of comments using embeddings and write them to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "The path to the file containing the comments.",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the most similar comments will be written.",
                    },
                },
                "required": ["input_file", "output_file"],
            },
        },
    }

CALCULATE_GOLD_TICKET_SALES = {
        "type": "function",
        "function": {
            "name": "calculate_gold_ticket_sales",
            "description": "Calculate the total sales of all items in the 'Gold' ticket type and write the result to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_file": {
                        "type": "string",
                        "description": "The path to the SQLite database file.",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the total sales will be written.",
                    },
                },
                "required": ["db_file", "output_file"],
            },
        },
}

FETCH_DATA_FROM_API = {
    "type": "function",
    "function": {
        "name": "fetch_data_from_api",
        "description": "Fetch data from an API and save it to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "api_url": {
                    "type": "string",
                    "description": "The URL of the API to fetch data from.",
                },
                "output_file": {
                    "type": "string",
                    "description": "The path to the file where the data will be saved.",
                },
            },
            "required": ["api_url", "output_file"],
        },
    },
}

CLONE_GIT_REPO = {
    "type": "function",
    "function": {
        "name": "clone_git_repo",
        "description": "Clone a git repository and make a commit",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_url": {
                    "type": "string",
                    "description": "The URL of the git repository to clone.",
                },
                "repo_dir": {
                    "type": "string",
                    "description": "The directory where the repository will be cloned.",
                },
                "commit_message": {
                    "type": "string",
                    "description": "The commit message to use.",
                },
            },
            "required": ["repo_url", "repo_dir", "commit_message"],
        },
    },
}

RUN_SQL_QUERY = {
    "type": "function",
    "function": {
        "name": "run_sql_query",
        "description": "Run a SQL query on a SQLite or DuckDB database and save the result to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "db_file": {
                    "type": "string",
                    "description": "The path to the SQLite or DuckDB database file.",
                },
                "query": {
                    "type": "string",
                    "description": "The SQL query to execute.",
                },
                "output_file": {
                    "type": "string",
                    "description": "The path to the file where the query result will be saved.",
                },
            },
            "required": ["db_file", "query", "output_file"],
        },
    },
}

SCRAPE_WEBSITE = {
    "type": "function",
    "function": {
        "name": "scrape_website",
        "description": "Extract data from a website (i.e., scrape) and save it to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the website to scrape.",
                },
                "output_file": {
                    "type": "string",
                    "description": "The path to the file where the scraped data will be saved.",
                },
            },
            "required": ["url", "output_file"],
        },
    },
}

COMPRESS_OR_RESIZE_IMAGE = {
    "type": "function",
    "function": {
        "name": "compress_or_resize_image",
        "description": "Compress or resize an image and save it to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The path to the input image file.",
                },
                "output_file": {
                    "type": "string",
                    "description": "The path to the file where the compressed/resized image will be saved.",
                },
                "size": {
                    "type": "string",
                    "description": "The size to resize the image to (e.g., '800x600').",
                    "optional": True,
                },
                "quality": {
                    "type": "integer",
                    "description": "The quality of the compressed image (e.g., 85).",
                    "optional": True,
                },
            },
            "required": ["input_file", "output_file"],
        },
    },
}

TRANSCRIBE_AUDIO = {
    "type": "function",
    "function": {
        "name": "transcribe_audio",
        "description": "Transcribe audio from an MP3 file and save the transcription to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The path to the input MP3 file.",
                },
                "output_file": {
                    "type": "string",
                    "description": "The path to the file where the transcription will be saved.",
                },
            },
            "required": ["input_file", "output_file"],
        },
    },
}

CONVERT_MARKDOWN_TO_HTML = {
    "type": "function",
    "function": {
        "name": "convert_markdown_to_html",
        "description": "Convert a Markdown file to HTML and save it to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The path to the input Markdown file.",
                },
                "output_file": {
                    "type": "string",
                    "description": "The path to the file where the HTML will be saved.",
                },
            },
            "required": ["input_file", "output_file"],
        },
    },
}

FILTER_CSV = {
    "type": "function",
    "function": {
        "name": "filter_csv",
        "description": "Filter a CSV file and return JSON data",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The path to the input CSV file.",
                },
                "filter_column": {
                    "type": "string",
                    "description": "The column to filter by.",
                },
                "filter_value": {
                    "type": "string",
                    "description": "The value to filter for.",
                },
            },
            "required": ["input_file", "filter_column", "filter_value"],
        },
    },
}

tools = list(solvers_descriptions.values())
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")


@app.get("/")
def hello():
    return {"Hello": "World"}


@app.get("/read", response_class=PlainTextResponse)
def read_file(path: str):
    try:
        with open(path, "r") as file:
            content = file.read()
        return content
    except Exception as e:
        raise HTTPException(status_code=404, detail="File doesn't exists")
   
@app.post("/api")
async def task_runner(question: Annotated[str, Form()]):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}

    # Prepare the data for the LLM
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": question},
            {
                "role": "system",
                "content": """ 
                You are an assistant who has to do variety of tasks. 
                Analyze the task and choose the appropriate tool from the available tools.
                """,
            },
        ],
        "tools": tools,
        "tool_choice": "auto",  # Let the LLM decide which tool to use
    }

    # Send the task to the LLM for analysis
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        # Extract the tool call from the LLM's response
        tool_call = response.json()['choices'][0]['message']['tool_calls'][0]
        print('tool_call:', tool_call)
        function_name = tool_call['function']['name']
        function_args = json.loads(tool_call['function']['arguments'])
        print('function_name:', function_name)
        func = getattr(solvers, function_name)
        result = func(**function_args)
        return {"answer": result}
    else:
        print(response.json())
        raise HTTPException(status_code=response.status_code, detail="Failed to process task")
    
# Security
def enforce_security_checks(task: str):
    # B1: Ensure no paths outside /data are accessed
    if re.search(r"(?<!\/data)\/[^\/]+", task):
        raise HTTPException(status_code=400, detail="Access to data outside /data is not allowed.")

    # B2: Ensure no deletion operations are performed
    if "delete" in task.lower() or "remove" in task.lower():
        raise HTTPException(status_code=400, detail="Deletion of data is not allowed.")



# Function Implementations

def script_runner(script_url: str, args: list):
    try:
        email = args[0]
        command = ['uv', 'run', script_url, email]
        p = subprocess.run(command, capture_output=True)
        print(p.stdout)
        print(p.stderr)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run script: {e}")

def format_markdown(file_path: str):
    print(file_path)
    try:
        p = subprocess.run(["npx", "prettier@3.4.2", "--write", file_path], check=True, capture_output=True)
        print(p.stdout)
        print(p.stderr)
        print(p.returncode)
        return {"status": "success", "message": f"Formatted {file_path}"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to format markdown: {e}")

def count_wednesdays(input_file: str, output_file: str):
    try:                                                                                                                                                                                                         
        with open(input_file, "r") as file:
            dates = file.readlines()
        wednesdays = sum(1 for date in dates if date.strip() and parse(date.strip()).weekday() == 2)
        with open(output_file, "w") as file:
            file.write(str(wednesdays))
        return {"status": "success", "message": f"Counted {wednesdays} Wednesdays"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to count Wednesdays: {e}")

def sort_contacts(input_file: str, output_file: str):
    try:
        with open(input_file, "r") as file:
            contacts = json.load(file)
        contacts.sort(key=lambda x: (x["last_name"], x["first_name"]))
        with open(output_file, "w") as file:
            json.dump(contacts, file, indent=2)
        return {"status": "success", "message": f"Sorted contacts written to {output_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sort contacts: {e}")

def extract_first_lines(log_dir: str, output_file: str):
    try:
        log_files = sorted(
            [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".log")],
            key=os.path.getmtime,
            reverse=True,
        )[:10]
        first_lines = []
        for log_file in log_files:
            with open(log_file, "r") as file:
                first_lines.append(file.readline().strip())
        with open(output_file, "w") as file:
            file.write("\n".join(first_lines))
        return {"status": "success", "message": f"Extracted first lines to {output_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract first lines: {e}")

def create_markdown_index(docs_dir: str, output_file: str):
    """
    Create a JSON index of markdown files and their H1 titles.
    
    Args:
        docs_dir: Base directory containing markdown files
        output_file: Path to output JSON file
    """
    try:
        index = {}
        # Walk through all files in the directory
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    # Get relative path without the docs_dir prefix
                    rel_path = os.path.relpath(file_path, docs_dir)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            # Look for the first H1 header
                            if line.strip().startswith('# '):
                                index[rel_path] = line[2:].strip()
                                break
        
        # Sort the dictionary case-insensitively
        sorted_index = dict(sorted(index.items(), key=lambda x: x[0].lower()))
        
        # Write the index to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_index, f, indent=2)
        return {"status": "success", "message": f"Created markdown index at {output_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create markdown index: {e}")

def extract_sender_email(input_file: str, output_file: str):
    try:
        # Read the email content from the input file
        with open(input_file, "r") as file:
            email_content = file.read()
        
        # Use the task_runner function to process the task
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Extract the sender's email address from the email."},
                {"role": "user", "content": email_content},
            ],
        }

        # Send the task to the AI Proxy
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            # Extract the sender's email from the response
            sender_email = response.json()["choices"][0]["message"]["content"].strip()
            
            # Write the sender's email to the output file
            with open(output_file, "w") as file:
                file.write(sender_email)
            
            return {"status": "success", "message": f"Extracted sender email to {output_file}"}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to extract sender email")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract sender email: {e}")

def extract_credit_card_number(input_file: str, output_file: str):
    try:
        # Read the image file as base64
        with open(input_file, "rb") as file:
            image_data = file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Use the AI Proxy chat/completions endpoint
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Extract numeric sequences from image."},
                {"role": "user", "content": f"data:image/png;base64,{image_base64}"},
            ],
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            credit_card_number = response.json()["choices"][0]["message"]["content"].strip()
            with open(output_file, "w") as file:
                file.write(credit_card_number)
            return {"status": "success", "message": f"Extracted credit card number to {output_file}"}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to extract credit card number")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract credit card number: {e}")

def find_similar_comments(input_file: str, output_file: str):
    try:
        with open(input_file, "r") as file:
            comments = file.readlines()
        # Use the AI Proxy embeddings endpoint
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
            headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
            json={"model": "text-embedding-3-small", "input": comments}
        )
        if response.status_code == 200:
            embeddings = np.array([item["embedding"] for item in response.json()["data"]])
            similarity_matrix = np.dot(embeddings, embeddings.T)
            np.fill_diagonal(similarity_matrix, -np.inf)
            i, j = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
            similar_comments = "\n".join(sorted([comments[i].strip(), comments[j].strip()]))
            with open(output_file, "w") as file:
                file.write(similar_comments)
            return {"status": "success", "message": f"Found similar comments at {output_file}"}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to get embeddings")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find similar comments: {e}")

def calculate_gold_ticket_sales(db_file: str, output_file: str):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
        total_sales = cursor.fetchone()[0]
        with open(output_file, "w") as file:
            file.write(str(total_sales))
        return {"status": "success", "message": f"Calculated gold ticket sales: {total_sales}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate gold ticket sales: {e}")

# Phase B
# Task B3: Fetch data from an API and save it
def fetch_data_from_api(api_url: str, output_file: str):
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            with open(output_file, "w") as file:
                file.write(response.text)
            return {"status": "success", "message": f"Data saved to {output_file}"}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch data from API")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {e}")

# Task B4: Clone a git repo and make a commit
def clone_git_repo(repo_url: str, repo_dir: str, commit_message: str):
    try:
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        subprocess.run(["git", "-C", repo_dir, "add", "."], check=True)
        subprocess.run(["git", "-C", repo_dir, "commit", "-m", commit_message], check=True)
        return {"status": "success", "message": f"Repository cloned and commit made in {repo_dir}"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to clone or commit: {e}")

# Task B5: Run a SQL query on a SQLite or DuckDB database
def run_sql_query(db_file: str, query: str, output_file: str):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        with open(output_file, "w") as file:
            for row in result:
                file.write(f"{row}\n")
        return {"status": "success", "message": f"Query result saved to {output_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run SQL query: {e}")

# Task B6: Extract data from a website (scrape)
def scrape_website(url: str, output_file: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            extracted_data = soup.get_text()
            with open(output_file, "w") as file:
                file.write(extracted_data)
            return {"status": "success", "message": f"Data saved to {output_file}"}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch website content")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape website: {e}")

# Task B7: Compress or resize an image
def compress_or_resize_image(input_file: str, output_file: str, size: str = None, quality: int = 85):
    try:
        image = Image.open(input_file)
        if size:
            width, height = map(int, size.split("x"))
            image = image.resize((width, height))
        image.save(output_file, quality=quality)
        return {"status": "success", "message": f"Image saved to {output_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compress or resize image: {e}")

# Task B8: Transcribe audio from an MP3 file
async def transcribe_audio(input_file: str, output_file: str):
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(input_file) as source:
            audio = recognizer.record(source)
            transcription = recognizer.recognize_google(audio)
        with open(output_file, "w") as file:
            file.write(transcription)
        return {"status": "success", "message": f"Transcription saved to {output_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {e}")

# Task B9: Convert Markdown to HTML
async def convert_markdown_to_html(input_file: str, output_file: str):
    try:
        with open(input_file, "r") as file:
            markdown_content = file.read()
        html_content = markdown.markdown(markdown_content)
        with open(output_file, "w") as file:
            file.write(html_content)
        return {"status": "success", "message": f"HTML saved to {output_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to convert Markdown to HTML: {e}")

# Task B10: Filter a CSV file and return JSON data
async def filter_csv(input_file: str, filter_column: str, filter_value: str):
    try:
        filtered_rows = []
        with open(input_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row[filter_column] == filter_value:
                    filtered_rows.append(row)
        return {"status": "success", "data": filtered_rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to filter CSV: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
