
# OrmaAI – AI Powered Digital Diary

OrmaAI is an AI-powered digital diary that goes beyond traditional journaling.
Instead of simply storing memories, the system can analyze, understand, and interact with past experiences using Artificial Intelligence.

Users can write or speak their daily experiences, and the system processes those entries to provide insights and intelligent responses.

## Project Overview

This application allows users to maintain digital diary entries while integrating several AI-based capabilities such as speech recognition, sentiment analysis, and question answering from stored memories.

The goal of this project is to demonstrate how AI and Natural Language Processing can be integrated into a real-world application that interacts with personal data.

## Key Features
### Diary Entry Management

Store daily diary entries

Each entry includes:

Date

Mood

Diary text

### Voice-to-Text Input

Users can record their experiences through voice input.
Speech is converted to text using a speech recognition model.

### Sentiment Analysis

The system analyzes diary entries to detect emotional tone, such as:

Positive

Neutral

Negative

### Question Answering from Diary

Users can ask questions about their past entries.

Example questions:

"When did I visit Munnar?"

"What did I do last summer?"

The system retrieves relevant entries and generates answers using a Retrieval-Augmented Generation approach.

### Intelligent Memory Retrieval

Natural Language Processing models are used to locate the most relevant diary entries based on user queries.

## Technologies Used

Python
Flask (Backend framework)
HuggingFace Transformers
Whisper (Speech-to-Text)
SQLite Database
Retrieval-Augmented Generation (RAG)
Natural Language Processing

## System Workflow

1.User writes or records a diary entry.

2.The entry is stored in the SQLite database.

3.AI models analyze the entry for sentiment and content.

4.When the user asks a question, the system retrieves relevant entries.

5.The model generates an answer based on the retrieved information.


## OrmaAI-a-digital-diary     

│

├── app.py

├── requirements.txt

├── templates/

├── static/

├── demo_video.mp4

└── README.md

