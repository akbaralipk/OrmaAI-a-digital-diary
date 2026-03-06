import os
import sqlite3
import whisper
import transformers
from datetime import datetime, timedelta
from flask import Flask, render_template, request,session, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from transformers import pipeline
from pydub import AudioSegment
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
print(transformers.__version__)

 
# =============================
# LOAD ENV
# =============================
load_dotenv()

app = Flask(__name__)
app.secret_key = "super_secure_random_key_123_change_this"
DIARY_PASSCODE = "1234"

DB = "diary.db"
VOICE_FOLDER = "voices"
os.makedirs(VOICE_FOLDER, exist_ok=True)

# =============================
# LOAD MODELS
# =============================
print("Loading Whisper...")
whisper_model = whisper.load_model("medium")
print("Whisper Loaded!")

print("Loading Sentiment...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask
from transformers import pipeline



# LOAD HERE
qa_model = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

# print("Loading Generative LLM...")

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# print("LLM Loaded!")

# =============================
# DATABASE INIT
# =============================
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS diary (
        id TEXT PRIMARY KEY,
        date TEXT,
        mood TEXT,
        voice_file TEXT,
        voice_text TEXT,
        text TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# =============================
# GENERATE UNIQUE ID
# =============================
def generate_id():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id FROM diary")
    rows = c.fetchall()
    conn.close()

    if not rows:
        return "abcd1"

    numbers = []

    for r in rows:
        try:
            num = int(r[0].replace("abcd", ""))
            numbers.append(num)
        except:
            pass

    new_number = max(numbers) + 1
    return f"abcd{new_number}"

# =============================
# TRANSCRIBE
# =============================
def transcribe_audio(path):
    result = whisper_model.transcribe(
        path,
        task="translate",
        fp16=False
    )
    return result["text"]

# =============================
# TEXT CLEANING (IMPORTANT FOR RAG)
# =============================
import re

def clean_text(text):
    if not text:
        return ""

    text = text.lower()                # lowercase
    text = text.strip()                # remove spaces
    text = re.sub(r"\s+", " ", text)   # remove extra spaces
    text = re.sub(r"[^\w\s]", "", text) # remove punctuation
    return text
# =============================
# EMBEDDINGS
# =============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

INDEX_FOLDER = r"C:\akbar\AKBARPK\AKBAR\PROJECTS\DIGITAL DIaRY\12.FINAL_DIARY -OG -\faiss_index"  # folder to store FAISS index

# =====================================================
# SYNC FAISS WITH SQLITE (MASTER SYNC FUNCTION)
# =====================================================
def sync_faiss_with_db():

    # Create folder if not exists
    os.makedirs(INDEX_FOLDER, exist_ok=True)

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM diary")
    rows = c.fetchall()
    conn.close()

    documents = []

    for entry in rows:
        content = (
            f"Date: {entry[1]}\n"
            f"Mood: {entry[2]}\n"
            f"Voice: {entry[4]}\n"
            f"Text: {entry[5]}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={"id": entry[0]}
            )
        )

    # If no diaries → remove FAISS index
    if not documents:
        if os.path.exists(os.path.join(INDEX_FOLDER, "index.faiss")):
            for file in os.listdir(INDEX_FOLDER):
                os.remove(os.path.join(INDEX_FOLDER, file))
        return

    # Rebuild index fresh every time
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_FOLDER)
# =============================
# LOGIN PAGE
# =============================
@app.route("/", methods=["GET", "POST"])
def login():
    message = None

    if request.method == "POST":
        entered = request.form["passcode"]

        if entered == DIARY_PASSCODE:
            session["logged_in"] = True
            return redirect("/home")   # your main page
        else:
            message = "Wrong passcode"

    return render_template("login.html", message=message)
# =============================
# HOME PAGE
# =============================
@app.route("/home")
def index():
    if not session.get("logged_in"):
        return redirect("/")
    new_id = generate_id()
    today = datetime.today().date()
    return render_template("index.html", new_id=new_id, today=today)

# =============================
# ADD ENTRY
# =============================
@app.route("/add", methods=["POST"])
def add_entry():
    if not session.get("logged_in"):
        return redirect("/")

    date = request.form["date"]
    text = request.form.get("text", "").strip()
    voice_file = request.files.get("voice")

    if not text and not voice_file:
        flash("Missing data! Either voice or text required.")
        return redirect(url_for("index"))

    unique_id = generate_id()
    voice_filename = ""
    voice_text = ""

    # =============================
    # HANDLE VOICE (WEBM → WAV)
    # =============================
    if voice_file and voice_file.filename != "":

        # Save uploaded webm
        webm_filename = secure_filename(unique_id + ".webm")
        webm_path = os.path.join(VOICE_FOLDER, webm_filename)
        voice_file.save(webm_path)

        # Convert webm → wav
        wav_filename = secure_filename(unique_id + ".wav")
        wav_path = os.path.join(VOICE_FOLDER, wav_filename)

        audio = AudioSegment.from_file(webm_path)
        audio.export(wav_path, format="wav")

        voice_filename = wav_filename

        # Transcribe using Whisper
        voice_text = transcribe_audio(wav_path)

        # delete webm after conversion
        os.remove(webm_path)

    # ✅ ADD THIS BLOCK (VERY IMPORTANT)
    text = clean_text(text)
    voice_text = clean_text(voice_text)

    # =============================
    # PREDICT MOOD
    # =============================
    mood_source = text if text else voice_text
    mood = sentiment_pipeline(mood_source)[0]["label"]

    conn = sqlite3.connect(DB,timeout=10)
    c = conn.cursor()
    c.execute("""
        INSERT INTO diary VALUES (?,?,?,?,?,?)
    """, (unique_id, date, mood, voice_filename, voice_text, text))
    conn.commit()
    conn.close()
    sync_faiss_with_db()

    flash(f"ID {unique_id.upper()} successfully saved!")

    return redirect(url_for("index"))
# =============================
    # VIEW SECTION
# =============================
@app.route("/view")
def view():
    if not session.get("logged_in"):
        return redirect("/")
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM diary ORDER BY date DESC")
    data = c.fetchall()
    conn.close()

    entries_with_voice = []

    for entry in data:
        entry_id = str(entry[0])
        wav_file = os.path.join(VOICE_FOLDER, f"{entry_id}.wav")

        if os.path.exists(wav_file):
            voice_file = f"{entry_id}.wav"  # This is what Flask will serve
        else:
            voice_file = None

        # entry[4] = voice_text
        entries_with_voice.append((entry, voice_file))

    return render_template("view.html", entries=entries_with_voice)
# =============================
# DELETE ENTRY
# =============================
@app.route("/delete/<id>")
def delete_entry(id):
    if not session.get("logged_in"):
        return redirect("/")
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT voice_file FROM diary WHERE id=?", (id,))
    voice = c.fetchone()

    if voice and voice[0]:
        path = os.path.join(VOICE_FOLDER, voice[0])
        if os.path.exists(path):
            os.remove(path)

    c.execute("DELETE FROM diary WHERE id=?", (id,))
    conn.commit()
    conn.close()
    sync_faiss_with_db()

    flash("Diary deleted.")
    return redirect(url_for("view"))

# =============================
# DELETE ALL
# =============================
@app.route("/delete_all")
def delete_all():
    if not session.get("logged_in"):
        return redirect("/")
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("DELETE FROM diary")
    conn.commit()
    conn.close()
    sync_faiss_with_db()

    for file in os.listdir(VOICE_FOLDER):
        os.remove(os.path.join(VOICE_FOLDER, file))

    flash("All diaries deleted.")
    return redirect(url_for("view"))

# =============================
# EDIT ENTRY
# =============================
@app.route("/edit/<id>", methods=["GET", "POST"])
def edit_entry(id):
    if not session.get("logged_in"):
        return redirect("/")

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    if request.method == "POST":

        new_date = request.form["date"]
        new_text = request.form.get("text", "").strip()

        # Get existing voice + voice_text
        c.execute("SELECT voice_file, voice_text FROM diary WHERE id=?", (id,))
        existing = c.fetchone()

        if not existing:
            conn.close()
            flash("Diary not found.")
            return redirect(url_for("view"))

        voice_file, voice_text = existing

        # If both empty → delete entry
        if not new_text and not voice_file:
            c.execute("DELETE FROM diary WHERE id=?", (id,))
            conn.commit()
            conn.close()
            sync_faiss_with_db()
            flash("Both text and voice empty. Diary deleted.")
            return redirect(url_for("view"))

        # Recalculate mood
        mood_source = new_text if new_text else voice_text
        mood = sentiment_pipeline(mood_source)[0]["label"]

        # Update
        c.execute("""
            UPDATE diary
            SET date=?, text=?, mood=?
            WHERE id=?
        """, (new_date, new_text, mood, id))

        conn.commit()
        conn.close()
        sync_faiss_with_db()

        flash(f"{id.upper()} updated successfully!")
        return redirect(url_for("view"))

    # GET REQUEST (load existing data)
    c.execute("SELECT * FROM diary WHERE id=?", (id,))
    entry = c.fetchone()
    conn.close()

    if not entry:
        flash("Diary not found.")
        return redirect(url_for("view"))

    return render_template("edit.html", entry=entry)
   
# =============================
# ASK DIARY (RETRIEVE + QA ANSWER)
# =============================
@app.route("/ask", methods=["GET", "POST"])
def ask():
    if not session.get("logged_in"):
        return redirect("/")

    results = []
    best_diary = None
    message = None
    answer = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        date_input = request.form.get("date", "").strip()  # NEW: get date input

        conn = sqlite3.connect(DB)
        c = conn.cursor()

        # =========================
        # If date is provided → filter by date first
        # =========================
        if date_input:
            try:
                # Convert date input to proper format (YYYY-MM-DD)
                query_date = datetime.strptime(date_input, "%Y-%m-%d").date()
                c.execute("SELECT * FROM diary WHERE date=?", (str(query_date),))
                date_entries = c.fetchall()
                
                if date_entries:
                    results = [(entry, 0) for entry in date_entries]  # score=0 for date filter
                    best_diary = results[0][0]  # optional: first entry as best
                else:
                    message = f"No diary entries found for {query_date}"
                    return render_template(
                        "ask.html",
                        results=[],
                        best_diary=None,
                        message=message,
                        answer=None
                    )
            except ValueError:
                message = "Incorrect date format. Use YYYY-MM-DD."
                return render_template(
                    "ask.html",
                    results=[],
                    best_diary=None,
                    message=message,
                    answer=None
                )
        else:
            # =========================
            # No date → do normal similarity search
            # =========================
            question = clean_text(question)
            faiss_file = os.path.join(INDEX_FOLDER, "index.faiss")

            if not os.path.exists(faiss_file):
                message = "No diary entries available yet."
                return render_template(
                    "ask.html",
                    results=[],
                    best_diary=None,
                    message=message,
                    answer=None
                )

            try:
                faiss_index = FAISS.load_local(
                    INDEX_FOLDER,
                    embeddings,
                    allow_dangerous_deserialization=True
                )

                docs = faiss_index.similarity_search_with_score(question, k=3)

                if not docs:
                    message = "No similar diary found."
                    return render_template(
                        "ask.html",
                        results=[],
                        best_diary=None,
                        message=message,
                        answer=None
                    )

                for doc, score in docs:
                    diary_id = doc.metadata.get("id")
                    c.execute("SELECT * FROM diary WHERE id=?", (diary_id,))
                    entry = c.fetchone()
                    if entry:
                        results.append((entry, score))

                if not results:
                    message = "No similar diary found."
                    return render_template(
                        "ask.html",
                        results=[],
                        best_diary=None,
                        message=message,
                        answer=None
                    )

                results.sort(key=lambda x: x[1])
                best_diary = results[0][0]

            except Exception as e:
                message = f"Error loading search system: {str(e)}"
                return render_template(
                    "ask.html",
                    results=[],
                    best_diary=None,
                    message=message,
                    answer=None
                )

        conn.close()

        # =========================
        # Generate answer if question exists
        # =========================
        if question and best_diary:
            context = f"""
            Date: {best_diary[1]}
            Mood: {best_diary[2]}
            Text: {best_diary[5]}
            Voice: {best_diary[4] if len(best_diary) > 4 else ''}
            Recorder: {best_diary[3] if len(best_diary) > 3 else ''}
            """
            qa_result = qa_model(
                question=question,
                context=context
            )
            answer = qa_result["answer"]

    return render_template(
        "ask.html",
        results=results,
        best_diary=best_diary,
        message=message,
        answer=answer
    )
# =============================
# VOICE SERVE
# =============================
@app.route("/voices/<filename>")
def serve_voice(filename):
    if not session.get("logged_in"):
        return redirect("/")
    return send_from_directory(VOICE_FOLDER, filename)

if __name__ == "__main__":
    sync_faiss_with_db()
    app.run(debug=True, use_reloader=False)