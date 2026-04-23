import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import cv2
import numpy as np
import time
import os
import subprocess
import sys

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTRO_PATH = os.path.join(BASE_DIR, "intro.txt")

# ---------------- SPEAK (SAPI5 – STABLE) ----------------
def speak(text):
    print("Jarvis:", text)
    engine = pyttsx3.init(driverName="sapi5")
    engine.setProperty("rate", 170)
    engine.setProperty("volume", 1.0)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    time.sleep(0.3)

# ---------------- SPEECH RECOGNIZER ----------------
r = sr.Recognizer()
r.pause_threshold = 0.8
r.energy_threshold = 300

try:
    with sr.Microphone() as source:
        print("Calibrating microphone...")
        r.adjust_for_ambient_noise(source, duration=1)
except Exception as e:
    print(f"⚠️ Warning: No microphone detected. Voice commands will be disabled. Error: {e}")

def take_command():
    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source, timeout=3, phrase_time_limit=5)
            command = r.recognize_google(audio)
            print("User:", command)
            return command.lower()
    except Exception:
        return ""

# ---------------- JARVIS UI ----------------
def jarvis_ui(text="JARVIS ONLINE"):
    img = np.zeros((500, 800, 3), np.uint8)
    cv2.putText(img, text, (50, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("JARVIS", img)
    cv2.waitKey(1)

# ---------------- STREAMLIT (SAFE LAUNCH) ----------------
streamlit_started = False

def open_resume_analyser():
    global streamlit_started
    if not streamlit_started:
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        streamlit_started = True

# ---------------- START ----------------
speak("Jarvis activated. Hi Dhaksha. How can I help you?")
jarvis_ui("JARVIS ACTIVATED")

# ---------------- MAIN LOOP ----------------
while True:
    command = take_command()

    if command == "":
        continue

    # EXIT
    if "bye" in command or "exit" in command:
        speak("Goodbye boss")
        jarvis_ui("SYSTEM SHUTDOWN")
        break

    # INTRODUCTION (TOP PRIORITY)
    elif any(x in command for x in [
        "intro",
        "introduction",
        "introduce",
        "introduce myself"
    ]):
        if os.path.exists(INTRO_PATH):
            with open(INTRO_PATH, "r", encoding="utf-8") as f:
                intro = f.read()

            speak("Here is your self introduction")
            speak(intro)
            jarvis_ui("SELF INTRODUCTION")
        else:
            speak("Resume not analyzed yet")
            jarvis_ui("NO RESUME FOUND")

    # RESUME PROJECT
    elif "project" in command or "resume" in command or "analyze" in command:
        speak("Opening resume analyser")
        open_resume_analyser()
        speak("Upload resume and then say introduction")
        jarvis_ui("RESUME ANALYSER")

    # TIME
    elif "time" in command:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        speak("Current time is " + current_time)
        jarvis_ui("TIME: " + current_time)
    elif "open find" in command:
        speak("What should I search?")
        query = take_command()
        webbrowser.open(f"https://www.google.com/search?q={query}")
        jarvis_ui("SEARCHING: " + query)

    # YOUTUBE
    elif "youtube" in command:
        speak("Opening YouTube")
        webbrowser.open("https://youtube.com")
        jarvis_ui("OPENING YOUTUBE")

    # GOOGLE
    elif "google" in command:
        speak("Opening Google")
        webbrowser.open("https://google.com")
        jarvis_ui("OPENING GOOGLE")

    # GITHUB
    elif "github" in command:
        speak("Opening GitHub")
        webbrowser.open("https://github.com")
        jarvis_ui("OPENING GITHUB")

    else:
        speak("Sorry, I did not understand")
        jarvis_ui("UNKNOWN COMMAND")

cv2.destroyAllWindows()
