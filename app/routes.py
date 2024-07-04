from flask import request, render_template
from .chatbot import get_response
from .sentiment_analysis import analyze_sentiment
from .translation import translate_text
import os
import speech_recognition as sr
from gtts import gTTS

r = sr.Recognizer()

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("start response.mp3")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['audio']
        audio = sr.AudioFile(file)
        with audio as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            response_text = get_response(text)
            text_to_speech(response_text)
            return render_template('index.html', response=response_text)
    return render_template('index.html')
