from flask import Flask, render_template, request, Response 
from flask import make_response
from hparams import hparams, hparams_debug_string
import os
from synthesizer import Synthesizer

from scipy.io import wavfile
import io
import wave

application = Flask(__name__)

@application.route("/")
def home():
    return render_template('home.html')

@application.route("/results", methods=['POST'])
def results():
    synthesizer = Synthesizer()
    synthesizer.load('./tmp/tacotron-20180906/model.ckpt')
    
    text = request.form['text']
    sentences = text.split('.')
    for idx, sent in enumerate(sentences):
        sentences[idx] = io.BytesIO(synthesizer.synthesize(sent))

    params_set = False
    temp_file = io.BytesIO()
    with wave.open(temp_file, 'wb') as temp_input:
        for sent in sentences:
            with wave.open(sent, 'rb') as w:
                if not params_set:
                    temp_input.setparams(w.getparams())
                    params_set = True
                temp_input.writeframes(w.readframes(w.getnframes()))
    
    response = make_response(temp_file.getvalue())
    temp_file.close()
    
    response.headers['Content-Type'] = 'audio/wav'
    response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'

    return response

if __name__ == '__main__':
    #synthesizer = Synthesizer()
    #synthesizer.load('./tmp/tacotron-20180906/model.ckpt')
    application.run(debug=True)
