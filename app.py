from flask import Flask, render_template, Response, request, redirect, url_for
import os
import subprocess
import shutil
import time
import numpy as np
import base64
import argparse

# create the flask app

app = Flask(__name__)


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


# what html should be loaded as the home page when the app loads?

@app.route('/')
def index():
    return redirect(url_for('home'))


@app.route('/home')
def home():
    return render_template('simple_template_2.html')


# define the logic for reading the inputs from the WEB PAGE,
# running the model, and displaying the prediction

@app.route('/tts', methods=['GET', 'POST'])
def predict():
    message = request.form.get('input_text')
    if message is None:
        return redirect((url_for("home")))
    message = message.strip() + "\n"
    name = request.form.get("input_name")
    video_name = request.form.get("input_video")
    start = time.time()
    try:
        tts_en_proc.stdin.write(message)
    except Exception as e:
        print(f"E: {e}")
    print("!1111!!")
    tts_en_proc.stdin.flush()
    print("ASD")
    tts_wav = ""
    for line in tts_en_proc.stdout:
        if not "ADC" in line: print(line)
        if line.strip().endswith(".wav"):
            print(line)

        if line.strip().startswith("ADC"):
            tts_wav = line[4:].strip()
            break
    tts_time = time.time()
    print("HHHHHHHHH")
    voiceconv_alex_proc.stdin.write(tts_wav + "\t" + name + "\n")
    voiceconv_alex_proc.stdin.flush()
    print("Cloning ...")
    for line in voiceconv_alex_proc.stdout:
        if not "ADC:" in line:
            print(line)
        else:
            if line.strip().startswith("ADC"):
                vc_wav = line[4:].strip()
                break
    wav2lip_proc.stdin.write(vc_wav + "\t" + video_name + "\n")
    wav2lip_proc.stdin.flush()
    print("Wav2lip:::")
    for line in wav2lip_proc.stdout:
        if not "RES" in line:
            print(line)
        else:
            break
    print("TTS time")
    print(tts_time - start)
    print("Total time")
    print(time.time() - start)
    # shutil.move("/project/OML/titanic/VoiceConv/converted_iwslt_4/converted_gen.wav","audio/converted_gen.wav")
    # get the description submitted on the web page
    return render_template('simple_template_2.html', voice="converted_gen.wav", TTS_voice="TTS_gen.wav",
                           video="lipgan_out.mp4",
                           sample_text=message, model_choice=name, video_choice=video_name)
    # return 'Description entered: {}'.format(a_description)


@app.route('/<voice>', methods=['GET'])
def stream(voice):
    def generate():
        with open(os.path.join('audio', voice), "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)

    if voice.endswith("wav"):
        return Response(generate(), mimetype="audio/")
    if voice.endswith("mp4"):
        return Response(generate(), mimetype="'video/mp4'")


# @app.route('/prediction', methods=['GET', 'POST'])
# def prediction():
#    if request.method == 'POST':
#        prediction_data = request.json
#        print(prediction_data)
#    return jsonify({'result': prediction_data})

# boilerplate flask app code

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str,
                    help='Filepath of video/image that contains faces to use', required=True)

if __name__ == "__main__":

    log_path = "./worker_log"
    tts_en_err_file = open(os.path.join(log_path, 'tts_en.log_'), 'a+')
    voiceconv_alex_err_file = open(os.path.join(log_path, 'voiceconv.log_'), 'a+')
    wav2lip_err_file = open(os.path.join(log_path, 'wav2lip.log_'), 'a+')

    tts_en_proc = subprocess.Popen('./process/inference_tts.sh {}'.format(3),
                                   shell=True, encoding="utf-8", bufsize=0, universal_newlines=False,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=tts_en_err_file)
    for line in tts_en_proc.stdout:
        if "Up and running" in line:
            break

    voiceconv_alex_proc = subprocess.Popen('process/inference_vc.sh {}'.format(3),
                                           shell=True, encoding="utf-8", bufsize=0, universal_newlines=False,
                                           stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=voiceconv_alex_err_file)
    print("Starting Voice conv for Alex")

    for line in voiceconv_alex_proc.stdout:
        if "VoiceConv READY" in line:
            break

    wav2lip_proc = subprocess.Popen('process/inference_w2l.sh {}'.format(3),
                                    shell=True, encoding="utf-8", bufsize=0, universal_newlines=False,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=wav2lip_err_file)

    for line in wav2lip_proc.stdout:
        if "Wav2lip READY" in line:
            break

    app.run(debug=True, host="0.0.0.0", port=8080, use_reloader=False)
