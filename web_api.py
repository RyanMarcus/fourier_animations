from flask import Flask, app, request, jsonify, url_for, redirect
from random_phase import make_image_from_pixels, read_image_and_convert_gif, upload_to_imgur
from uuid import uuid4
import os
import shutil

app = Flask(__name__)

@app.route("/")
def index():
    #return redirect(url_for('static', filename='index.html'))
    # If you have a static index.html uncomment the above line
    return "HELLO WORLD"

@app.route("/getgif", methods=["POST"])
def pixels_to_gif_api():
    person_uuid = str(uuid4())
    pixel_data = request.get_json(force=True)
    make_image_from_pixels(pixel_data, "{}.png".format(person_uuid))
    read_image_and_convert_gif(person_uuid)
    r = upload_to_imgur("{}.gif".format(person_uuid))
    os.remove("{}.png".format(person_uuid))
    os.remove("{}.gif".format(person_uuid))
    shutil.rmtree(person_uuid)
    return jsonify(r)

if __name__=="__main__":
    app.run('0.0.0.0')