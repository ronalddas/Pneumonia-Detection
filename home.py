from flask import Flask,request,render_template,redirect,url_for
from werkzeug.utils import secure_filename
import json
import traceback
import os
import sys
from model_classify import predict
from opencv_functions import merge_images
from shutil import copy

app = Flask(__name__)
UPLOADS_FOLDER=os.path.join("static","uploads")
app.config['UPLOAD_FOLDER'] = UPLOADS_FOLDER
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    try:
        print("insi")
        if request.method == 'POST':
            f = request.files['myfile']
            print(app.config['UPLOAD_FOLDER'])
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
            return render_template("upload.html",value=full_filename)
        if request.method == 'GET':
            print("Inside Get")
            f = request.args['comp_select']
            file_loc="static/preloaded/"+str(f)+".png"
            copy(file_loc,"static/uploads/"+str(f)+".png")
            return render_template("upload.html", value=file_loc)
    except Exception:
        traceback.print_exc()

@app.route('/ml', methods=['GET', 'POST'])
def ml():
    task=request.args["method"]
    image_loc=request.args["imageloc"]
    print(task,image_loc)
    #print(params,type(params))
    result=predict(task,image_loc)
    #full_filename=os.path.join(app.config['UPLOAD_FOLDER'], params)
    return render_template("ml.html",value=str(result),image_loc=image_loc)

@app.route('/ml_segnet', methods=['GET', 'POST'])
def ml_segnet():
    task=request.args["method"]
    image_loc=request.args["imageloc"]
    print(task,image_loc)
    filename=image_loc.split("/")[-1]
    #print(params,type(params))
    os.system(" python3 -m keras_segmentation predict  --checkpoints_path='path_to_checkpoints'  --input_path='static/uploads/'  --output_path='static/predictions/'")
    #predict_multiple(inp_dir=image_loc, out_dir="static/predictions/", checkpoints_path="model_files/vgg_segnet/vgg_segnet")
    #result=predict(task,image_loc)
    #full_filename=os.path.join(app.config['UPLOAD_FOLDER'], params)
    merge_images(image_loc,"static/predictions/"+filename)
    return render_template("ml_segnet.html",image_loc=image_loc,image_loc_new="static/merged/"+filename)


if __name__=="__main__":
    os.system("rm -f static/predictions/*")
    os.system("rm -f static/uploads/*")
    os.system("rm -f static/merged/*")
    app.run(host="0.0.0.0",debug=True)