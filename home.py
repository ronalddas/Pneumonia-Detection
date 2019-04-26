from flask import Flask,request,render_template,redirect,url_for
from werkzeug.utils import secure_filename
import json
import traceback
import os
from model import predict
app = Flask(__name__)
UPLOADS_FOLDER=os.path.join("static","uploads")
app.config['UPLOAD_FOLDER'] = UPLOADS_FOLDER
@app.route('/')
def hello_world():
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

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)