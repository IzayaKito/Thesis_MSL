
from flask_cors import CORS, cross_origin
from models.models import create_model
from options.train_options import TrainOptions
from flask import Flask, request

import json
import util.util as util
import pyrebase

firebaseConfig = {
    "apiKey": "AIzaSyDFb6ik6ghOye2BW1Y1bDvbe8W9NrSsg4M",
    "authDomain": "mfcnn-storage.firebaseapp.com",
    "databaseURL": "https://mfcnn-storage.firebaseio.com",
    "projectId": "mfcnn-storage",
    "storageBucket": "mfcnn-storage.appspot.com",
    "serviceAccount": "/home/ubuntu/Thesis_MSL/credentials.json"
}

firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
	
#opt = TrainOptions().parse()
#util.save_object(opt,'opt.pkl')
opt = util.read_object('opt.pkl')
model = create_model(opt)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        segmentation = util.get_prediction(img_bytes, model)

        util.saveImageStatic(app.root_path, file.filename, segmentation)

        storage.child(
            "database/" + file.filename).put("static/" + file.filename)
        url = storage.child("database/" + file.filename).get_url(None)

        data = {"data": url}
        response = app.response_class(response=json.dumps(data),
                                      status=200,
                                      mimetype='application/json')

        util.deleteImageStatic(app.root_path, file.filename)

        return response


@app.route('/result/list', methods=['GET'])
@cross_origin()
def listResults():
    if request.method == 'GET':
        files = storage.list_files()
        fileData = []
        urls = []
        for file in files:
            if(file.name == "database/"):
                continue
            url = storage.child(file.name).get_url(None)
            name = (file.name).replace("database/", "")

            dic = {"name": name, "url": url}
            fileData.append(dic)

        data = {"data": fileData}
        response = app.response_class(response=json.dumps(data),
                                      status=200,
                                      mimetype='application/json')
        return response


@app.route('/result', methods=['GET'])
@cross_origin()
def getImage():
    if request.method == 'GET':
        img = request.args.get('img')
        url = storage.child(img).get_url(None)
        data = {"data": url}
        response = app.response_class(response=json.dumps(data),
                                      status=200,
                                      mimetype='application/json')
        return response


@app.route('/result/delete', methods=['DELETE'])
@cross_origin()
def deleteImage():
    if request.method == 'DELETE':
        img = request.args.get('img')
        url = storage.delete('database/'+img)
        data = {"data": img + " deleted succesfully"}
        response = app.response_class(response=json.dumps(data),
                                      status=200,
                                      mimetype='application/json')
        return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
