from flask import Flask, request, jsonify
from utils.model_prog import final_predict
from uuid import uuid4
from os import remove

app = Flask(__name__)

@app.route("/im_size", methods=["POST"])
def process_image():
    file = request.files['image']
    extension = file.filename.split(".")[-1]
    file_name = str(uuid4()) + "." + extension
    with open(file_name, "wb") as img_file:
        img_file.write(file.getvalue())
        
    result = final_predict(file_name)
    remove(file_name)
    return jsonify({'msg': 'success', 'prediction': result})


if __name__ == "__main__":
    app.run(debug=True)