#Flaskとrender_template（HTMLを表示させるための関数）をインポート
from flask import Flask,render_template,request
from flask_bootstrap import Bootstrap
#Flaskオブジェクトの生成
app = Flask(__name__)
bootstrap=Bootstrap(app)

#「/」へアクセスがあった場合に、"Hello World"の文字列を返す
@app.route("/")
def index():
    return render_template("index.html")


#「/index」へアクセスがあった場合に、「index.html」を返す
@app.route('/predict', methods=['POST'])
def predict():
    import numpy as np
    from PIL import Image
    import pickle
    def preProcess(file):
        img = Image.open(file)
        img = img.convert('L')
        img = img.resize((32, 32))
        img = np.asarray(img)

        #     ret,img = cv2.threshold(img,130,255,cv2.THRESH_BINARY)

        img = img.reshape(-1)
        img = img / 255
        return img

    data = request.files['input_file'].stream
    img = preProcess(data)
    img = img.reshape(1, 1024)
    load_model=pickle.load(open('./skmodel.sav', 'rb'))
    result = load_model.predict(img)

    if result==1:
        last=3
    else:
        last=2
    return render_template('result.html',result_output=last)
#おまじない
if __name__ == "__main__":
    app.run(debug=True)