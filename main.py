from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import abort
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import numpy as np
from scipy.fft import rfft, rfftfreq

app = Flask(__name__)
app.config['SECRET_KEY'] = 'my the best secret key'
aboba = []
y_main = 1

extralerning = False
need_data = False

modelL = Sequential([
    #  LSTM(256, input_shape = X_train.shape[1:]),
    LSTM(256, return_sequences=True, input_shape=(1, 256)),
    LSTM(256, return_sequences=True),
    SpatialDropout1D(0.5),
    Dense(512, activation="linear"),
    SpatialDropout1D(0.5),
    LSTM(512, return_sequences=True),
    SpatialDropout1D(0.5),
    Dense(256, activation="linear")
])

modelL.compile(loss="mse", optimizer=Adam(lr=1e-5))

modelL = load_model('static/modelL.h5')


# db_session.global_init("db/jobs.sqlite")
# login_manager = LoginManager(app)
# login_manager.login_view = 'main'


@app.route('/api/get_data', methods=['GET'])
def get_data():
    if len(aboba) >= 256:
        yf = rfft(aboba[-256:])
        yf = np.abs(yf)
        xf = rfftfreq(44100 * 2, 1 / 44100)
        ans = {
            "title": "all data",
            "data": aboba[-256:],
            "predict": list(map(float, modelL.predict(np.array(aboba[-256:]).reshape((1, 1, 256))).reshape(256))),
            "len": len(aboba[-256:]),
            "labels": [i for i in range(256)],
            "yf": list(map(float, yf[1:])),
            "xf": list(map(float, xf[1:]))
        }
        return jsonify(ans)
    abort(404)


@app.route('/api/new_data', methods=['POST'])
def add_data():
    global aboba, extralerning, modelL, need_data
    if not request.json:
        abort(404)
    aboba += request.json["data"]
    # print(len(aboba))
    # print(request.json["data"])
    if len(aboba) >= 128 * 2 * 4 * 8 and need_data:
        f = open('data.txt', 'w')
        f.write(str(aboba))
        f.close()
        need_data = False
        # print(type(eval(f.read())))
    if len(aboba) >= 256 and extralerning:
        x_train = np.array(aboba[-256:]).reshape((1, 1, 256))
        modelL.fit(x_train, x_train,
                   epochs=100,
                   verbose=1,
                   batch_size=5)
        extralerning = False
    return "ok"


@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('grafik.html', title='График')


if __name__ == '__main__':
    # app.run(port=80, host='10.34.196.73')
    app.run(port=int(os.environ.get("PORT")), host='0.0.0.0')
