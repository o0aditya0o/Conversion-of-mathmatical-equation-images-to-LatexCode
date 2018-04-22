import cv2
import numpy as np
from keras.models import model_from_json
import preprocess as pre
import mapping as mp
import array
from PIL import Image
import pickle
import features as fea
import probability as proba

def display_image(img, str='image'):
    cv2.imshow(str, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cnvt_bool_to_uint8(img):
    ret = np.zeros(img.shape, dtype=np.uint8)
    r, c = img.shape
    for i in range(0,r):
        for j in range(0,c):
            if img[i,j] == 1:
                ret[i,j] = 255
    return ret

def predict_class(img, model):
    # Load ANN
    with open('MLPClassifier.pkl', 'rb') as f:
        clf1 = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    #  Load CNN
    json_file = open('CNNmodelFinal.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("F:\CODING\ProjectLatex\draft\models\.014-0.783.hdf5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    if model == "cnn":
        img = cv2.resize(img, (128, 128))
        # img = pre.filter_image(img)
        # img = pre.otsu_thresh(img)
        # print(img)
        immatrix = []
        # img_arr = array(np.asarray(img)).flatten()
        immatrix.append(img)
        inp = np.asarray(immatrix)
        Output = proba.prob(img)
        inp = inp.reshape(inp.shape[0], 128, 128, 1)
        inp = inp.astype('float32')
        inp /= 255
        # print(inp)
        output = loaded_model.predict_classes(inp)
        # print(output)
        z = mp.list[int(output[0])]
        # output = proba(Output, z)
        return Output
    else:
        x = fea.get_data(img)
        temp = []
        temp.append(x)
        temp = scaler.transform(temp)
        y = clf1.predict(temp)
        y = mp.list[int(y[0])]
        return y