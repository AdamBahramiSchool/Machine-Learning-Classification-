import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from skimage.color import rgb2lab
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage.color import rgb2hsv

OUTPUT_TEMPLATE = (
    'Bayesian classifier:    {bayes_rgb:.3f} {bayes_convert:.3f}\n'
    'kNN classifier:         {knn_rgb:.3f} {knn_convert:.3f}\n'
    'Rand forest classifier: {rf_rgb:.3f} {rf_convert:.3f}\n'
)


# representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 113, 0),
    'yellow': (255, 255, 0),
    'green': (0, 230, 0),
    'blue': (0, 0, 255),
    'purple': (187, 0, 187),
    'brown': (117, 60, 0),
    'pink': (255, 186, 186),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[np.uint8, np.uint8, np.uint8])


def plot_predictions(model, lum=70, resolution=256):
    """
    Create a slice of LAB colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    # create a hei*wid grid of LAB colour values, with L=lum
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)

    # convert to RGB for consistency with original input
    X_grid = lab2rgb(lab_grid)

    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((-1, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))

    # plot input and predictions
    plt.figure(figsize=(10, 5))
    plt.suptitle('Predictions at L=%g' % (lum,))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.ylabel('B')
    plt.imshow(X_grid.reshape((hei, wid, -1)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)


def lab_converter(X):
    X=np.array(X)
    lab_values=rgb2lab(X[np.newaxis, np.newaxis, :])
    return lab_values.reshape(-1, 3)

def hsv_converter(X):
    X=np.array(X)
    hsv_values=rgb2hsv(X/255)
    return hsv_values.reshape(-1, 3)

def bayes_rgb_model_getter(x_train,y_train):
    bayes_rgb_model=GaussianNB()
    bayes_rgb_model.fit(x_train,y_train)
    return bayes_rgb_model

def knn_rgb_model_getter(x_train,y_train):
    knn_rgb_model=KNeighborsClassifier()
    knn_rgb_model.fit(x_train,y_train)
    return knn_rgb_model

    
def bayes_rgb_2_lab(x_train,y_train):
    model_lab=make_pipeline(
        FunctionTransformer(hsv_converter),
        GaussianNB()
    )
    model_lab.fit(x_train,y_train)
    return model_lab

def knn_rgb_2_hsv(x_train,y_train):
    knn_model_hsv=make_pipeline(
        FunctionTransformer(hsv_converter),
        KNeighborsClassifier()
    )
    knn_model_hsv.fit(x_train,y_train)
    return knn_model_hsv

def rf_rgb_getter(x_train,y_train):
    rf_rgb_model=RandomForestClassifier()
    rf_rgb_model.fit(x_train,y_train)
    return rf_rgb_model

def rf_rgb_2_hsv(x_train,y_train):
    rf_rgb_model=make_pipeline(
        FunctionTransformer(hsv_converter),
        RandomForestClassifier()
    )
    rf_rgb_model.fit(x_train,y_train)
    return rf_rgb_model

def main():
    data = pd.read_csv(sys.argv[1])
    X = data[['R', 'G', 'B']].values / 255
    y = data['Label'].values
    X_train, X_valid, y_train, y_valid =train_test_split(X,y,random_state=10)
    # TODO: create some models
    bayes_rgb_model=bayes_rgb_model_getter(X_train,y_train)
    bayes_convert_model=bayes_rgb_2_lab(X_train,y_train)
    knn_rgb_model=knn_rgb_model_getter(X_train,y_train)
    knn_convert_model=knn_rgb_2_hsv(X_train,y_train)
    rf_rgb_model=rf_rgb_getter(X_train,y_train)
    rf_convert_model=rf_rgb_2_hsv(X_train,y_train)
    # train each model and output image of predictions
    models = [bayes_rgb_model, bayes_convert_model, knn_rgb_model, knn_convert_model, rf_rgb_model, rf_convert_model]
    for i, m in enumerate(models):  # yes, you can leave this loop in if you want.
        m.fit(X_train, y_train)
        plot_predictions(m)
        plt.savefig('predictions-%i.png' % (i,))

    print(OUTPUT_TEMPLATE.format(
        bayes_rgb=bayes_rgb_model.score(X_valid, y_valid),
        bayes_convert=bayes_convert_model.score(X_valid, y_valid),
        knn_rgb=knn_rgb_model.score(X_valid, y_valid),
        knn_convert=knn_convert_model.score(X_valid, y_valid),
        rf_rgb=rf_rgb_model.score(X_valid, y_valid),
        rf_convert=rf_convert_model.score(X_valid, y_valid),
    ))


if __name__ == '__main__':
    main()
