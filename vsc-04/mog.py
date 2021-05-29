import numpy as np
import matplotlib.image as mpimage
import matplotlib.pyplot as plt


def show_image(img):
    """
    Shows an image (img) using matplotlib
    """
    if isinstance(img, np.ndarray):
        if img.shape[-1] == 3 or img.shape[-1] == 4:
            plt.imshow(img[...,:3])
        if img.shape[-1] == 1 or img.shape[-1] > 4:
            plt.imshow(img[:,:], cmap="gray")
        plt.show()


def convolution2D(img, kernel):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix
    :return: result of the convolution
    """
    # 1.1.1 TODO. Initialisieren Sie das resultierende Bild
    new_img = np.zeros(img.shape)
    p_img = np.pad(img, kernel.shape[0]//2, mode='edge')
    # 1.1.2 TODO. Implementieren Sie die Faltung.
    for i in range(kernel.shape[0]//2, p_img.shape[0]-kernel.shape[0]//2):
        for j in range(kernel.shape[0]//2, p_img.shape[1]-kernel.shape[1]//2):
            new_img[i-kernel.shape[0]//2,j-kernel.shape[1]//2] = (kernel * p_img[i-kernel.shape[0]//2: i-kernel.shape[0]//2+kernel.shape[0], j-kernel.shape[1]//2: j-kernel.shape[1]//2+kernel.shape[1]]).sum()
    # Achtung: die Faltung (convolution) soll mit beliebig großen Kernels funktionieren.
    # Tipp: Nutzen Sie so gut es geht Numpy, sonst dauert der Algorithmus zu lange.
    # D.h. Iterieren Sie nicht über den Kernel, nur über das Bild. Der Rest geht mit Numpy.

    # Achtung! Achteten Sie darauf, dass wir ein Randproblem haben. Wie ist die Faltung am Rand definiert?
    # Tipp: Es gibt eine Funktion np.pad(Matrix, 5, mode="edge") die ein Array an den Rändern erweitert.

    # 1.1.3 TODO. Returnen Sie das resultierende "Bild"/Matrix
    return new_img


def magnitude_of_gradients(RGB_img):
    """
    Computes the magnitude of gradients using x-sobel and y-sobel 2Dconvolution

    :param img: RGB image
    :return: length of the gradient
    """
    # 3.1.1 TODO. Wandeln Sie das RGB Bild in ein grayscale Bild um.
    RGB_img = RGB_img.astype("float64")
    g_img = RGB_img[..., :3] @ np.array([0.299, 0.587, 0.114])
    # 3.1.2 TODO: Definieren Sie den x-Sobel Kernel und y-Sobel Kernel.
    x_sobel = np.array([[1,0,-1],
                        [2,0,-2],
                        [1,0,-1]])
    y_sobel = np.array([[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]])
    # 3.1.3 TODO: Nutzen Sie sie convolution2D Funktion um die Gradienten in x- und y-Richtung zu berechnen.
    x_grad = convolution2D(g_img, x_sobel)
    y_grad = convolution2D(g_img, y_sobel)
    # 3.1.4 TODO: Nutzen Sie die zwei resultierenden Gradienten um die gesammt Gradientenlängen an jedem Pixel auszurechnen.
    return np.sqrt(x_grad**2 + y_grad**2)
    #return np.arctan(x_grad/y_grad)

# Diese if Abfrage (if __name__ == '__main__':) sorgt dafür, dass der Code nur
# ausgeführt wird, wenn die Datei (mog.py) per python/jupyter ausgeführt wird ($ python mog.py).
# Solltet Ihr die Datei per "import mog" in einem anderen Script einbinden, wird dieser Code übersprungen.
if __name__ == '__main__':
    # Bild laden und zu float konvertieren
    img = mpimage.imread('bilder/tower.jpg')
    show_image(img)
    #show_image(img)
    img = img.astype("float64")
    x_sobel = np.array([[1,0,-1],
                        [2,0,-2],
                        [1,0,-1]])
    y_sobel = np.array([[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]])
    sharpen = np.array([[0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])
    emboss = np.array([[-2,-1,0],
                        [-1,1,1],
                        [0,1,2]])
    edge_detection = np.array([ [-1,-1,-1],
                                [-1,8,-1],
                                [-1,-1,-1]])
    blur = (1/250)/np.array([[1,4,6,4,1],
                           [4,16,24,16,4],
                           [6,24,36,24,6],
                           [4,16,24,16,4],
                           [1,4,6,4,1]])

    # Wandelt RGB Bild in ein grayscale Bild um
    gray = img[...,:3]@np.array([0.299, 0.587, 0.114])
    conv_img_sobelx = convolution2D(gray,x_sobel)
    conv_img_sobely = convolution2D(gray, y_sobel)
    conv_img_sharpen = convolution2D(gray, sharpen)
    conv_img_emboss = convolution2D(gray, emboss)
    conv_img_edge_detection = convolution2D(gray, edge_detection)
    conv_img_blur = convolution2D(gray, blur)
    #print(gray)

    #show_image(conv_img)

    mog = magnitude_of_gradients(img)
    show_image(mog)
    # Aufgabe 1.
    # 1.1 TODO: Implementieren Sie die convolution2D Funktion (oben)

    # Aufgabe 2.
    # 2.1 TODO: Definieren Sie mindestens 5 verschiedene Kernels (darunter sollten beide Sobel sein) und testen Sie sie auf dem grayscale Bild indem Sie convolution2D aufrufen.
    # 2.2 TODO: Speichern Sie alle Resultate als Bilder (sehe Tipp 2). Es sollten 5 Bilder sein.

    # Aufgabe 3:
    # 3.1 TODO: Implementieren Sie die magnitude_of_gradients Funktion (oben) und testen Sie sie mit dem RGB Bild.
    # 3.2 TODO: Speichern Sie das Resultat als Bild (sehe Tipp 2).

    # ------------------------------------------------
    # Nützliche Funktionen:
    # ------------------------------------------------
    # Tipp 1: So können Sie eine Matrix als Bild anzeigen:
    # show_image(gray)

    # Tipp 2: So können Sie eine NxMx3 Matrix als Bild speichern:
    # mpimage.imsave("test.png", img)
    # und so können Sie eine NxM Matrix als grayscale Bild speichern:
    mpimage.imsave("sobel_x.png", conv_img_sobelx, cmap="gray")
    mpimage.imsave("sobel_ly.png", conv_img_sobely, cmap="gray")
    mpimage.imsave("sharpen.png", conv_img_sharpen, cmap="gray")
    mpimage.imsave("emboss.png", conv_img_emboss, cmap="gray")
    mpimage.imsave("edge_detection.png", conv_img_edge_detection, cmap="gray")
    mpimage.imsave("blur.png", conv_img_blur, cmap="gray")
    mpimage.imsave("mog.png", mog, cmap="gray")
