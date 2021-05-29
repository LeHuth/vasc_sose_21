import numpy as np
import matplotlib.image as mpimage
from mog import magnitude_of_gradients, show_image


def seam_carve(image, seam_mask):
    """
    Removes a seam from the image depending on the seam mask. Returns an image
     that has one column less than <image>
    flattend_img = np.reshape(image, (image.shape[0] * image.shape[1], 3))
    #print(image.shape)
    #print(flattend_img.shape)
    #print(np.where(seam_mask.flatten() == False))
    new_arr = np.delete(flattend_img,np.where(seam_mask.flatten() == False), axis=0)
    #print(image.shape)

    return new_arr.reshape(image.shape[0],image.shape[1]-1, 3)
    :param image:
    :param seam_mask:
    :return: smaller image

    print(image.shape)
        """
    shrunken = image[seam_mask].reshape((image.shape[0],-1,image[...,None].shape[2]))
    return shrunken.squeeze()



def update_global_mask(global_mask, new_mask):
    """
    Updates the global_mask that contains all previous seams by adding the new path contained in new_mask.

    :param global_mask: The global mask (bool-Matrix) where the new path should be added
    :param new_mask: The mask (bool-Matrix) containing a new path
    :return: updated global Mask
    """
    reduced_idc = np.indices(global_mask.shape)[:, ~global_mask][:,new_mask.flat]
    seam_mask = np.ones_like(global_mask, dtype=bool)
    seam_mask[reduced_idc[0],reduced_idc[1]] = False
    return seam_mask


def calculate_accum_energy(energy):
    """
    Function computes the accumulated energies

    :param energy: ndarray (float)
    :return: ndarray (float)
    """
    # 2.1 TODO: Initialisieren Sie das neue resultierende Array
    # Codebeispiel: accumE = np.array(energy)
    accumE = np.array(energy)
    # 2.2 TODO: Füllen Sie das Array indem Sie die akkumulierten
    # Energien berechnen (dynamische Programmierung)
    for i in range(energy.shape[0]):
        for j in range(energy.shape[1]):
            if i == 0:
                accumE[i,j] = energy[i,j]
            elif j + 1 >= energy.shape[1]:
                accumE[i, j] = energy[i,j] + np.amin(np.array([accumE[i - 1, j - 1], accumE[i - 1, j]]))
            elif j - 1 < 0:
                accumE[i, j] = energy[i,j] + np.amin(np.array([accumE[i - 1, j], accumE[i - 1, j + 1]]))
            else:
                accumE[i,j] = energy[i,j] + np.amin(np.array([accumE[i-1,j-1], accumE[i-1,j], accumE[i-1,j+1]]))
    # Tipp: Benutzen Sie das Beispiel aus der Übung zum debuggen

    # 2.3 TODO: Returnen Sie die die akkumulierten Energien
    return accumE

def create_seam_mask(accumE):
    """
    Creates and returns boolean matrix containing zeros (False) where to remove the seam

    :param accumE: ndarray (float)
    :return: ndarray (bool)
    """
    # 3.1.1 TODO: Initialisieren Sie eine Maske voller True-Werte
    # Codebeispiel: Mask = np.ones(accumE.shape, dtype=bool)
    Mask = np.ones(accumE.shape, dtype=bool)
    # 3.1.2 TODO: Finden Sie das erste Minimum der akkumulierten Energien.

    # Achtung! Nach welchem Minimum ist gefragt? Wo muss nach dem Minimum gesucht werden?

    # 3.1.3 TODO: Setzten Sie die entsprechende Stelle (np.argmin) in der Maske auf False

    # 3.1.4 TODO: Wiederholen Sie das für alle Zeilen von unten nach oben.

    # Codebeispiel: for row in reversed(range(0, accumE.shape[0])):

    # Achtung! Wieder: Wo muss nach dem nächsten Minimum gesucht werden?
    # Denkt dran, die Minimums müssen benachbart sein. Das schränkt die Suche
    # nach dem nächsten Minimum enorm ein.

    # 3.1.5 TODO: Returnen Sie die fertige Maske
    mask = np.ones(accumE.shape, dtype=bool)
    mini = 0
    j_index = 0
    for i in range(len(accumE) - 1, -1, -1):
        if i == accumE.shape[0] - 1:
            mini = np.argmin(accumE[i])
            mask[i, mini] = False
            j_index = mini
        else:
            mini = np.argmin(accumE[i, mini - 1 if mini - 1 >= 0 else mini:mini + 2 if mini + 2 <= accumE.shape[1] else mini + 1])
            if mini == 0 and j_index > 0:
                j_index -= 1
            elif mini == 2 and j_index < accumE.shape[1]:
                j_index += 1
            mini = j_index
            mask[i, mini] = False
    return mask

# ------------------------------------------------------------------------------
# Main Bereich
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Initalisierung
    # --------------------------------------------------------------------------
    # lädt das Bild
    img = mpimage.imread('bilder/tower.jpg')  # 'bilder/bird.jpg')

    # erstellt eine globale Maske
    # In der Maske sollen alle Pfade gespeichert werden die herrausgeschnitten wurden
    # An Anfang ist noch nichts herrausgeschnitten, also ist die Maske komplett False
    global_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    # Parameter einstellen:
    # Tipp: hier number_of_seams_to_remove am Anfang einfach mal auf 1 setzen
    number_of_seams_to_remove = 20

    # erstellet das neue Bild, welches verkleinert wird
    new_img = np.array(img, copy=True)
    copy_img = np.array(img, copy=True)

    # --------------------------------------------------------------------------
    # Der Algorithmus
    # --------------------------------------------------------------------------
    # Für jeden Seam, der entfernt werden soll:
    for idx in range(number_of_seams_to_remove):
        # Aufgabe 1:
        # 1.1 TODO: Berechnen Sie die Gradientenlängen des Eingabe Bildes
        # und nutzen Sie diese als Energie-Werte. Sie können dazu Ihre Funktion
        # aus Aufgabe 1. (mog.py) nutzen. Dazu müssen Sie oben Ihr Skript einfügen:
        # Codebeispiel: from mog import magnitude_of_gradients
        #               energy = magnitude_of_gradients(new_img)
        # Tipp: Als Test wäre eine einfache Matrix hilfreich:
        #energy = np.array([[40, 60, 40, 10, 20],[53.3, 50, 25, 47.5, 40],[50, 40, 40, 60, 90],[30,70,75,25,50],[65,70,30,30,10]])
        energy = magnitude_of_gradients(new_img)
        # Aufgabe 2:
        # 2.1 TODO: Implementieren Sie die Funktion calculate_accum_energy.
        # Sie soll gegeben eine Energy-Matrix die akkumulierten Energien berechnen.
        accumE = calculate_accum_energy(energy)
        # Aufgabe 3:
        # 3.1 TODO: Implementieren Sie die Funktion create_seam_mask.
        # Sie soll gegeben einer akkumulierten Energie-matrix einen Pfad finden,
        # der entfernt werden soll. Der Pfad wird mithilfe einer Maske gespeichert.
        # Beispiel:
        #     Bild                         Maske
        # |. . . / . .|     [[True, True, True, False, True, True],
        # |. . / . . .| --> [True, True, False, True, True, True],
        # |. . \ . . .|     [True, True, False, True, True, True],
        # |. . . \ . .|     [True, True, True, False, True, True]]
        #       Seam
        # Codebeispiel: seam_mask = create_seam_mask(accumE)

        # Aufgabe 4:
        # 4.1 TODO: Entfernen Sie den "seam" aus dem Bild mithilfe der Maske und
        # der Funktion seam_carve. Diese Funktion ist vorgegeben und muss nicht
        # implementiert werden.

        seam_mask = create_seam_mask(accumE)
        new_img = seam_carve(new_img, seam_mask)

        # Aufgabe 5:
        # 5.1 TODO: Updaten Sie die globale Maske mit dem aktuellen Seam (update_global_mask).
        global_mask = update_global_mask(global_mask,seam_mask)
        # 5.2 TODO: Kopieren Sie das Originalbild und färben Sie alle Pfade, die bisher
        #            entfert wurden, rot mithilfe der globalen Maske
        # Codebeispiel: copy_img[global_mask, :] = [255,0,0]
        copy_img[global_mask, :] = [255, 0, 0]
        # Aufgabe 6:
        # 6.1 TODO: Speichere das verkleinerte Bild
        show_image(copy_img)
        mpimage.imsave("iterations/tower_iteration_" + str(idx) + ".png", new_img)
        mpimage.imsave("iterations/tower_mask_iteration_" + str(idx) + ".png", copy_img)
        # 6.2 TODO: Speichere das Orginalbild mit allen bisher entfernten Pfaden

        # 6.3 TODO: Gebe die neue Bildgröße aus:
        print(idx, " image carved:" + str(idx), new_img.shape)

    # 6.4. TODO: Speichere das resultierende Bild nocheinmal extra.
    mpimage.imsave("tower_original.png", img)
    mpimage.imsave("tower_w_seam_mask.png", copy_img)
    mpimage.imsave("tower_carved.png", new_img)