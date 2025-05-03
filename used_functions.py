import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict
from typing import Tuple

def get_equalization_transform_of_img(img_array: np.ndarray,) -> np.ndarray:
    array = img_array.flatten()
    hist = np.zeros(256)
    prob = np.zeros(256)
    z = np.zeros(256)
    u = np.zeros(256)
    y = np.zeros(256)
    for item in array: #Βρίσκω το ιστόγραμμα των τιμών των pixel
        hist[item] += 1


    total_pixels = img_array.size
    for i in range(len(hist)):#Αντιστοιχώ την συχνότητα εμφάνισης των pixel σε πιθανότητα
        prob[i] = hist[i] / total_pixels

    #Εφαρμόζω τις σχέσεις της θεωρίας
    u[0] = prob[0]

    for i in range(1, len(y)):
        u[i] = u[i - 1] + prob[i]
    for i in range(len(y)):
        z[i] = ((u[i] - u[0]) / (1 - u[0]) * 255)
        y[i] = round(z[i])

    equalization_transform = np.zeros(255)
    equalization_transform = y

    return equalization_transform

def perform_global_hist_equalization(img_array: np.ndarray,) -> np.ndarray:
    eq_transform = get_equalization_transform_of_img(img_array)

    equalized_img_array = np.zeros_like(img_array)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            # Αντιστοιχίζω το κάθε pixel στην αντίστοιχη εξισορροπημένη τιμή του
            equalized_img_array[i, j] = eq_transform[img_array[i, j]]

    return equalized_img_array

def calculate_eq_transformations_of_regions(img_array: np.ndarray,region_len_h: int,region_len_w: int) -> Dict[Tuple, np.ndarray]:
    h, w = img_array.shape[:2]
    num_regions_h = h // region_len_h #Βρίσκω τον αριθμό των περιοχών
    num_regions_w = w // region_len_w

    region_dict = {}

    for i in range(num_regions_h):
        for j in range(num_regions_w):
            start_row = i * region_len_h
            start_col = j * region_len_w
            end_row = start_row + region_len_h
            end_col = start_col + region_len_w

            region = img_array[start_row:end_row, start_col:end_col]
            region_dict[(start_row, start_col)] = get_equalization_transform_of_img(region)#Αναθέτω τον κάθε μετασχηματισμό στα στοιχεία της αρχικής γωνιας
    return region_dict


def perform_adaptive_hist_equalization(img_array: np.ndarray, region_len_h: int, region_len_w: int) -> np.ndarray:
    h, w = img_array.shape[:2]
    num_regions_h = h // region_len_h
    num_regions_w = w // region_len_w

    my_dict = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)

    new_image = np.zeros_like(img_array)
    #Χωρίζω περιπτώσεις ανάλογα που βρισκεταί το κάθε pixel και προσαρμόζω τις κατάλληλες σχέσεις
    for i in range(h):
        for j in range(w):

            start_row = round(i/region_len_h) * region_len_h
            start_col = round(j/region_len_w) * region_len_w
            end_row = start_row + region_len_h
            end_col = start_col + region_len_w

            #Αν βρίσκεται στη γύρω περιοχή
            if (i <= region_len_h // 2) or (j <= region_len_w // 2 ) or (i >= h - (region_len_h // 2)) or (j >= w - (region_len_w // 2)):
                region_row = i // region_len_h
                region_col = j // region_len_w
                transformation = my_dict[(region_row * region_len_h, region_col * region_len_w)]
                new_image[i][j] = transformation[img_array[i, j]]
            #Αν είναι στο πάνω δεξια τεταρτημόριο
            elif (i < start_row + region_len_h / 2) and (j > start_col + region_len_w / 2):
                h_m = start_row - region_len_h // 2 #Το h -
                h_p = start_row + region_len_h // 2 #Το h +
                w_m = start_col + region_len_w // 2 #Το w -
                w_p = end_col + region_len_w // 2  #Το w +

                a = (j - w_m) / (w_p - w_m)
                b = (i - h_m) / (h_p - h_m)
                t_m_p = my_dict[(start_row - region_len_h, start_col + region_len_w)] #το Τ-+
                t_m_m = my_dict[(start_row - region_len_h, start_col)]# το Τ--
                t_p_m = my_dict[(start_row, start_col)]#Το Τ+-
                t_p_p = my_dict[(start_row, start_col + region_len_w)]# το Τ++

                new_image[i][j] = int((1 - a) * (1 - b) * t_m_m[img_array[i, j]] + (1 - a) * b * t_p_m[img_array[i, j]] + a * (1 - b) * t_m_p[img_array[i, j]] + a * b * t_p_p[img_array[i, j]])
            #Αν είναι στο κάτω δεξια τεταρτημόριο
            elif (i >= start_row + region_len_h / 2) and (j >= start_col + region_len_w / 2):
                h_m = start_row + region_len_h // 2
                h_p = end_row + region_len_h // 2
                w_m = start_col + region_len_w // 2
                w_p = end_col + region_len_w // 2

                a = (j - w_m) / (w_p - w_m)
                b = (i - h_m) / (h_p - h_m)
                t_m_m = my_dict[(start_row , start_col )]#το Τ--
                t_m_p = my_dict[(start_row, start_col + region_len_w)]#το Τ-+
                t_p_p = my_dict[(start_row + region_len_h, start_col + region_len_w)]#το Τ++
                t_p_m = my_dict[(start_row + region_len_h, start_col)]#το Τ+-

                new_image[i][j] = int((1 - a) * (1 - b) * t_m_m[img_array[i, j]] + (1 - a) * b * t_p_m[img_array[i, j]] + a * (1 - b) * t_m_p[img_array[i, j]] + a * b * t_p_p[img_array[i, j]])
            #Αν ειναι στο πάνω αριστερά τεταρτημόριο
            elif (i <= start_row + region_len_h / 2) and (j <= start_col + region_len_w / 2):
                h_p = start_row + region_len_h // 2
                h_m = start_row - region_len_h // 2
                w_p = start_col + region_len_w // 2
                w_m = start_col - region_len_w // 2

                a = (j - w_m) / (w_p - w_m )
                b = (i - h_m) / (h_p - h_m )
                t_m_m = my_dict[(start_row - region_len_h, start_col - region_len_w)]#το Τ--
                t_m_p = my_dict[(start_row - region_len_h, start_col)]#το Τ-+
                t_p_m = my_dict[(start_row, start_col - region_len_w)]#το Τ+-
                t_p_p = my_dict[(start_row, start_col)]#το Τ++

                new_image[i][j] =int((1 - a) * (1 - b) * t_m_m[img_array[i, j]] + (1 - a) * b * t_p_m[img_array[i, j]]+ a * (1 - b) * t_m_p[img_array[i, j]] + a * b * t_p_p[img_array[i, j]])
            #Αν ειναι στο κάτω δεξία τεταρτημόριο
            else:
                if (i >= start_row + region_len_h / 2) and (j <= start_col + region_len_w / 2):

                    h_m = start_row + region_len_h // 2
                    h_p = end_row + region_len_h // 2
                    w_p = start_col + region_len_w // 2
                    w_m = start_col - region_len_w // 2

                    a = (j - w_m) / (w_p - w_m )
                    b = (i - h_m) / (h_p - h_m )
                    t_m_m = my_dict[(start_row, start_col - region_len_w)]#το Τ--
                    t_p_p = my_dict[(start_row + region_len_h , start_col)]#το Τ++
                    t_p_m = my_dict[(start_row + region_len_h , start_col - region_len_w)]#το Τ+-
                    t_m_p = my_dict[(start_row, start_col)]#το Τ-+

                    new_image[i][j] = int((1 - a) * (1 - b) * t_m_m[img_array[i, j]] + (1 - a) * b * t_p_m[img_array[i, j]] + a * ( 1 - b) * t_m_p[img_array[i, j]] + a * b * t_p_p[img_array[i, j]])


    return new_image



























