from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from typing import Tuple

from used_functions import get_equalization_transform_of_img
from used_functions import perform_global_hist_equalization
from used_functions import perform_adaptive_hist_equalization
from used_functions import calculate_eq_transformations_of_regions

#Αυτό είναι το αρχείο που περιλαμβάνει τις συναρτήσεις που χρησιμοποιήθηκαν



filename = "input_img.png"
img = Image.open(fp=filename)
bw_img = img.convert("L")
img_array = np.array(bw_img)

sub_array = img_array[0:336 , 0:448] #Κρατάω το πάνω αριστερά κομμάτι της εικόνας
img = Image.open("input_img.png")
# Display the image
plt.imshow(bw_img, cmap='gray')  #Προβάλω την αρχική εικόνα
plt.axis('off')  # Hide axis
plt.show()

array = sub_array.flatten() #Αποθηκεύω τα στοιχεία των pixel σε ένα μονοδιάστατο πίνακα για ευκολότερους υπολογισμούς

hist = np.zeros(256)

#Μετράω τον αριθμό εμφάνισης κάθε pixel για την δημιουργία ιστογράμματος
for item in array:
    hist[item] += 1


plt.bar(range(len(hist)), hist, color='blue', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram before equalization')

plt.show()






#Εφαρμόζω την γενική εξισορρόπηση ιστογράμματος
equalized_img_array = perform_global_hist_equalization(sub_array)

hist = np.zeros(256)

#Μετράω τον αριθμό εμφάνισης κάθε pixel για την δημιουργία ιστογράμματος
array = equalized_img_array.flatten()
for item in array:
    hist[item] += 1

plt.bar(range(len(hist)), hist, color='blue', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram after global equalization')
plt.show()

#Κάνω ιστόγραμμα με 16 bins για να φανεί καλύτερα το αποτέλεσμα
num_bins = 16

bins = np.zeros(num_bins)

for val in array:
    bin_index = val // 16
    bins[bin_index] += 1

plt.bar(range(len(bins)), bins, color='blue', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram after equalization')
plt.xticks(np.arange(0, num_bins, 1), np.arange(0, 256, 16))
plt.show()

plt.imshow(equalized_img_array, cmap='gray')
plt.title('Grayscale image after general histogram equalization')
plt.axis('off')  # Hide axis
plt.show()

#Εφαρμόζω προσαρμοστική εξισορρόπηση ιστογράμματος χωρίς παρεμβολή


region_len_h = 48
region_len_w = 64

h, w = sub_array.shape[:2]
num_regions_h = h // region_len_w
num_regions_w = w // region_len_w
my_dict = calculate_eq_transformations_of_regions(sub_array, region_len_h, region_len_w)

new_image = np.zeros_like(sub_array)

for i in range(h):
    for j in range(w):
        region_row = i // region_len_h
        region_col = j // region_len_w
        transformation = my_dict[(region_row * region_len_h, region_col * region_len_w)]#Βρίσκω ποιός μετασχηματισμός πρέπει να εφαρμοστεί αναλογα την περιοχή που βρισκεταi το pixel
        new_image[i][j] = transformation[img_array[i, j]]

#Τυπώνω την εικόνα με την προσαρμοστική εξισορρόπηση ιστογράμματος χωρις παρεμβολή
plt.imshow(new_image, cmap='gray')
plt.title('Grayscale image after adaptive histogram equalization')
plt.axis('off')  # Hide axis
plt.show()


equalized_img_array_2 = perform_adaptive_hist_equalization(sub_array ,48,64)


#Τυπώνω την εικόνα με την προσαρμοστική εξισορρόπηση ιστογράμματος
plt.imshow(equalized_img_array_2, cmap='gray')
plt.title('Grayscale image after adaptive histogram equalization with interpolation')
plt.axis('off')  # Hide axis
plt.show()













