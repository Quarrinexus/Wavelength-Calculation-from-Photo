from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# measured values used for calculations
width_of_screen = 31.2-1.6 # in cm
delta_width = 0.5 # uncertainty in width of screen in cm (to be discussed)
L = 492.9 # distance from slits to screen in cm
delta_L = 1.6 # uncertainty in L in cm
d = 0.2005 # slit separation in mm (0.2005 or 0.1002 depending on dataset)
delta_d = 0.0003 # uncertainty in d in mm (0.0003 or 0.0001 depending on dataset)
pixel_detection_uncertainty = 1  # pixels (uncertainty in locating peak centers)

dataset = 1 # which dataset to analyze (1-n, corresponding to Filtered_Image_1.jpeg to Filtered_Image_n.jpeg)
min_width = 22 # minimum width of peaks in pixels (to be changed manually)
min_height = 0 # minimum height of peaks as a fraction of the central maximum (to be changed manually)
data_crop_distance = 10 # distance in cm to crop from each side of the image to remove noise (to be changed manually)

print(f"Analyzing dataset {dataset} with parameters: min_width={min_width}, min_height={min_height}, data_crop_distance={data_crop_distance} cm")

img = io.imread(f'Filtered_Images/Filtered_Image_{dataset}.jpeg', as_gray=True)

intensity_profile = np.sum(img, axis=0)  # sum over cols for every x to get horizontal intensity
x_axis = np.arange(len(intensity_profile))

# convert pixels to distances along the screen
total_pixels = img.shape[1]
distance_per_pixel = width_of_screen / total_pixels  # cm per pixel
distance_axis = x_axis * distance_per_pixel

central_idx = np.argmax(intensity_profile)
central_dist = distance_axis[central_idx]
distance_axis -= central_dist  # recentre so central max is at 0

plt.plot(distance_axis, intensity_profile)
plt.xlabel('Distance / cm')
plt.ylabel('Intensity')
plt.title('Intensity Profile Along X-axis, Dataset ' + str(dataset))

# find peaks; we'll calculate their distance from the central maximum
cropped_indices = (distance_axis > -data_crop_distance) & (distance_axis < data_crop_distance)
cropped_intensity = intensity_profile[cropped_indices]
cropped_distance_axis = distance_axis[cropped_indices]
peaks, _ = find_peaks(cropped_intensity, width=min_width, height=max(cropped_intensity)*min_height, distance=min_width)
peak_distances = cropped_distance_axis[peaks]  # distance_axis is already centered

# propagate both calibration and detection uncertainties into distance error
calibration_error = delta_width / width_of_screen * np.abs(peak_distances)
detection_error = pixel_detection_uncertainty * distance_per_pixel
distance_errors = np.sqrt(calibration_error**2 + detection_error**2)

# mark detected peaks
for peak in peaks:
    pd = cropped_distance_axis[peak]
    plt.axvline(x=cropped_distance_axis[peak], color='r', linestyle='--')

plt.savefig(f"Intensity Graphs/Intensity_Profile_Dataset_{dataset}.png")
print(f"Saved intensity profile graph for dataset {dataset}")

plt.show()

# calculate angles using centred distances
theta_values = np.arctan(peak_distances / L)  # angles in radians
theta_errors = np.abs(np.cos(theta_values)) * np.sqrt((distance_errors / L)**2 + (peak_distances * delta_L / L**2)**2)  # propagate uncertainty in angle
sin_theta_errors = np.abs(np.cos(theta_values)) * theta_errors
n_limits = (len(peak_distances)+1)/2
n_values = np.arange(-n_limits+1, n_limits)  # order numbers, centered around 0

def linear_model(x, m, c):
    return m * x + c

# curve fit
popt, pcov = curve_fit(linear_model, n_values, np.sin(theta_values), sigma=sin_theta_errors, absolute_sigma=True)
m, c = popt
m_error = np.sqrt(pcov[0][0])

# (m, c), cov_matrix = np.polyfit(n_values, np.sin(theta_values), 1, full=False, cov=True, w=1/sin_theta_errors)  # weighted fit using angle uncertainties
x_vals = np.linspace(min(n_values), max(n_values), 100)
y_vals = m * x_vals + c

plt.plot(x_vals, y_vals, 'g--', label=f'fit: sin(θ) = {m:.4f}·n + {c:.4f}')
plt.errorbar(n_values, np.sin(theta_values), yerr=theta_errors, fmt='o', label='Data')
plt.xlabel('n')
plt.ylabel('sin(θ)')
plt.title('Order of Peaks vs sin(θ)')
plt.legend()

plt.savefig(f"Lambda Calculation Graphs/Order_vs_SinTheta_Dataset_{dataset}.png")
print(f"Saved graph for dataset {dataset}")

plt.show()

wavelength = m * d * 1e-3  # d in mm converted to m
delta_wavelength = wavelength * np.sqrt((m_error / m)**2 + (delta_d / d)**2)  # propagate uncertainty in wavelength
print(f"Calculated Wavelength: {wavelength:.4e} m ± {delta_wavelength:.4e} m")

# Save results to a CSV file, overwriting existing data if it is of the same dataset
import csv

# Read existing data
data = []
with open('Data.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    data = list(reader)

# Find if dataset exists and update, or append if not
found = False
for i, row in enumerate(data):
    if row and int(row[0]) == dataset:
        data[i] = [dataset, wavelength, delta_wavelength]
        found = True
        break
if not found:
    data.append([dataset, wavelength, delta_wavelength])

# Write back the updated data
with open('Data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Saved wavelength data for dataset {dataset} to Data.csv")