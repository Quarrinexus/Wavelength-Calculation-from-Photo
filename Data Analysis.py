import numpy as np
import matplotlib.pyplot as plt

# Load CSV (dataset (not needed), value, error)
datasets, values, errors = np.loadtxt("Data-backup.csv", delimiter=",", unpack=True)

weights = 1 / errors**2
weighted_mean = np.sum(weights * values) / np.sum(weights)
weighted_error = np.sqrt(1 / np.sum(weights))

chi2 = np.sum(((values - weighted_mean) / errors)**2)
reduced_chi2 = chi2 / (len(values) - 1)

value_nm = weighted_mean * 1e9
error_nm = weighted_error * 1e9

plt.errorbar(datasets, values * 1e9, yerr=errors * 1e9, fmt='o', label='Data with error bars')
plt.axhline(value_nm, color='r', linestyle='--', label=f'Weighted Mean: {value_nm:.2f} nm')
plt.xlabel('Dataset')
plt.ylabel('Value (nm)')
plt.title('Data Analysis')
plt.legend()
plt.show()

print(f"Final result: λ = ({value_nm:.2f} ± {error_nm:.3f}) nm")
print(f"Chi-squared: {chi2:.2f}, Reduced Chi-squared: {reduced_chi2:.2f}")