@@ -1,54 +0,0 @@
# Lorentz Invariance Violation in the Standard Model Extensions

This project investigates the effects of Lorentz Invariance Violation (LIV) in the Standard Model Extensions (SME) on the cross section of the Drell-Yan process. The code is designed to perform these calculations using various methods and tools from physics, including PDF sets and matrix rotations.

## Project Structure

- **`constants.py`**: Contains the physical constants required for the calculations, such as the Z boson mass, coupling constants, and other QCD-related values.

- **`functions.py`**: The core file containing the main computational functions for cross section calculations. This includes functions for numerical integration, calculating parton distribution functions (PDFs), and combining standard model and LIV effects.

- **`rotation.py`**: Defines the rotation matrices required for transforming between different coordinate frames, such as from the Standard Collisional Frame (SCF) to the Collider Frame (CMS). It includes Earth’s angular velocity and other geometric aspects relevant to the rotational matrices.

- **`cross_section.ipynb`**: A Jupyter notebook where the cross-section calculations are performed and explored interactively.

- **`likelihood.ipynb`**: Another Jupyter notebook where the likelihood estimation and related statistical analysis are carried out to evaluate the results of the cross-section calculations.

## Setup

It is recommended to use a Conda environment for this project. The following libraries are required:

- **LHAPDF**: Used for parton distribution functions. This can be installed via `conda install -c conda-forge lhapdf`.
- **Torch**: Required for tensor operations and matrix manipulations. Install it via `conda install pytorch`.
- **NumPy**: For basic numerical operations.
- **SciPy**: For numerical integration.
- **Matplotlib**: Used for plotting data.
- **Astropy**: For astronomical coordinate transformations, mainly used in `rotation.py`.

### Steps to Set Up:

1. Create a new Conda environment and activate it:
   ```bash
   conda create -n liv-sme python=3.8
   conda activate liv-sme
2. Install the required dependencies:
   ```bash
   conda install -c conda-forge lhapdf astropy numpy scipy matplotlib pytorch
3. Download the NNPDF set (e.g., NNPDF3.1 NNLO):
   ```bash
   lhapdf install NNPDF31_nnlo_as_0118
## Running the Code
1. Cross Section Calculations:
   The main calculations of the cross section can be found in the **`cross_section.ipynb`** notebook. Open the notebook and run the cells interactively to compute the cross section under various conditions.

2. Likelihood Calculations:
   For likelihood evaluation, open and run the cells in the **`likelihood.ipynb`** notebook. This will allow you to assess the statistical significance of the results obtained.

3. Customizing Constants:
   If needed, you can modify the constants in **`constants.py`** to use different values for the calculations.

4. Coordinate Frame Rotations:
   **`rotation.py`** contains functions for transforming between frames. You may adjust the rotation matrices and angles depending on the desired orientation and application.

## Usage Notes
* The **`sigma_full()`** function integrates over the full range of Q values to obtain the total cross section, while **`d_sigma()`** computes the differential cross section $\frac{d\sigma}{dQ^2}$.