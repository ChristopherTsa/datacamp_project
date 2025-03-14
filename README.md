# RAMP starting kit for eVTOL Battery Degradation Prediction

This RAMP challenge focuses on predicting battery degradation in electric vertical takeoff and landing (eVTOL) vehicles based on cycling data.

## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Download the data

To download the dataset for this challenge, run the provided `download_data.py` script:

```bash
python download_data.py
```

This will:
1. Download the battery cycling dataset from Carnegie Mellon University's KiltHub repository
2. Create a `data` directory if it doesn't exist
3. Extract all files from the downloaded zip archive directly into the `data` directory
4. Keep the original zip file for reference
5. Create the dataset for the problem.

The script will show progress information and confirmation when the download and extraction are complete.

### Challenge description

The growth of electric aerial mobility has highlighted the importance of accurate battery health prediction. Battery degradation affects flight safety, mission planning, and maintenance costs for eVTOL vehicles. This challenge provides cycling data from Sony-Murata 18650 VTC-6 lithium-ion cells subjected to various operating conditions designed to simulate eVTOL flight profiles.

Your task is to predict the maximum discharge capacity (in mAh) of batteries based on features extracted from their cycling data. The discharge capacity is a key indicator of battery health and directly relates to the remaining useful life of the battery. Accurate predictions will help operators schedule maintenance and ensure safe operation of eVTOL aircraft.

### Dataset description

The dataset consists of battery cycling data from multiple cells (VAH01, VAH02, etc.), each subjected to different experimental protocols:
- Different cruise durations (400s, 600s, 1000s)
- Various charge currents (0.5C to 1.5C) 
- Different CV charge voltages (4.0V to 4.2V)
- Various operating temperatures (20°C to 35°C)
- Different power profiles during discharge

Each raw data file contains time series measurements of:
- Cell voltage
- Current
- Energy (charge and discharge)
- Capacity (charge and discharge)
- Temperature
- Cycle information

For the competition, we've extracted cycle-level features from the raw time series data, including minimum current, minimum voltage, and maximum temperature for each cycle.

### Prediction task

You must build a model to predict the `discharge_capacity` of batteries based on the provided features. This is a regression task evaluated using the Root Mean Square Error (RMSE).

Smaller RMSE value indicate better performance.

### Starting kit notebook

Get started with this RAMP challenge using the [battery degradation starting kit notebook](batteries_starting_kit.ipynb). The notebook guides you through:

1. Exploring the dataset
2. Visualizing battery cycling behavior
4. Feature engineering approaches
5. Building baseline models

### Making a submission

The submissions need to be located in the `submissions` folder. For instance,
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### Acknowledgements

The data was provided by Carnegie Mellon University researchers. We thank Alexander Bills, Shashank Sripad, Leif Fredericks, Matthew Guttenberg, Devin Charles, Evan Frank, and Venkat Viswanathan for their contribution to this dataset.

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)