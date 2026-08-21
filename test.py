import mne
from mne.datasets import eegbci

paths = eegbci.load_data(subjects=[1], runs=[1, 2], path="./data")
raw = mne.io.read_raw_edf(paths[0], preload=True)
print(raw.info)