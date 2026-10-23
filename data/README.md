# Data

All experiments use the **IoT Greenhouse Irrigation Sensor Dataset** (version 1):

> I. Keok, "IoT Greenhouse Irrigation Sensor Dataset," Mendeley Data, V1, 2026.
> doi: [10.17632/2tvpsht4s4.1](https://doi.org/10.17632/2tvpsht4s4.1)

The dataset is released under CC BY 4.0, so a copy of `greenhouse_dataset.csv` is kept in this folder with attribution.
If you prefer to fetch it yourself, download it from the Mendeley page and place it here as `data/greenhouse_dataset.csv`.

Quick integrity check of the file used in the paper:

| Property | Value |
|---|---|
| SHA-256 | `e64a904a981d323f67210138cfc078e98d1911e6d2bb42195bf76aa818c51aff` |
| Size | 4,829,005 bytes |
| Rows (excluding header) | 95,669 |
| Separator | `;` |
| Columns | `Timestamp; Soil Percent; Soil Raw; Suhu Tanah; Suhu Udara; Humidity; Label` |
| Labelled rows | 53,209 (53,001 × 0, 208 × 1) |
| Unlabelled rows | 42,460 (not used) |

`Suhu Tanah` is soil temperature and `Suhu Udara` is air temperature (Indonesian).
The Mendeley description also mentions a confidence value produced by the logging system; that column is not present in this CSV and is never used as a predictor.

Two quirks matter for the analyses:
- **Timestamps** come in two layouts (`2026-04-23_04-51-49` and `2026-05-25 21:48:31`) and the file is not in time order. `src/prep.py` reads both layouts and sorts where time order matters.
- **DHT22 failure code.** In 4,770 unique labeled rows, Humidity is exactly 0 % and air temperature exactly 0.2 °C. These values come from a failed read of the DHT22 air sensor, not from the greenhouse climate. 135 of the 208 `Label = 1` rows carry this code. The rows are kept: the paper measures how much the models lean on them.
