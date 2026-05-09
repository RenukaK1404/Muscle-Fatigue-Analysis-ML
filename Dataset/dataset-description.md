# Dataset Description

The dataset consists of physiological and lifestyle-related parameters used for estimating muscle fatigue and Maximum Voluntary Contraction (MVC). The data set used in this study was downloaded from Mendeley Data: 'Reference data on hand grip and lower limb strength using the Nintendo Wii Balance Board: A cross-sectional study of 354 subjects from 20 to 99 years of age' (Eika & Blomkvist, 2019, DOI: 10.17632/zptvcx6yrx.1). 

## Parameters Used

- Age
- Gender
- Work Activity
- Spare Time Activity
- Smoking History
- Grip Force Measurements
- Muscle Strength Indicators

| Attribute | Type | Range / Categories | Description |
|---|---|---|---|
| Age (years) | Continuous | 20–99 | Chronological age of the subject |
| Gender (M/F) | Binary | 1 = Male, 0 = Female | Biological sex of the subject |
| Pack-years | Continuous | 0–40+ | Cumulative smoking exposure |
| Activity Level – Work | Ordinal | 1–4 (Sedentary to Heavy) | Physical demand of the occupation |
| Activity Level – Spare | Ordinal | 1–4 (Inactive to Athletic) | Recreational physical activity |
| RFD (N) | Continuous | ~500–1200 | Right-hand force, dominant |
| RFND (N) | Continuous | ~500–1200 | Right-hand force, non-dominant |
| RHD (N) | Continuous | ~400–900 | Dynamometer reading, dominant |
| RHND (N) | Continuous | ~400–900 | Dynamometer reading, non-dominant |

The dataset was processed and analyzed for statistical modeling and machine learning-based fatigue prediction.
