# Kenya Road Accident Risk Dashboard

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

A machine learning powered dashboard that analyzes Kenya road accident 
patterns, predicts journey risk levels and delivers evidence based road 
safety recommendations — built on real NTSA accident data.

---

## The Problem

Kenya loses over 3,000 lives to road accidents every year. Most drivers 
have no way of knowing how dangerous a particular journey is before they 
set off. This project turns historical accident data into actionable 
risk intelligence.

---

## Live Dashboard

> Coming soon — deploying to Streamlit Cloud

---

## What This Dashboard Does

| Feature | Description |
|---|---|
| Accident Hotspot Map | Interactive Kenya map showing accident frequency and risk level by county |
| Risk Predictor | Input your county, road and travel time to get a personalised risk score |
| Analytics | Charts showing peak accident hours, dangerous roads and victim patterns |
| Recommendations | Evidence based safety advice for drivers, authorities and insurers |

---

## Key Findings from the Data

- **8pm is the most dangerous hour** - night driving accounts for a disproportionate share of casualties
- **Sunday is the deadliest day** - weekend fatigue and late night travel elevate risk significantly
- **Nairobi-Mombasa highway** has the highest recorded accident count of any road in the dataset
- **85% of victims are male** - male drivers aged 25 to 40 represent the highest risk demographic
- **Careless driving and speeding** are the two leading causes - both preventable

---

## Model Details

The risk prediction model classifies journeys as High or Low risk
based on time and location features.

| | |
|---|---|
| Algorithm | Random Forest Classifier |
| Features | Hour, night indicator, weekend indicator, peak hour indicator, county, road |
| Class Imbalance | Oversampling applied to training set only |
| Test Evaluation | Original unseen data - no data leakage |
| High Risk Recall | 100% — model catches every dangerous journey |
| Weighted F1 | 0.61 |
| Dataset | 1,119 records across 47 Kenya counties |

The model prioritises recall over precision for the high risk class.
In a road safety context, missing a dangerous journey is far more
costly than an unnecessary warning.

---

## Tech Stack

| | |
|---|---|
| Data Analysis | pandas, numpy |
| Visualisation | matplotlib, seaborn, folium |
| Machine Learning | scikit-learn, Random Forest |
| Dashboard | Streamlit |
| Mapping | Folium, streamlit-components |
| Language | Python 3.14 |

---

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/LauraOminde01/Kenya-Road-Accident-Risk.git
cd Kenya-Road-Accident-Risk

# Install dependencies
pip install -r requirements.txt

# Add your dataset
# Place accidents_clean.csv in data/clean/

# Run the dashboard
streamlit run app/app.py
```

---

## Limitations and Future Work

**Current limitations**
- Dataset contains 1,119 records - a larger dataset would improve model accuracy
- Missing features: weather conditions, road surface, vehicle type, visibility
- Model weighted F1 of 0.61 reflects the constraints of available data

**Planned improvements**
- Integrate real time weather data via API
- Expand dataset with more recent NTSA records
- Add Swahili language support for recommendations
- Build SMS based risk alerts for rural drivers without smartphones
- Expand coverage to Mombasa, Kisumu and cross border East Africa routes

---

## Data Source

National Transport and Safety Authority (NTSA) Kenya
Raw accident records including county, road, time, cause and victim data.

---

## Author

**Laura Ominde**  
Junior Data Scientist | Nairobi, Kenya  
Focus: Data science for real world impact in East Africa  
[GitHub](https://github.com/LauraOminde01)

---

## Acknowledgements

NTSA Kenya for maintaining public accident records.
This project is for educational and portfolio purposes only.
