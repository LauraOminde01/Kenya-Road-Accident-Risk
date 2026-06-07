# ============================================
# SAFEROUTE KENYA - ACCIDENT RISK PREDICTOR
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="SafeRoute Kenya - Accident Risk Predictor",
    layout="wide"
)

# ============================================
# LOAD DATA AND MODEL
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv('data/clean/accidents_clean.csv')
    df['hour'] = pd.to_numeric(
        df['TIME 24 HOURS'].astype(str).str.zfill(4).str[:2],
        errors='coerce'
    )
    df['hour'] = df['hour'].fillna(df['hour'].median())
    df['is_night'] = df['hour'].apply(
        lambda x: 1 if x >= 20 or x <= 5 else 0
    )
    df['is_weekend'] = df['day_of_week'].apply(
        lambda x: 1 if str(x) in ['Saturday', 'Sunday'] else 0
    )
    df['is_peak_hour'] = df['hour'].apply(
        lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0
    )
    df['COUNTY'] = df['COUNTY'].str.strip().str.upper()
    df['ROAD'] = df['ROAD'].str.strip().str.upper()
    return df

@st.cache_resource
def load_model():
    from sklearn.preprocessing import LabelEncoder

    with open('models/accident_risk_model.pkl', 'rb') as f:
        model = pickle.load(f)

    df_enc = pd.read_csv('data/clean/accidents_clean.csv')
    df_enc['COUNTY'] = df_enc['COUNTY'].str.strip().str.upper()
    df_enc['ROAD'] = df_enc['ROAD'].str.strip().str.upper()

    le_county = LabelEncoder()
    le_county.fit(df_enc['COUNTY'].dropna().unique())

    le_road = LabelEncoder()
    le_road.fit(df_enc['ROAD'].dropna().unique())

    return model, le_county, le_road

df = load_data()
model, le_county, le_road = load_model()

# ============================================
# COUNTY COORDINATES
# ============================================
county_coords = {
    'NAIROBI': (-1.2921, 36.8219),
    'MOMBASA': (-4.0435, 39.6682),
    'NAKURU': (-0.3031, 36.0800),
    'KISUMU': (-0.0917, 34.7680),
    'MACHAKOS': (-1.5177, 37.2634),
    'KILIFI': (-3.5107, 39.9093),
    'TAITA TAVETA': (-3.3167, 38.4833),
    'MAKUENI': (-1.8033, 37.6247),
    'KAJIADO': (-1.8520, 36.7820),
    'KIAMBU': (-1.0312, 36.8312),
    'NYERI': (-0.4167, 36.9500),
    'MERU': (0.0467, 37.6490),
    'EMBU': (-0.5300, 37.4500),
    'KITUI': (-1.3667, 38.0167),
    'GARISSA': (-0.4532, 39.6461),
    'WAJIR': (1.7471, 40.0573),
    'MANDERA': (3.9366, 41.8670),
    'MARSABIT': (2.3284, 37.9899),
    'ISIOLO': (0.3540, 37.5820),
    'LAIKIPIA': (0.2000, 36.7000),
    'SAMBURU': (1.2000, 36.9000),
    'TRANS NZOIA': (1.0167, 35.0000),
    'UASIN GISHU': (0.5500, 35.2700),
    'NANDI': (0.1833, 35.1167),
    'BARINGO': (0.4667, 35.9667),
    'KERICHO': (-0.3667, 35.2833),
    'BOMET': (-0.7833, 35.3500),
    'NYAMIRA': (-0.5667, 34.9333),
    'KISII': (-0.6817, 34.7667),
    'MIGORI': (-1.0634, 34.4731),
    'HOMA BAY': (-0.5167, 34.4500),
    'SIAYA': (0.0617, 34.2883),
    'KAKAMEGA': (0.2827, 34.7519),
    'VIHIGA': (0.0833, 34.7167),
    'BUNGOMA': (0.5635, 34.5594),
    'BUSIA': (0.4608, 34.1116),
    'TURKANA': (3.1130, 35.5670),
    'KWALE': (-4.1833, 39.4500),
    'NAROK': (-1.0833, 35.8667),
    'MURANGA': (-0.7167, 37.1500),
    'KIRINYAGA': (-0.5590, 37.3280),
    'NYANDARUA': (-0.1833, 36.3333),
    'HOMA-BAY': (-0.5167, 34.4500),
    'HOMABAY': (-0.5167, 34.4500),
    'TAITA-TAVETA': (-3.3167, 38.4833),
    'TRAN NZOIA': (1.0167, 35.0000),
    'UASIN NGISHU': (0.5500, 35.2700),
    'MAKURU': (-0.3031, 36.0800),
    'KERCHO': (-0.3667, 35.2833),
    'KILFI': (-3.5107, 39.9093),
    'MALINDI': (-3.2175, 40.1169),
    'MARAKWET': (0.9167, 35.5167),
    'MOYALE': (3.5210, 39.0560),
    'MWINGI': (-0.9333, 38.0667),
    'NYAHURURU': (0.0333, 36.3667),
    'TIGANIA': (0.1500, 37.8500),
}

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("SafeRoute Kenya")
st.sidebar.markdown("Data source: NTSA Kenya Accident Database")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Accident Hotspot Map",
        "Risk Predictor",
        "Analytics",
        "Recommendations"
    ]
)

# ============================================
# PAGE 0 - OVERVIEW
# ============================================
if page == "Overview":
    st.title("SafeRoute Kenya — Accident Risk Predictor")
    st.markdown(
        "This dashboard analyzes Kenya road accident data from the NTSA database. "
        "Use the sidebar to navigate between the map, risk predictor, analytics and recommendations."
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Accidents", f"{len(df):,}")
    col2.metric("Counties Affected", df['COUNTY'].nunique())
    col3.metric("Roads in Dataset", df['ROAD'].nunique())
    col4.metric("Average Victim Age", f"{df['AGE'].mean():.1f} yrs")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Most Dangerous County")
        top_county = df['COUNTY'].value_counts().index[0]
        top_county_count = df['COUNTY'].value_counts().iloc[0]
        st.markdown(f"**{top_county}** with **{top_county_count}** recorded accidents")

    with col2:
        st.subheader("Most Dangerous Hour")
        top_hour = int(df['hour'].value_counts().index[0])
        st.markdown(f"**{top_hour}:00** — late night driving is the highest risk period")

    st.markdown("---")
    st.subheader("Key Findings")
    st.markdown("""
    - 8pm is the peak accident hour — night driving is significantly more dangerous
    - Sunday has the highest accident rate of any day of the week
    - Male victims account for over 85% of all recorded casualties
    - The Nairobi-Mombasa highway is the most dangerous road in the dataset
    - Careless driving and speeding are the two leading causes of accidents
    """)

# ============================================
# PAGE 1 - ACCIDENT HOTSPOT MAP
# ============================================
elif page == "Accident Hotspot Map":
    st.title("Kenya Accident Hotspot Map")
    st.markdown("Circle size reflects accident volume. Color indicates risk level by county.")

    county_stats = df.groupby('COUNTY').agg(
        accident_count=('COUNTY', 'count'),
        avg_age=('AGE', 'mean'),
        night_accidents=('is_night', 'sum')
    ).reset_index()

    county_stats['night_pct'] = (
        county_stats['night_accidents'] / county_stats['accident_count'] * 100
    ).round(1)

    county_stats['risk_level'] = pd.cut(
        county_stats['accident_count'],
        bins=[0, 10, 50, 999],
        labels=['Low', 'Medium', 'High']
    )

    kenya_map = folium.Map(
        location=[-0.0236, 37.9062],
        zoom_start=6,
        tiles='CartoDB positron'
    )

    risk_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}

    for _, row in county_stats.iterrows():
        county = row['COUNTY']
        if county in county_coords:
            lat, lon = county_coords[county]
            risk = row['risk_level']
            color = risk_colors.get(str(risk), 'gray')

            popup_text = f"""
            <b>{county}</b><br>
            Accidents: {row['accident_count']}<br>
            Risk Level: {risk}<br>
            Night Accidents: {row['night_pct']}%<br>
            Avg Victim Age: {row['avg_age']:.1f} years
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=max(5, row['accident_count'] / 5),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(kenya_map)

    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
         background-color: white; padding: 15px; border-radius: 5px;
         border: 2px solid grey; font-size: 14px;">
         <b>Accident Risk Level</b><br>
         <span style="color:red;">&#9679;</span> High Risk<br>
         <span style="color:orange;">&#9679;</span> Medium Risk<br>
         <span style="color:green;">&#9679;</span> Low Risk<br>
         <br>
         <i>Circle size = number of accidents</i>
    </div>
    """

    kenya_map.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs('outputs', exist_ok=True)
    kenya_map.save('outputs/kenya_accident_map.html')

    with open('outputs/kenya_accident_map.html', 'r', encoding='utf-8') as f:
        map_html = f.read()

    components.html(map_html, width=900, height=550, scrolling=True)

# ============================================
# PAGE 2 - RISK PREDICTOR
# ============================================
elif page == "Risk Predictor":
    st.title("Accident Risk Predictor")
    st.markdown("Enter your journey details to get a personalised risk assessment.")

    col1, col2 = st.columns(2)

    with col1:
        county = st.selectbox(
            "Select County",
            sorted(df['COUNTY'].dropna().unique())
        )
        hour = st.slider("Hour of Travel (24hr format)", 0, 23, 8)
        day = st.selectbox("Day of Week", [
            'Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])

    with col2:
        road = st.selectbox(
            "Select Road",
            sorted(df['ROAD'].dropna().unique())
        )
        st.markdown("####")
        st.markdown(f"Selected time: **{hour}:00**")
        if hour >= 20 or hour <= 5:
            st.warning("Night time travel — higher risk period")
        elif 6 <= hour <= 9 or 16 <= hour <= 19:
            st.info("Peak hour travel — moderate risk period")
        else:
            st.success("Off peak travel — lower risk period")

    st.markdown("---")

    if st.button("Check My Risk Level", use_container_width=True):
        is_night = 1 if hour >= 20 or hour <= 5 else 0
        is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0
        is_peak = 1 if (6 <= hour <= 9) or (16 <= hour <= 19) else 0

        county_enc = le_county.transform([county])[0] if county in le_county.classes_ else 0
        road_enc = le_road.transform([road])[0] if road in le_road.classes_ else 0

        input_data = pd.DataFrame([{
            'hour': hour,
            'is_night': is_night,
            'is_weekend': is_weekend,
            'is_peak_hour': is_peak,
            'county_encoded': county_enc,
            'road_encoded': road_enc
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data).max()

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 'High':
                st.error(f"Risk Level: HIGH ({probability:.0%} confidence)")
            else:
                st.success(f"Risk Level: LOW ({probability:.0%} confidence)")

        with col2:
            st.metric("County", county)
            st.metric("Road", road)

        st.markdown("---")
        st.subheader("What This Means")

        if prediction == 'High':
            st.markdown("""
            Your journey falls into a high risk category based on historical accident patterns.

            **Recommended actions:**
            - Consider travelling at a different time if possible
            - Ensure your vehicle is in good condition before setting off
            - Avoid using your phone while driving
            - Keep to the speed limit especially on highways
            - Stay alert — fatigue is a major cause of night accidents
            - Ensure all passengers are wearing seatbelts
            """)
        else:
            st.markdown("""
            Your journey falls into a lower risk category based on historical accident patterns.

            **General road safety reminders:**
            - Always wear your seatbelt
            - Observe speed limits
            - Stay alert and avoid distractions
            - Keep a safe following distance
            - Ensure your vehicle lights are working properly
            """)

# ============================================
# PAGE 3 - ANALYTICS
# ============================================
elif page == "Analytics":
    st.title("Accident Analytics")
    st.markdown("Patterns and trends from the NTSA Kenya accident database.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Accidents", f"{len(df):,}")
    col2.metric("Night Accidents", f"{df['is_night'].sum():,}")
    col3.metric("Weekend Accidents", f"{df['is_weekend'].sum():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Dangerous Counties")
        county_counts = df['COUNTY'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(county_counts.index, county_counts.values, color='crimson')
        ax.set_xlabel("Number of Accidents")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Accidents by Hour of Day")
        hour_counts = df['hour'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(hour_counts.index, hour_counts.values, color='steelblue')
        ax.set_xlabel("Hour")
        ax.set_ylabel("Accidents")
        ax.axvspan(20, 23, alpha=0.2, color='red', label='High risk hours')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accidents by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday',
                     'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df['day_of_week'].value_counts().reindex(day_order)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(day_counts.index, day_counts.values, color='purple')
        ax.set_xticklabels(day_order, rotation=45)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Top 10 Dangerous Roads")
        road_counts = df['ROAD'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(road_counts.index, road_counts.values, color='darkorange')
        ax.set_xlabel("Number of Accidents")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

# ============================================
# PAGE 4 - RECOMMENDATIONS
# ============================================
elif page == "Recommendations":
    st.title("Road Safety Recommendations")
    st.markdown(
        "Evidence based recommendations derived from patterns in the NTSA accident database."
    )
    st.markdown("---")

    st.subheader("For Drivers")
    st.markdown("""
    **Avoid night driving where possible**
    Accidents peak between 8pm and midnight. If you must travel at night,
    reduce speed, use full headlights and take regular breaks on long journeys.

    **Extra caution on Sundays**
    Sunday has the highest accident rate of any day. Weekend travel combined
    with fatigue and social activities creates elevated risk conditions.

    **Know your high risk roads**
    The Nairobi-Mombasa highway accounts for the highest number of recorded
    accidents. Maintain safe following distances and avoid overtaking on bends.

    **Speed is the leading cause**
    Cause code 26 (speeding) is the second most common accident cause.
    Reducing speed by even 10km/h significantly reduces accident severity.

    **Motorcyclists need extra protection**
    Motorcyclists make up the largest victim category. Always wear a helmet
    and avoid carrying passengers on highways.
    """)

    st.markdown("---")
    st.subheader("For Transport Authorities")
    st.markdown("""
    **Deploy traffic enforcement at peak hours**
    Concentrate enforcement between 7am to 9am and 4pm to 8pm when accident
    frequency is highest.

    **Target Nairobi county resources**
    Nairobi accounts for the highest accident volume. Additional road
    markings, speed cameras and lighting on key corridors would reduce casualties.

    **Night time road safety campaigns**
    Data shows night driving is disproportionately dangerous. Public awareness
    campaigns targeting late night driving behaviour could have significant impact.

    **Focus on male drivers aged 25 to 35**
    The average victim age is 34 years and over 85% are male.
    Targeted driver education for this demographic would address the majority of incidents.
    """)

    st.markdown("---")
    st.subheader("For Insurers and Policymakers")
    st.markdown("""
    **Risk based insurance pricing**
    Time of day, road type and county should be factored into vehicle
    insurance pricing models. Night driving on highways represents
    significantly higher actuarial risk.

    **Parametric road safety bonds**
    Counties with consistently high accident rates could be targeted
    with infrastructure improvement bonds tied to measurable safety outcomes.

    **Data collection improvements**
    The current dataset lacks weather, road surface and vehicle condition
    data. Enriching NTSA records with these variables would significantly
    improve predictive modelling accuracy.
    """)

    st.markdown("---")
    st.info(
        "This dashboard was built using NTSA Kenya accident data. "
        "All recommendations are derived from statistical patterns in the data "
        "and should be considered alongside domain expertise from road safety professionals."
    )
