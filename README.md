Bantay-Baha — Camarines Sur 🌧️🌊

Real-Time Rainfall & Flood Simulation Dashboard for Camarines Sur, Philippines

Bantay-Baha is a Streamlit-based geospatial flood-monitoring and simulation prototype focused specifically on Camarines Sur, Philippines. It is designed for rapid visualization of rainfall-driven inland flooding across the province.

It combines live weather observations, forecast rainfall, terrain elevation, a land/sea mask, and a terrain-based flow model to estimate potential flood depth over selected areas within Camarines Sur.

Important: Bantay-Baha is a prototype screening and visualization tool. Its simulated flood depth and risk classifications are not official PAGASA warnings, flood forecasts, or emergency instructions. Always rely on official PAGASA, LGU/MDRRMO, and other government advisories for operational decisions.

📍 Geographic Focus

This version of Bantay-Baha is designed specifically for Camarines Sur.

The application focuses its flood simulation and visualization on the province's inland and low-lying areas while using a Natural Earth land/sea mask to keep offshore areas outside the simulated flood surface.

The system is intended for local situational awareness and scenario exploration for communities, responders, researchers, and local stakeholders in Camarines Sur.

✨ Features

🌦️ Real-Time & Forecast Weather

Current weather conditions from Open-Meteo.

ECMWF IFS HRES is preferred when available, with Open-Meteo Best Match as fallback.

Hourly historical and forecast precipitation.

Up to 24 hours of rainfall-driven simulation.

Current temperature, humidity, wind, and rainfall indicators.

Rainfall summaries for 1h, 6h, 12h, and 24h periods.

Forecast rainfall summaries for the next 6h and 24h.

🌊 Rainfall-Driven Flood Simulation

The primary simulation is driven by precipitation rather than a manually imposed static water level.

The model combines:

Digital elevation data from SRTM.

Terrain-derived flow concentration using a D8-style approach.

Rainfall accumulation.

A runoff coefficient.

Terrain/flow concentration to estimate a flood-depth proxy.

The simulation can be viewed from the current hour through +24 hours.

🗺️ Geospatial Flood Visualization

Flood depth is displayed as a color gradient:

Simulated depth

Visualization

0–0.10 m

Green

0.10–0.30 m

Blue

0.30–0.75 m

Yellow

0.75–1.50 m

Red

1.50–2.50 m

Purple

2.50–4.00 m

Violet

>4.00 m

Deep violet

The map also includes:

Satellite and street basemaps.

Natural Earth land boundary overlay.

Flood-depth legend.

Correct handling of ocean/no-data cells.

Clickable map locations for point-level inspection.

📍 Point Inspection

Clicking the flood map reports information for the nearest simulation cell, including:

Latitude and longitude.

Land/sea status.

Terrain elevation.

Simulated flood depth.

Rainfall for the selected simulation hour.

24-hour rainfall context.

Simulation time.

Model risk classification.

Ocean and outside-land clicks are explicitly identified instead of being interpreted as flooded terrain.

🟡🟠🔴 PAGASA-Style Rainfall Classification

Bantay-Baha keeps rainfall classification separate from the experimental flood-depth model.

The rainfall indicator uses the following threshold scheme:

Rainfall rate

Classification

< 7.5 mm/h

Normal

7.5–<15 mm/h

Yellow

15–<30 mm/h

Orange

≥ 30 mm/h

Red

These thresholds are presented as PAGASA-style rainfall intensity categories for context. They should not be interpreted as an official PAGASA flood alert.

🧠 How the Simulation Works

Bantay-Baha currently uses a deterministic GIS/hydrologic screening approach rather than an AI/ML model.

1. Weather input

Hourly precipitation is retrieved for the selected location from Open-Meteo.

2. Rainfall scenario

The user selects the simulation horizon from the current hour through +24h. The application builds the rainfall scenario around that selected hour.

3. Terrain

SRTM elevation data provides the terrain surface used by the model.

4. Land/sea mask

Natural Earth 10m Land data is used to prevent the simulation from incorrectly treating ocean cells as land.

5. Flow concentration

A simplified D8-style terrain flow analysis identifies areas where runoff is more likely to concentrate.

6. Flood-depth proxy

Rainfall is converted to runoff using a configurable runoff relationship and combined with terrain/flow concentration to generate a simulated flood-depth proxy for the selected Camarines Sur area.

This is intended for visualization and scenario exploration, not calibrated hydraulic prediction.

🧩 Technology Stack

Python

Streamlit – web application UI

NumPy – numerical computation

Pandas – time-series handling

Requests – API requests

GeoPandas – geospatial processing

Shapely – geometric operations and land masking

Folium – interactive web mapping

streamlit-folium – Folium integration with Streamlit

Pillow – raster/image handling

Plotly – charts and data visualization

📡 Data Sources

Weather

Open-Meteo is used for current/historical/forecast weather and precipitation data.

The application prefers the ECMWF IFS HRES forecast where available and falls back to Open-Meteo Best Match.

Elevation

SRTM (Shuttle Radar Topography Mission) terrain/elevation data is used as the digital elevation model.

Land Boundary

Natural Earth 10m Land is used for the land/sea mask and boundary visualization:

https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/

📁 Project Structure

Bantay-Baha/
├── .streamlit/
│   └── config.toml
├── .gitignore
├── app_v2.py
├── requirements.txt
└── README.md

Main application

app_v2.py is the Streamlit entry point.

Dependencies

requirements.txt contains the Python packages required by the application and Streamlit Community Cloud deployment.

🚀 Run Locally

1. Clone the repository

git clone https://github.com/jaylab1995/bantay-baha.git
cd bantay-baha

2. Create a virtual environment

python3 -m venv venv
source venv/bin/activate

On Windows:

venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Start Streamlit

streamlit run app_v2.py

The terminal will provide the local Streamlit URL, normally:

http://localhost:8501

☁️ Streamlit Community Cloud Deployment

Bantay-Baha is structured for deployment on Streamlit Community Cloud.

Recommended settings:

Repository:  jaylab1995/bantay-baha
Branch:      main
Main file:   app_v2.py

The platform installs dependencies from:

requirements.txt

and uses:

.streamlit/config.toml

for application configuration.

🔧 Configuration Notes

Environment

The current application is designed to work without private API keys for its primary data sources.

Caching

Weather and terrain requests may be cached locally by Streamlit to reduce repeated downloads and improve responsiveness.

Ocean handling

Ocean cells are intentionally retained as no-data values. They are not converted to zero elevation and are not included in the flood-depth raster.

⚠️ Limitations

Bantay-Baha should be considered a research/prototyping and visualization system for Camarines Sur.

Current limitations include:

Single weather point – precipitation is currently sourced from the selected weather coordinate rather than a fully spatial rainfall radar/satellite grid.

Hydrologic simplification – the flood calculation is a runoff/concentration proxy, not a calibrated hydraulic model.

No river routing model – river discharge, channel capacity, bridge constraints, drainage networks, and reservoir operations are not explicitly modeled.

No local calibration – flood depth thresholds and terrain relationships have not been calibrated against observed flood events in Camarines Sur.

No official warning integration – the application does not replace PAGASA, PHIVOLCS, Camarines Sur LGU/MDRRMO, or emergency management systems.

Forecast uncertainty – rainfall forecasts can change significantly during severe weather events.

🛡️ Intended Use

Bantay-Baha is intended to support:

Situational awareness.

Rapid rainfall/flood scenario visualization.

GIS and disaster-risk research.

Demonstration of open geospatial data integration.

Exploration of rainfall-driven inland flooding.

Potential future integration with local government disaster-risk workflows.

It should not be used as the sole basis for evacuation, rescue, road closure, or other safety-critical decisions.

🔮 Planned Improvements

Potential future development areas include:

Spatial rainfall grids from radar/satellite products.

PAGASA rainfall and flood-warning integration.

Near-real-time rainfall observation layers.

River and stream gauge integration.

More advanced hydrologic/hydraulic modeling.

Local flood-event calibration.

Barangay-level impact analysis.

Road and critical-infrastructure exposure layers.

Evacuation-center and safe-route visualization.

Historical flood replay.

Automated alerts and notifications.

Mobile-first interface improvements.

Optional machine-learning models after a sufficiently large local training dataset is available.

📜 Disclaimer

Bantay-Baha is an experimental software project. Simulated flood depth, risk levels, rainfall classifications, maps, and forecasts may contain errors or uncertainty. The project does not constitute an official government warning or forecast.

For emergency decisions in the Philippines, consult official government sources and your local disaster-risk reduction and management authorities.

👨‍💻 Project

Bantay-Baha

Rainfall-driven flood monitoring and simulation for Camarines Sur, Philippines.

Repository:

https://github.com/jaylab1995/bantay-baha

📄 License

No explicit open-source license has been assigned to this repository yet.

Until a license is added, the repository contents should be treated as all rights reserved by the project author.
