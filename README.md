# SmartFarm - Crop Yield Prediction

IoT & Machine Learning application for predicting crop yield from sensor data.

## Features

- **Multiple ML Models**: KNN, Random Forest, SVR, and XGBoost
- **Interactive UI**: Streamlit-based web interface
- **Real-time Predictions**: Input hypothetical farm data and get instant yield predictions
- **Dataset**: Uses Smart Farming Crop Yield 2024 dataset

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/existenze/smart-farm.git
cd smart-farm
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

This app is ready for deployment on Streamlit Cloud. The dataset (`Smart_Farming_Crop_Yield_2024.csv`) is included in the repository.

### Deployment Steps:

1. **Push to GitHub**: Ensure all files are committed and pushed to your repository
2. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)
3. **Sign in** with your GitHub account
4. **Click "New app"**
5. **Configure**:
   - **Repository**: `existenze/smart-farm`
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `app.py`
6. **Click "Deploy"**

The app will be available at a URL like: `https://smart-farm-xxxxx.streamlit.app`

## Requirements

- Python 3.9+
- See `requirements.txt` for package dependencies

## Project Structure

- `app.py` - Main Streamlit application
- `Smart_Farming_Crop_Yield_2024.csv` - Dataset
- `smartfarm_*.py` - Individual ML model implementations
- `requirements.txt` - Python dependencies

## License

This project is for educational purposes.

