# Content-Based Music Recommendation System

## Abstract

This project implements a content-based collaborative filtering system for music recommendation using Spotify's audio feature dataset. The system leverages cosine similarity in standardized feature space to identify musically similar tracks, with an accompanying interactive interface for real-world deployment.

## Problem Statement

Traditional music recommendation systems rely heavily on collaborative filtering, which suffers from cold-start problems and requires extensive user interaction data. This project addresses these limitations by implementing a content-based approach that:

1. Operates solely on audio feature similarity
2. Provides interpretable recommendations
3. Functions immediately without user history
4. Maintains consistent performance across genres

## Methodology

### Algorithm Design

The recommendation system employs a feature-based similarity approach:

```
Input: User seed tracks T = {t₁, t₂, ..., tₙ}
Output: Ranked recommendation list R = {r₁, r₂, ..., rₖ}

1. Feature Extraction: F(T) → X ∈ ℝⁿˣᵈ where d=6
2. Standardization: Z = (X - μ) / σ  
3. Taste Vector: v = (1/n) Σᵢ₌₁ⁿ zᵢ
4. Similarity Computation: sim(v, zⱼ) = cos(v, zⱼ) ∀j ∈ dataset
5. Ranking: R = top-k(sim) \ T
```

### Feature Engineering

**Selected Features**: Based on music information retrieval research, we selected 6 audio features that capture complementary aspects of musical content:

- **Danceability** (0-1): Rhythmic stability and beat strength
- **Energy** (0-1): Perceptual intensity and dynamic range  
- **Tempo** (BPM): Speed in beats per minute
- **Valence** (0-1): Musical positivity/happiness conveyed
- **Acousticness** (0-1): Confidence measure of acoustic vs. electronic
- **Loudness** (dB): Overall volume and dynamic range

**Preprocessing**: StandardScaler normalization ensures equal feature contribution:
- Zero-mean centering: x̄ᵢ = xᵢ - μᵢ
- Unit variance scaling: zᵢ = x̄ᵢ / σᵢ

### Similarity Metric

**Cosine Similarity**: Chosen for orientation-based matching, robust to magnitude differences:

```
cos(u,v) = (u·v) / (||u|| ||v||)
```

This metric captures musical "direction" in feature space rather than absolute values, making it suitable for preference modeling.

## Data

### Dataset Description

**Source**: [30,000 Spotify Songs - Kaggle Dataset](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)  
**Created by**: Joe Beach Capital  
**Size**: 32,833 tracks across 6 major genres  
**License**: Open source for research and educational purposes  

**Dataset Composition**:
- **Total Tracks**: 32,833 unique songs
- **Audio Features**: 11 numerical features from Spotify Web API
- **Metadata**: Track names, artists, album information, playlist context
- **Genre Distribution**: 
  - Pop: ~5,507 tracks
  - Latin: ~5,155 tracks  
  - Rock: ~5,507 tracks
  - Rap: ~5,746 tracks
  - R&B: ~5,431 tracks
  - EDM: ~6,043 tracks

**Data Collection Method**: 
The dataset was compiled using Spotify's Web API, extracting tracks from popular genre-specific playlists. Each track includes comprehensive audio feature analysis performed by Spotify's algorithms, providing quantitative measures of musical characteristics.

### Audio Features (Spotify Web API)

The dataset includes 11 quantitative audio features extracted by Spotify's audio analysis algorithms:

| Feature | Range | Description |
|---------|--------|-------------|
| `danceability` | 0.0-1.0 | Rhythmic stability and beat predictability |
| `energy` | 0.0-1.0 | Perceptual measure of intensity and power |
| `speechiness` | 0.0-1.0 | Presence of spoken words in the track |
| `acousticness` | 0.0-1.0 | Confidence measure of acoustic vs. electronic |
| `instrumentalness` | 0.0-1.0 | Prediction of vocal content absence |
| `liveness` | 0.0-1.0 | Presence of audience in the recording |
| `valence` | 0.0-1.0 | Musical positivity/happiness conveyed |
| `loudness` | -60 to 0 dB | Overall volume and dynamic range |
| `tempo` | ~60-200 BPM | Track speed in beats per minute |
| `key` | 0-11 | Musical key (C, C#, D, etc.) |
| `mode` | 0-1 | Modality (major=1, minor=0) |

### Data Quality Assessment

```python
# Dataset completeness analysis
Shape: (32833, 23)
Missing values:
├── track_name: 5 (0.015%)
├── track_artist: 5 (0.015%)  
├── track_album_name: 5 (0.015%)
└── audio_features: 0 (0.000%)

# Feature distributions (selected subset)
Danceability: μ=0.65, σ=0.14, range=[0.0, 0.98]
Energy: μ=0.70, σ=0.18, range=[0.00, 1.00]  
Valence: μ=0.51, σ=0.23, range=[0.00, 0.99]
```

**Preprocessing Pipeline**:
1. **Missing Value Handling**: Linear interpolation for metadata fields
2. **Feature Selection**: 6 of 11 audio features retained for modeling
3. **Outlier Detection**: IQR method with 1.5× threshold
4. **Standardization**: Zero-mean, unit-variance scaling via StandardScaler
5. **Data Validation**: Integrity checks and range verification

## Repository Structure

```
├── notebooks/
│   └── Notebook.ipynb              # Primary analysis notebook
├── src/
│   ├── recommender.py              # Core recommendation engine
│   ├── search_utils.py             # Fuzzy search implementation
│   └── llm_utils.py               # LLM integration utilities
├── app/
│   └── streamlit_app.py           # Interactive web application
├── data/
│   ├── spotify_songs.csv          # Raw dataset
│   └── scaler.pkl                 # Fitted preprocessing pipeline
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

## Installation & Setup

### Environment Requirements

```bash
# System requirements
Python >= 3.8
Memory >= 4GB (for dataset loading)
Storage >= 100MB

# Core dependencies
pandas >= 2.0.0
numpy >= 1.24.0  
scikit-learn >= 1.3.0
rapidfuzz >= 3.0.0

# Optional (for interactive app)
streamlit >= 1.33.0
openai >= 1.30.0
```

### Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd music-recommender

# 2. Setup environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run analysis notebook
jupyter notebook notebooks/Notebook.ipynb

# 5. Launch interactive app (optional)
export OPENAI_API_KEY="your-api-key"
streamlit run app/streamlit_app.py
```

## Experiments & Analysis

### Option 1: Jupyter Notebook Analysis

**Best for**: Understanding the methodology, learning data science techniques, experimentation

```bash
jupyter notebook Notebook.ipynb
```

**What you'll find:**
- **Executive Summary**: Project overview and methodology
- **Data Exploration**: Schema analysis and quality assessment
- **Feature Engineering**: Selection strategy and standardization
- **User Preference Modeling**: Taste vector construction
- **Recommendation Engine**: Cosine similarity implementation
- **Results Analysis**: Quality assessment and future improvements

**Key Sections:**
1. Environment Setup and Imports
2. Data Loading and Initial Exploration
3. Feature Engineering and Data Preparation
4. User Preference Modeling
5. Optional: Neural Embedding Approach
6. Recommendation Engine Implementation
7. Results Analysis and Next Steps

### Option 2: Interactive Streamlit App

**Best for**: End-user experience, demonstrating the system, getting recommendations

```bash
streamlit run streamlit_app.py
```

**Features:**
- **Natural Language Input**: "I love Blinding Lights by The Weeknd and Rolling in the Deep by Adele"
- **Intelligent Search**: Fuzzy matching finds tracks even with typos or partial names
- **LLM Assistant**: ChatGPT helps clarify song requests and provides conversational experience
- **Real-time Recommendations**: Get 10 similar tracks instantly with similarity scores
- **Clean Interface**: Easy-to-use chat interface with clear results

**Example Workflow:**
1. Open the app in your browser (usually `http://localhost:8501`)
2. Type songs you like: "I enjoy upbeat pop songs like Shape of You and Can't Stop the Feeling"
3. The assistant will search the database and provide recommendations
4. Get detailed similarity scores and discover new music!

## 🛠️ Technical Details

### Core Algorithm

1. **Feature Selection**: 6 audio features chosen for musical similarity
2. **Standardization**: Zero-mean, unit-variance scaling using `StandardScaler`
3. **User Modeling**: Simple averaging of seed track features
4. **Similarity Computation**: Cosine similarity in standardized space
5. **Ranking**: Sort by similarity, exclude seed tracks

### Dataset

- **Source**: Spotify songs dataset
- **Size**: 32,833 tracks
- **Features**: 23 columns including audio features and metadata
- **Scope**: Multiple genres and playlists for diverse recommendations

### Technologies Used

- **Data Science**: pandas, numpy, scikit-learn
- **Machine Learning**: Cosine similarity, StandardScaler
- **Search**: RapidFuzz for fuzzy string matching
- **LLM Integration**: OpenAI API with function calling
- **Web Interface**: Streamlit
- **Visualization**: Jupyter notebooks

## 🎯 Example Use Cases

### Data Science Learning
```python
# Load the notebook to learn about:
# - Feature engineering strategies
# - Similarity metrics for recommendations
# - Data preprocessing best practices
# - Professional data science documentation
```

### Getting Music Recommendations
```
User: "I like electronic dance music, especially songs like Titanium and Animals"
Assistant: I'll search for those tracks and find similar electronic dance music for you!

Results:
1. Clarity - Zedd (similarity: 0.892)
2. Levels - Avicii (similarity: 0.884)
3. Bangarang - Skrillex (similarity: 0.879)
...
```

### Building Your Own Recommender
```python
from recommender import Recommender

# Initialize with your dataset
rec = Recommender("your_music_data.csv")

# Get recommendations
recommendations = rec.recommend_by_track_ids(
    example_track_ids=["track1", "track2"], 
    top_n=10
)
```

## 📈 Model Performance

### Similarity Score Interpretation
- **High similarity (>0.99)**: Near-identical audio profiles
- **Good matches (0.85-0.99)**: Strong musical similarity
- **Moderate matches (0.70-0.85)**: Decent recommendations worth exploring
- **Low matches (<0.70)**: May introduce beneficial diversity

### Limitations
- **Feature scope**: Only 6 audio features, missing genre/mood context
- **User modeling**: Simple averaging may not capture preference diversity
- **Cold start**: Requires seed tracks, no handling for new users
- **Popularity bias**: No consideration of track popularity or recency

## 🔮 Future Enhancements

### Technical Improvements
1. **Feature expansion**: Include genre, mood, lyrical themes
2. **Advanced user modeling**: Weighted averages, preference clustering
3. **Hybrid approaches**: Combine content-based with collaborative filtering
4. **Evaluation metrics**: Implement precision@k, diversity measures
5. **Production optimizations**: Approximate nearest neighbors, caching

### Application Features
1. **Playlist generation**: Auto-create themed playlists
2. **User profiles**: Save preferences and recommendation history
3. **Music discovery**: Explore new genres based on current taste
4. **Social features**: Share recommendations with friends

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References & Acknowledgments

### Dataset Citation
```
Beach Capital, Joe. (2023). 30,000 Spotify Songs Dataset. 
Kaggle. https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs
```

### Technical Dependencies
- **Spotify Web API**: Audio feature extraction and music analysis
- **scikit-learn**: Machine learning utilities and preprocessing
- **pandas/numpy**: Data manipulation and numerical computing
- **OpenAI API**: Natural language processing capabilities
- **Streamlit**: Interactive web application framework

### Research Context
This project builds upon established research in Music Information Retrieval (MIR) and content-based recommendation systems. The audio features used are derived from Spotify's comprehensive music analysis pipeline, which employs advanced signal processing and machine learning techniques.

## 📞 Support

If you encounter any issues or have questions:

1. **Check the notebook**: Detailed explanations of the methodology
2. **Review this README**: Comprehensive setup and usage instructions
3. **Open an issue**: Describe your problem with steps to reproduce
4. **Check dependencies**: Ensure all packages are installed correctly

---

**Built with ❤️ for music lovers and data science enthusiasts**
