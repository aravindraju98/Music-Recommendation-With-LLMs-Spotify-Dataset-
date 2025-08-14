# 🎵 Content-Based Music Recommender System Spotify Data


A comprehensive music recommendation system built with Spotify audio features, featuring both a Jupyter notebook for analysis and an interactive Streamlit web application with LLM-powered chat interface.

## 📊 Project Overview

This project demonstrates two approaches to music recommendation:

1. **Jupyter Notebook**: Professional data science analysis with detailed explanations
2. **Streamlit Web App**: Interactive chat interface powered by OpenAI's LLM for natural song search and recommendations

### Key Features

- **Content-Based Filtering**: Uses 6 core audio features (danceability, energy, tempo, valence, acousticness, loudness)
- **Intelligent Search**: Fuzzy matching to find songs from natural language descriptions
- **LLM Integration**: ChatGPT-powered assistant that understands music preferences
- **Similarity Analysis**: Cosine similarity in standardized feature space
- **Professional Documentation**: Complete data science workflow with explanations

## 🗂️ Project Structure

```
├── Notebook.ipynb          # Main analysis notebook
├── app.py                    # Original prototype script
├── streamlit_app.py          # Main Streamlit web application
├── recommender.py            # Core recommendation engine
├── search_utils.py           # Fuzzy search functionality
├── llm_utils.py             # OpenAI integration and tool calling
├── requirements.txt         # Python dependencies
├── spotify_songs.csv        # Dataset (32,833 tracks)
├── scaler.pkl              # Fitted StandardScaler (generated)
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (for the Streamlit app)
- Git (for cloning)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd music-recommender
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .\.venv\Scripts\Activate.ps1
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key** (for Streamlit app)
   ```bash
   # Option 1: Environment variable
   export OPENAI_API_KEY="sk-your-api-key-here"
   
   # Option 2: Create .env file
   echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
   ```

## 📚 Usage

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

## 🙏 Acknowledgments

- **Spotify**: For providing comprehensive audio feature data
- **OpenAI**: For enabling natural language interaction
- **Streamlit**: For the beautiful web interface framework
- **scikit-learn**: For machine learning utilities

## 📞 Support

If you encounter any issues or have questions:

1. **Check the notebook**: Detailed explanations of the methodology
2. **Review this README**: Comprehensive setup and usage instructions
3. **Open an issue**: Describe your problem with steps to reproduce
4. **Check dependencies**: Ensure all packages are installed correctly

---

**Built with ❤️ for music lovers and data science enthusiasts**
