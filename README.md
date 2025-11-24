# 🔬 NLP Q&A Research Dashboard

A modern, professional Streamlit application that provides comprehensive analysis of multiple Large Language Model responses. Compare answers from different APIs, analyze them with multiple summarization techniques, and perform advanced similarity analysis using three different approaches.

## ✨ Features

### 🤖 Multi-API Integration
- **Gemini** (Google) - Advanced reasoning capabilities
- **Groq** - Ultra-fast inference with Llama models  
- **HuggingFace** - Access to open-source models
- **Perplexity** - Real-time web search integration

### 📝 Advanced Summarization
- **Extractive Methods**: LexRank, Luhn, LSA
- **Abstractive**: BART transformer-based summarization
- **LLM-based**: Gemini-powered intelligent summarization

### 🔍 Three Similarity Analysis Methods
- **TF-IDF Cosine Similarity**: Lexical similarity analysis
- **Soft Cosine Similarity**: Word-level semantic similarity using Gensim
- **Sentence-BERT**: Context-aware semantic similarity

### 📊 Professional Dashboard
- Modern tabbed interface with clean design
- Interactive visualizations with Plotly
- Real-time API status monitoring
- Comprehensive performance metrics
- Research-grade comparison tools

## 🚀 Quick Start

1. **Clone and Setup Environment**
```bash
git clone <repository>
cd NLP_QNA_CURSOR
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure API Keys**
Copy the example secrets file and add your API keys:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml` with your actual API keys:
```toml
# Required API Keys
GEMINI_API_KEY = "your_gemini_api_key_here"
GROQ_API_KEY = "your_groq_api_key_here"
HF_TOKEN = "your_huggingface_token_here"
PERPLEXITY_API_KEY = "your_perplexity_api_key_here"
```

Or set environment variables:
```bash
export GEMINI_API_KEY="..."
export GROQ_API_KEY="..."
export HF_TOKEN="..."
export PERPLEXITY_API_KEY="..."
```

3. **Run the Application**
```bash
streamlit run app.py
```

## 🎯 Usage

1. **Configure Models**: Select which APIs to query in the sidebar
2. **Choose Summarizers**: Pick from extractive, abstractive, or LLM-based methods
3. **Ask Questions**: Enter your question in the main text area
4. **Analyze Results**: Explore answers, summaries, and similarity analysis across tabs

## 📊 Dashboard Sections

### 📋 Answers Tab
- Side-by-side comparison of all model responses
- Clean, organized display with model icons
- Error handling and status indicators

### 📝 Summaries Tab  
- Multiple summarization techniques applied
- Extractive vs abstractive comparison
- LLM-powered intelligent summarization

### 🔍 Similarity Analysis Tab
- Three similarity matrices with interactive heatmaps
- Statistical analysis of similarity measures
- Shared concept extraction across responses

### 📊 Dashboard Tab
- Key performance metrics
- Response length analysis
- Model performance comparison
- Research-grade statistics

## 🛠️ Technical Architecture

```
NLP_QNA_CURSOR/
├── apis/                    # API integrations
│   ├── gemini_api.py       # Google Gemini API
│   ├── groq_api.py         # Groq API  
│   ├── hf_api.py          # HuggingFace API
│   └── perplexity_api.py  # Perplexity API
├── summarizers/            # Summarization methods
│   ├── extractive.py      # LexRank, Luhn, LSA
│   ├── abstractive.py     # BART transformer
│   └── llm_summary.py     # LLM-based summarization
├── utils/                  # Analysis utilities
│   ├── compare.py         # Similarity analysis
│   └── helpers.py         # Helper functions
└── app.py                 # Main Streamlit application
```

## 🔧 Dependencies

- **Streamlit**: Web application framework
- **Transformers**: HuggingFace model integration
- **Sentence-Transformers**: Semantic similarity
- **Scikit-learn**: TF-IDF and cosine similarity
- **Gensim**: Soft cosine similarity
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

## 🎨 Design Features

- **Modern UI**: Clean, professional interface with subtle colors
- **Responsive Layout**: Optimized for different screen sizes  
- **Interactive Elements**: Hover effects, progress bars, status indicators
- **Icon Integration**: Visual cues for better UX
- **Metric Cards**: Highlighted key statistics
- **Tabbed Navigation**: Organized content sections

## 🔍 Research Applications

This dashboard is perfect for:
- **Model Comparison**: Evaluate different LLMs side-by-side
- **Summarization Research**: Compare extractive vs abstractive methods
- **Similarity Analysis**: Study lexical vs semantic similarity measures
- **Academic Research**: Generate research-grade comparisons
- **Educational Tools**: Teach NLP concepts through interactive analysis

## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect repository to Streamlit Cloud
3. Add secrets in the dashboard
4. Deploy!

### Local Development
```bash
streamlit run app.py --server.port 8501
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add new APIs, summarizers, or analysis methods
4. Submit a pull request

## 📝 License

MIT License - feel free to use for research and educational purposes.

## 🙏 Acknowledgments

- HuggingFace for transformer models
- Streamlit for the amazing web framework
- Google, Groq, and Perplexity for API access
- The open-source NLP community
