# Complete Enhanced app.py - With question-to-text similarity using all 3 methods
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from apis.gemini_api import query_gemini
from apis.hf_api import query_hf
from apis.groq_api import query_groq
from apis.perplexity_api import query_perplexity
from summarizers.extractive import summarize_lexrank, summarize_luhn, summarize_lsa
from summarizers.abstractive import AbstractiveSummarizer
from summarizers.llm_summary import summarize_with_gemini
from utils.compare import (
    extract_shared_ngrams,
    similarity_matrix,
    calculate_all_similarities,
    get_similarity_stats,
    calculate_question_similarities  # NEW FUNCTION
)
from utils.metrics import evaluate_text


# Page configuration
st.set_page_config(
    page_title="Question-Answering and Summarization System",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize abstractive summarizer with caching
@st.cache_resource
def load_abstractive_summarizer_for_language(lang: str):
    """Cache an abstractive summarizer per target language."""
    try:
        model_key = "bart-cnn" if lang in ("en", "english") else "mt5-small"
        return AbstractiveSummarizer(model_key=model_key)
    except Exception as e:
        st.error(f"Failed to load abstractive summarizer for {lang}: {e}")
        return None


# Custom CSS for light theme with MUCH LARGER fonts
st.markdown("""
<style>
    /* Light background styling */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%);
        background-attachment: fixed;
    }
    
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(226, 232, 240, 0.5);
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(226, 232, 240, 0.5);
    }
    
    /* MUCH LARGER FONT SIZES */
    .main-header {
        font-size: 4rem !important;
        font-weight: 900 !important;
        color: #1e293b !important;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        line-height: 1.1;
    }
    
    .sub-header {
        font-size: 2.5rem !important;
        color: #475569 !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700 !important;
    }
    
    /* Default text much larger */
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        line-height: 1.6;
    }
    
    /* Headers even bigger */
    .stMarkdown h1 {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        color: #0f172a !important;
    }
    
    .stMarkdown h2 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }
    
    .stMarkdown h3 {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }
    
    /* Form elements larger */
    .stSelectbox label, .stMultiselect label, .stSlider label, .stTextArea label {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #1e293b !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiselect div[data-baseweb="select"] > div {
        font-size: 1.8rem !important;
    }
    
    .stTextArea textarea {
        font-size: 1.8rem !important;
    }
    
    .stButton button {
        font-size: 2rem !important;
        font-weight: 800 !important;
        padding: 1.5rem 4rem !important;
        border-radius: 15px !important;
    }
    
    /* Expander headers larger */
    .stExpander .streamlit-expanderHeader {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #1e293b !important;
    }
    
    /* Content inside expanders */
    .stExpander .streamlit-expanderContent {
        font-size: 1.8rem !important;
    }
    
    /* Metric cards */
    .stMetric label {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #1e293b !important;
    }
    
    .stMetric div[data-testid="metric-value"] {
        font-size: 3rem !important;
        font-weight: 900 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(248, 250, 252, 0.8);
        border-radius: 20px;
        padding: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 80px;
        padding-left: 35px;
        padding-right: 35px;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px 15px 0px 0px;
        border-bottom: 4px solid #e2e8f0;
        font-size: 2rem !important;
        font-weight: 800 !important;
        transition: all 0.3s ease;
        color: #475569 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Data tables */
    .stDataFrame {
        font-size: 1.6rem !important;
    }
    
    /* Checkbox styling */
    .stCheckbox label {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }
    
    .stRadio div[role="radiogroup"] label {
        font-size: 1.8rem !important;
    }
    
    /* Page navigation styling */
    .page-nav {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Info/success/error messages */
    .stAlert {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    
    .stSuccess, .stError, .stWarning, .stInfo {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# Header
st.markdown('<h1 class="main-header">🔬 Question-Answering and Summarization System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-LLM Comparison & Advanced Text Analysis Platform</p>', unsafe_allow_html=True)


# Page Navigation
st.markdown('<div class="page-nav">', unsafe_allow_html=True)
page = st.radio(
    "Navigate to:",
    ["🏠 Home - Q&A Analysis", "📝 Summaries", "🔍 Similarity Analysis", "📊 Comparison Dashboard"],
    horizontal=True,
    key="page_selector"
)
st.markdown('</div>', unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Optional API Status (hidden by default)
    show_api_status = st.checkbox("🔑 Show API Status", value=False)
    if show_api_status:
        st.subheader("🔑 API Status")
        api_status = {
            "Gemini": "✅" if st.secrets.get("GEMINI_API_KEY") else "❌",
            "Groq": "✅" if st.secrets.get("GROQ_API_KEY") else "❌", 
            "HuggingFace": "✅" if st.secrets.get("HF_TOKEN") else "❌",
            "Perplexity": "✅" if st.secrets.get("PERPLEXITY_API_KEY") else "❌"
        }
        
        for api, status in api_status.items():
            st.write(f"{status} {api}")
        
        st.divider()
    
    # Language Selection - Default to English
    st.subheader("🌐 Language")
    LANGUAGE_OPTIONS = {
        "English": "en",
        "Hindi (हिंदी)": "hi",
        "Telugu (తెలుగు)": "te",
        "Tamil (தமிழ்)": "ta",
        "Bangla (বাংলা)": "bn",
        "Gujarati (ગુજરાતી)": "gu",
        "Kannada (ಕನ್ನಡ)": "kn",
        "Dogri (डोगरी)": "doi",
    }
    # Default to English (index=0)
    selected_language_label = st.selectbox("Choose language", list(LANGUAGE_OPTIONS.keys()), index=0)
    selected_language = LANGUAGE_OPTIONS[selected_language_label]
    st.session_state["selected_language"] = selected_language
    # Ensure default is English if not set
    if not st.session_state.get("selected_language"):
        st.session_state["selected_language"] = "en"

    # Model Selection
    st.subheader("🤖 Model Selection")
    AVAILABLE_MODELS = {
        "Gemini Flash": {"api": "gemini", "model": "gemini-2.0-flash", "icon": "🧠"},
        "Groq Llama 3.3 70B": {"api": "groq", "model": "llama-3.3-70b-versatile", "icon": "⚡"},
        "Groq Mixtral 8x7B": {"api": "groq", "model": "mixtral-8x7b-32768", "icon": "⚡"},
        "Groq Gemma 7B": {"api": "groq", "model": "gemma-7b-it", "icon": "⚡"},
        "Groq Qwen 2.5 72B": {"api": "groq", "model": "qwen2.5-72b-instruct", "icon": "⚡"},
        "HuggingFace BART": {"api": "hf", "model": "facebook/bart-large-cnn", "icon": "🤗"},
        "Perplexity Sonar": {"api": "perplexity", "model": "sonar", "icon": "🔎"}
    }
    
    selected_models = st.multiselect(
        "Choose models to query",
        list(AVAILABLE_MODELS.keys()),
        default=["Gemini Flash", "Groq Llama 3.3 70B"]
    )
    
    st.divider()
    
    # Summarization Settings
    st.subheader("📝 Summarization")
    summarizer_choices = st.multiselect(
        "Choose techniques",
        ["lexrank", "luhn", "lsa", "abstractive", "llm_gemini"],
        default=["lexrank", "abstractive", "llm_gemini"],
        format_func=lambda x: {
            "lexrank": "📊 LexRank (Graph-based)",
            "luhn": "📈 Luhn (Frequency-based)", 
            "lsa": "🔍 LSA (Latent Semantic)",
            "abstractive": "🧠 BART (Abstractive)",
            "llm_gemini": "✨ LLM Summary (Gemini)"
        }[x]
    )
    
    num_sentences = st.slider("Sentences for extractive methods", 1, 5, 2)


# Main Input Area
st.subheader("❓ Your Question")
question = st.text_area(
    "Enter your question here...",
    height=125,
    placeholder="What is machine learning? How does it differ from traditional programming?"
)


# Initialize abstractive summarizer if needed
abstractive_summarizer = None
if "abstractive" in summarizer_choices:
    with st.spinner("🔧 Loading abstractive summarizer for selected language..."):
        abstractive_summarizer = load_abstractive_summarizer_for_language(st.session_state.get("selected_language", "en"))
    if abstractive_summarizer:
        st.success("✅ Abstractive summarizer loaded!")
    else:
        st.error("❌ Failed to load abstractive summarizer")


# Process Button
if st.button("🚀 Analyze Question", type="primary", use_container_width=True):
    if not question.strip():
        st.error("Please enter a question first!")
    elif not selected_models:
        st.error("Please select at least one model!")
    else:
        # Initialize session state
        if 'results' not in st.session_state:
            st.session_state.results = {}
        
        # Query APIs
        with st.spinner("🔄 Querying selected models..."):
            answers = {}
            progress_bar = st.progress(0)
            
            for i, display_name in enumerate(selected_models):
                info = AVAILABLE_MODELS[display_name]
                icon = info["icon"]
                
                try:
                    if info["api"] == "gemini":
                        ans = query_gemini(info["model"], question, as_text=True)
                    elif info["api"] == "groq":
                        ans = query_groq(info["model"], question)
                    elif info["api"] == "perplexity":
                        ans = query_perplexity(info["model"], question)
                    else:
                        ans = query_hf(info["model"], question)
                        
                    answers[display_name] = ans
                    st.success(f"{icon} {display_name} completed")
                    
                except Exception as e:
                    answers[display_name] = f"[ERROR: {str(e)[:100]}...]"
                    st.error(f"{icon} {display_name} failed: {str(e)[:50]}...")
                    
                
                progress_bar.progress((i + 1) / len(selected_models))
        
        # Store results in session state
        st.session_state.results['answers'] = answers
        st.session_state.results['question'] = question
        st.session_state.results['selected_models'] = selected_models
        
        # Generate summaries
        if summarizer_choices:
            with st.spinner("📝 Generating summaries..."):
                combined_text = "\n\n".join([f"{name}: {txt}" for name, txt in answers.items()])
                summaries = {}


                if "lexrank" in summarizer_choices:
                    try:
                        summaries["LexRank"] = summarize_lexrank(
                            combined_text,
                            num_sentences,
                            st.session_state.get("selected_language", "en")
                        )
                    except Exception as e:
                        summaries["LexRank"] = f"[Error: {str(e)}]"
                        
                if "luhn" in summarizer_choices:
                    try:
                        summaries["Luhn"] = summarize_luhn(
                            combined_text,
                            num_sentences,
                            st.session_state.get("selected_language", "en")
                        )
                    except Exception as e:
                        summaries["Luhn"] = f"[Error: {str(e)}]"
                        
                if "lsa" in summarizer_choices:
                    try:
                        summaries["LSA"] = summarize_lsa(
                            combined_text,
                            num_sentences,
                            st.session_state.get("selected_language", "en")
                        )
                    except Exception as e:
                        summaries["LSA"] = f"[Error: {str(e)}]"
                        
                # FIX THE ABSTRACTIVE PART HERE:
                if "abstractive" in summarizer_choices and abstractive_summarizer:
                    try:
                        summaries["BART Abstractive"] = abstractive_summarizer.summarize(
                            combined_text,
                            language=st.session_state.get("selected_language", "en")
                        )
                    except Exception as e:
                        summaries["BART Abstractive"] = f"[Error: {str(e)}]"
                elif "abstractive" in summarizer_choices:
                    summaries["BART Abstractive"] = "[Error: Abstractive model not loaded]"
                    
                if "llm_gemini" in summarizer_choices:
                    try:
                        summaries["LLM Summary"] = summarize_with_gemini(combined_text, lang_hint=st.session_state.get("selected_language", "en"))
                    except Exception as e:
                        summaries["LLM Summary"] = f"[Error: {str(e)}]"
                
                st.session_state.results['summaries'] = summaries
        else:
            summaries = {}
            st.session_state.results['summaries'] = summaries
        
        # NEW: Calculate question-to-text similarities using all 3 methods
        with st.spinner("🔍 Calculating question-to-text similarities..."):
            # Prepare all texts and labels
            all_texts = []
            all_labels = []
            
            # Add answers
            for name, text in answers.items():
                all_texts.append(text)
                all_labels.append(f"Answer: {name}")
            
            # Add summaries
            for method, text in summaries.items():
                all_texts.append(text)
                all_labels.append(f"Summary: {method}")
            
            # Calculate similarities using all 3 methods
            question_sims = calculate_question_similarities(question, all_texts, all_labels)
            st.session_state.results['question_similarities'] = question_sims
            st.session_state.results['text_labels'] = all_labels
        
        st.success("✅ Analysis complete! Navigate through the tabs below to explore results.")


# Page Content Based on Navigation
if page == "🏠 Home - Q&A Analysis":
    # Main Q&A Analysis Page
    if 'results' in st.session_state and st.session_state.results:
        st.header("📋 Model Answers")
        
        # Create columns for answers
        answers = st.session_state.results['answers']
        
        if len(answers) <= 2:
            cols = st.columns(len(answers))
            for i, (name, text) in enumerate(answers.items()):
                with cols[i]:
                    icon = AVAILABLE_MODELS[name]["icon"]
                    with st.expander(f"{icon} {name}", expanded=True):
                        st.markdown(text)
        else:
            # For more than 2 models, show in rows
            for name, text in answers.items():
                icon = AVAILABLE_MODELS[name]["icon"]
                with st.expander(f"{icon} {name}", expanded=False):
                    st.markdown(text)
    else:
        st.info("👆 Configure your settings in the sidebar and click 'Analyze Question' to begin your NLP research!")


elif page == "📝 Summaries":
    if 'results' in st.session_state and st.session_state.results and 'summaries' in st.session_state.results:
        st.header("📝 Summaries")
        
        summaries = st.session_state.results['summaries']
        
        if len(summaries) <= 2:
            cols = st.columns(len(summaries))
            for i, (method, summary) in enumerate(summaries.items()):
                with cols[i]:
                    with st.expander(f"📊 {method}", expanded=True):
                        st.markdown(summary)
        else:
            # For more than 2 summaries, show in rows
            for method, summary in summaries.items():
                with st.expander(f"📊 {method}", expanded=False):
                    st.markdown(summary)
    else:
        st.warning("No summaries available. Run analysis on the Home page first.")


elif page == "🔍 Similarity Analysis":
    if 'results' in st.session_state and st.session_state.results:
        st.header("🔍 Similarity Analysis")
        
        answers = st.session_state.results['answers']
        texts = list(answers.values())
        model_names = list(answers.keys())
        
        if len(texts) >= 2:
            # Calculate all similarity measures
            st.info("🔄 Computing similarity matrices using three different techniques...")
            similarity_results = calculate_all_similarities(texts, model_names)
            
            # Create tabs for each similarity method
            similarity_tabs = st.tabs(["📊 All Methods Overview", "🔤 TF-IDF Cosine", "🔀 Soft Cosine", "🧠 Sentence-BERT"])
            
            with similarity_tabs[0]:
                st.subheader("📈 Similarity Comparison Across Methods")
                
                # Create comparison metrics
                cols = st.columns(3)
                for i, (method, data) in enumerate(similarity_results.items()):
                    with cols[i]:
                        stats = get_similarity_stats(data['matrix'], method)
                        st.metric(
                            label=f"{method} (Avg)",
                            value=f"{stats['mean']:.3f}",
                            delta=f"σ: {stats['std']:.3f}",
                            help=data['description']
                        )
                
                # Summary table
                st.subheader("📋 Detailed Statistics")
                stats_data = []
                for method, data in similarity_results.items():
                    stats = get_similarity_stats(data['matrix'], method)
                    stats_data.append(stats)
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
            
            # Individual method tabs
            for i, (method, data) in enumerate(similarity_results.items(), 1):
                with similarity_tabs[i]:
                    st.subheader(f"{method} Similarity Matrix")
                    st.write(f"**Method:** {data['description']}")
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=data['matrix'],
                        x=model_names,
                        y=model_names,
                        colorscale='RdYlBu_r',
                        showscale=True,
                        text=np.round(data['matrix'], 3),
                        texttemplate="%{text}",
                        textfont={"size": 20},
                        hovertemplate="%{x} vs %{y}<br>Similarity: %{z:.3f}<extra></extra>"
                    ))
                    
                    fig.update_layout(
                        title=f"{method} Similarity Heatmap",
                        xaxis_title="Models",
                        yaxis_title="Models",
                        height=500,
                        font=dict(size=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Method-specific insights
                    stats = get_similarity_stats(data['matrix'], method)
                    
                    insight_cols = st.columns(4)
                    with insight_cols[0]:
                        st.metric("Mean Similarity", f"{stats['mean']:.3f}")
                    with insight_cols[1]:
                        st.metric("Std Deviation", f"{stats['std']:.3f}")
                    with insight_cols[2]:
                        st.metric("Min Similarity", f"{stats['min']:.3f}")
                    with insight_cols[3]:
                        st.metric("Max Similarity", f"{stats['max']:.3f}")
            
            # Shared concepts analysis
            st.divider()
            st.subheader("🔗 Shared Concepts Analysis")
            
            concept_cols = st.columns(2)
            
            with concept_cols[0]:
                st.write("**Common 2-grams (phrases):**")
                shared_bigrams = extract_shared_ngrams(texts, n=2, min_occurrence=2)
                if shared_bigrams:
                    for i, bigram in enumerate(shared_bigrams[:10], 1):
                        st.write(f"{i}. {bigram}")
                else:
                    st.info("No common 2-grams found across responses.")
            
            with concept_cols[1]:
                st.write("**Common 3-grams (phrases):**")
                shared_trigrams = extract_shared_ngrams(texts, n=3, min_occurrence=2)
                if shared_trigrams:
                    for i, trigram in enumerate(shared_trigrams[:10], 1):
                        st.write(f"{i}. {trigram}")
                else:
                    st.info("No common 3-grams found across responses.")
        
        else:
            st.warning("⚠️ Need at least 2 model responses to perform similarity analysis.")
            st.info("Please go back to the Home page and select at least 2 models for analysis.")
    
    else:
        st.warning("❌ No analysis results available.")
        st.info("Please run the analysis on the Home page first by entering a question and selecting models.")


elif page == "📊 Comparison Dashboard":
    if 'results' in st.session_state and st.session_state.results:
        st.header("📊 Research Dashboard")
        
        answers = st.session_state.results['answers']
        summaries = st.session_state.results.get('summaries', {})
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Models Queried", len(answers))
        
        with col2:
            total_chars = sum(len(text) for text in answers.values())
            st.metric("Total Response Length", f"{total_chars:,} chars")
        
        with col3:
            avg_length = total_chars / len(answers) if answers else 0
            st.metric("Average Response", f"{avg_length:.0f} chars")
        
        with col4:
            if len(answers) >= 2:
                # Quick similarity check
                texts = list(answers.values())
                quick_sim = similarity_matrix(texts)
                # Get average off-diagonal similarity
                mask = ~np.eye(len(texts), dtype=bool)
                avg_sim = np.mean(np.array(quick_sim)[mask])
                st.metric("Avg Similarity", f"{avg_sim:.3f}")
            else:
                st.metric("Avg Similarity", "N/A")
        
        st.divider()
        
        # NEW: Enhanced comparison table with question similarities
        st.subheader("🎯 Comprehensive Comparison Table")
        
        if 'question_similarities' in st.session_state.results:
            question_sims = st.session_state.results['question_similarities']
            text_labels = st.session_state.results['text_labels']
            
            # Build comprehensive table
            rows = []
            text_idx = 0
            
            # Add answers with similarities
            for name, text in answers.items():
                rows.append({
                    'Type': 'Answer',
                    'Model/Method': name,
                    'Length (chars)': len(text),
                    'Words': len(text.split()),
                    'TF-IDF Sim': round(question_sims['TF-IDF'][text_idx], 3),
                    'Soft Cosine Sim': round(question_sims['Soft Cosine'][text_idx], 3),
                    'SBERT Sim': round(question_sims['Sentence-BERT'][text_idx], 3),
                    'Avg Similarity': round(np.mean([
                        question_sims['TF-IDF'][text_idx],
                        question_sims['Soft Cosine'][text_idx], 
                        question_sims['Sentence-BERT'][text_idx]
                    ]), 3)
                })
                text_idx += 1
            
            # Add summaries with similarities
            for method, text in summaries.items():
                rows.append({
                    'Type': 'Summary',
                    'Model/Method': method,
                    'Length (chars)': len(text),
                    'Words': len(text.split()),
                    'TF-IDF Sim': round(question_sims['TF-IDF'][text_idx], 3),
                    'Soft Cosine Sim': round(question_sims['Soft Cosine'][text_idx], 3),
                    'SBERT Sim': round(question_sims['Sentence-BERT'][text_idx], 3),
                    'Avg Similarity': round(np.mean([
                        question_sims['TF-IDF'][text_idx],
                        question_sims['Soft Cosine'][text_idx],
                        question_sims['Sentence-BERT'][text_idx]
                    ]), 3)
                })
                text_idx += 1
            
            df = pd.DataFrame(rows)
            
            # Sort by average similarity (highest first)
            df = df.sort_values('Avg Similarity', ascending=False)
            
            st.dataframe(df, use_container_width=True)
            
            # Show best performing answer and summary
            st.subheader("🏆 Best Performers")
            
            col1, col2 = st.columns(2)
            
            with col1:
                best_answer = df[df['Type'] == 'Answer'].iloc[0] if len(df[df['Type'] == 'Answer']) > 0 else None
                if best_answer is not None:
                    st.metric(
                        "🥇 Best Answer",
                        best_answer['Model/Method'],
                        f"Similarity: {best_answer['Avg Similarity']}"
                    )
            
            with col2:
                best_summary = df[df['Type'] == 'Summary'].iloc[0] if len(df[df['Type'] == 'Summary']) > 0 else None
                if best_summary is not None:
                    st.metric(
                        "🥇 Best Summary",
                        best_summary['Model/Method'],
                        f"Similarity: {best_summary['Avg Similarity']}"
                    )
        
        else:
            # Fallback to basic table if similarities not calculated
            performance_data = []
            for name, text in answers.items():
                performance_data.append({
                    'Type': 'Answer',
                    'Model/Method': name,
                    'Length (chars)': len(text),
                    'Words': len(text.split()),
                    'Has Error': '[ERROR' in text or '[Error' in text
                })
            
            for method, text in summaries.items():
                performance_data.append({
                    'Type': 'Summary', 
                    'Model/Method': method,
                    'Length (chars)': len(text),
                    'Words': len(text.split()),
                    'Has Error': '[ERROR' in text or '[Error' in text
                })
            
            df = pd.DataFrame(performance_data)
            st.dataframe(df, use_container_width=True)
        
        # Metrics evaluation (optional reference)
        st.divider()
        st.subheader("🧪 Evaluation Metrics (ROUGE, BLEU, BERTScore)")
        with st.expander("Add a reference answer (gold) to evaluate outputs", expanded=False):
            st.markdown(
                "- ROUGE and BLEU compare to a reference summary; they are reference-based.\n"
                "- BERTScore uses embeddings and is more semantic, but still needs a reference to compare against.\n"
                "- Use a human-written gold answer if you have one. Otherwise, you may pick one of the generated summaries below as a temporary reference."
            )
            ref_text = st.text_area("Reference answer (same language as question)", height=150, key="reference_answer")
            # Convenience: pick a generated output as reference
            with st.popover("Or pick an existing output as reference"):
                options = [f"Answer: {k}" for k in answers.keys()] + [f"Summary: {k}" for k in summaries.keys()]
                pick = st.selectbox("Choose an output to use as reference", options, index=None, placeholder="Select…")
                if pick:
                    if pick.startswith("Answer: "):
                        name = pick.replace("Answer: ", "")
                        ref_text = answers.get(name, "")
                    elif pick.startswith("Summary: "):
                        name = pick.replace("Summary: ", "")
                        ref_text = summaries.get(name, "")
                    st.session_state["reference_answer"] = ref_text
            if ref_text and ref_text.strip():
                try:
                    candidates = {**{f"Answer: {k}": v for k, v in answers.items()}, **{f"Summary: {k}": v for k, v in summaries.items()}}
                    eval_rows = evaluate_text(ref_text, candidates, lang=st.session_state.get("selected_language", "en"))
                    if eval_rows:
                        eval_df = pd.DataFrame(eval_rows)
                        # Order columns
                        ordered_cols = ["Name", "ROUGE-1", "ROUGE-2", "ROUGE-Lsum", "BLEU", "BERTScore_P", "BERTScore_R", "BERTScore_F1"]
                        eval_df = eval_df[[c for c in ordered_cols if c in eval_df.columns]]
                        st.dataframe(eval_df.sort_values("BERTScore_F1", ascending=False), use_container_width=True)
                except Exception as e:
                    st.warning(f"Metrics evaluation failed: {e}")

        # Response length comparison
        if len(answers) > 1:
            st.divider()
            st.subheader("📏 Response Length Analysis")
            
            lengths = [len(text) for text in answers.values()]
            model_names = list(answers.keys())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=model_names,
                    y=lengths,
                    marker=dict(
                        color=lengths,
                        colorscale='viridis',
                        showscale=True,
                        colorbar=dict(title="Character Count")
                    ),
                    text=[f"{l:,}" for l in lengths],
                    textposition='auto',
                    textfont=dict(size=20)
                )
            ])
            fig.update_layout(
                title="Response Length by Model",
                xaxis_title="Model",
                yaxis_title="Character Count",
                showlegend=False,
                font=dict(size=20),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No analysis results available. Run analysis on the Home page first.")