# Complete Fixed app.py
import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from apis.gemini_api import query_gemini
from apis.hf_api import query_hf
from apis.groq_api import query_groq
from apis.perplexity_api import query_perplexity
from summarizers.extractive import summarize_lexrank, summarize_luhn, summarize_lsa
from summarizers.abstractive import AbstractiveSummarizer, available_abstractive_models
from summarizers.llm_summary import summarize_with_gemini
from utils.compare import (
    extract_shared_ngrams,
    similarity_matrix,
    calculate_all_similarities,
    calculate_question_similarities,
    get_similarity_stats
)

# Page configuration
st.set_page_config(
    page_title="Question-Answering and Summarization System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize abstractive summarizer with caching
@st.cache_resource
def load_abstractive_summarizer(model_key: str, prefer_offline: bool = False):
    """Cache abstractive summarizers per model to avoid redundant loads."""
    return AbstractiveSummarizer(model_key=model_key, prefer_offline=prefer_offline)

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

    LANGUAGE_OPTIONS = {"English": "en", "Hindi": "hi"}
    env_default = os.environ.get("APP_DEFAULT_LANG")
    default_index = 0
    if env_default == "hi":
        default_index = 1
    language_label = st.selectbox("Target summary language", list(LANGUAGE_OPTIONS.keys()), index=default_index)
    summary_language = LANGUAGE_OPTIONS[language_label]
    st.session_state["selected_language"] = summary_language


    summarizer_choices = st.multiselect(
        "Extractive & LLM techniques",
        ["lexrank", "luhn", "lsa", "llm_gemini"],
        default=["lexrank", "llm_gemini"],
        format_func=lambda x: {
            "lexrank": "📊 LexRank (Graph-based)",
            "luhn": "📈 Luhn (Frequency-based)", 
            "lsa": "🔍 LSA (Latent Semantic)",
            "llm_gemini": "✨ LLM Summary (Gemini)"
        }[x]
    )

    num_sentences = st.slider("Sentences for extractive methods", 1, 5, 2)

    abstractive_catalog = available_abstractive_models()
    selected_abstractive_models = st.multiselect(
        "Abstractive models (multilingual)",
        list(abstractive_catalog.keys()),
        default=["mt5-small"],
        format_func=lambda key: abstractive_catalog[key]
    )

# Main Input Area
st.subheader("❓ Your Question")
question = st.text_area(
    "Enter your question here...",
    height=125,
    placeholder="What is machine learning? How does it differ from traditional programming?"
)

# Inform about abstractive model selection
if selected_abstractive_models:
    chosen_labels = [abstractive_catalog.get(key, key) for key in selected_abstractive_models]
    st.info(
        "🧠 Abstractive models selected: " + ", ".join(chosen_labels)
    )

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
        st.session_state.results['selected_abstractive_models'] = selected_abstractive_models
        st.session_state.results['summary_language'] = summary_language
        
        summaries = {}

        # Generate summaries
        if summarizer_choices or selected_abstractive_models:
            with st.spinner("📝 Generating summaries..."):
                # Build two versions: verbose (for extractive) and cleaned (for abstractive)
                combined_text = "\n\n".join([f"{name}: {txt}" for name, txt in answers.items()])
                # Clean for abstractive: drop model name prefixes, trim each answer in fast mode
                cleaned_parts = []
                per_answer_limit = 1200
                for name, txt in answers.items():
                    part = str(txt)
                    # remove name prefixes and stray special markers
                    part = part.replace(f"{name}:", "").replace("< s >", " ")
                    part = part.replace("<s>", " ").replace("</s>", " ")
                    part = part.replace("<extra_id_0>", " ")
                    if len(part) > per_answer_limit:
                        part = part[:per_answer_limit]
                    cleaned_parts.append(part.strip())
                abstractive_input = "\n\n".join(cleaned_parts).strip()
                language_labels = {"en": "English", "hi": "Hindi"}
                lang_display = language_labels.get(summary_language, summary_language)

                if "lexrank" in summarizer_choices:
                    try:
                        summaries[f"LexRank ({lang_display})"] = summarize_lexrank(
                            combined_text,
                            num_sentences,
                            language=summary_language
                        )
                    except Exception as e:
                        summaries[f"LexRank ({lang_display})"] = f"[Error: {str(e)}]"
                        
                if "luhn" in summarizer_choices:
                    try:
                        summaries[f"Luhn ({lang_display})"] = summarize_luhn(
                            combined_text,
                            num_sentences,
                            language=summary_language
                        )
                    except Exception as e:
                        summaries[f"Luhn ({lang_display})"] = f"[Error: {str(e)}]"
                        
                if "lsa" in summarizer_choices:
                    try:
                        summaries[f"LSA ({lang_display})"] = summarize_lsa(
                            combined_text,
                            num_sentences,
                            language=summary_language
                        )
                    except Exception as e:
                        summaries[f"LSA ({lang_display})"] = f"[Error: {str(e)}]"

                if selected_abstractive_models:
                    for model_key in selected_abstractive_models:
                        display_name = abstractive_catalog.get(model_key, model_key)
                        try:
                            summarizer = load_abstractive_summarizer(model_key)
                            # Use cleaned input for abstractive models for better quality/latency
                            summary_text = summarizer.summarize(
                                abstractive_input,
                                language=summary_language
                            )
                        except Exception as e:
                            summary_text = f"[Error: {str(e)}]"
                        summaries[f"{display_name} ({lang_display})"] = summary_text
                    
                if "llm_gemini" in summarizer_choices:
                    try:
                        summaries[f"LLM Summary (Gemini, {lang_display})"] = summarize_with_gemini(
                            combined_text,
                            lang_hint=summary_language
                        )
                    except Exception as e:
                        summaries[f"LLM Summary (Gemini, {lang_display})"] = f"[Error: {str(e)}]"

        st.session_state.results['summaries'] = summaries

        # Calculate question-to-text similarities for answers and summaries
        with st.spinner("🔍 Calculating question-to-text similarities..."):
            all_texts = []
            all_labels = []

            for name, text in answers.items():
                all_texts.append(text)
                all_labels.append(f"Answer: {name}")

            for method, text in summaries.items():
                all_texts.append(text)
                all_labels.append(f"Summary: {method}")

            if all_texts:
                question_sims = calculate_question_similarities(question, all_texts, all_labels)
                st.session_state.results['question_similarities'] = question_sims
                st.session_state.results['text_labels'] = all_labels
            else:
                st.session_state.results['question_similarities'] = {}
                st.session_state.results['text_labels'] = []
        
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
        summary_language = st.session_state.results.get('summary_language')
        language_labels = {"en": "English", "hi": "Hindi"}
        if summary_language:
            st.caption(f"Summaries generated in {language_labels.get(summary_language, summary_language)}")
        selected_abstractive = st.session_state.results.get('selected_abstractive_models', [])
        if selected_abstractive:
            abstractive_catalog = available_abstractive_models()
            names = ", ".join(abstractive_catalog.get(key, key) for key in selected_abstractive)
            st.caption(f"Abstractive models: {names}")
        
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

        st.subheader("🎯 Comprehensive Comparison Table")

        summaries = st.session_state.results.get('summaries', {})
        question_sims = st.session_state.results.get('question_similarities')
        text_labels = st.session_state.results.get('text_labels', [])

        if question_sims and text_labels:
            rows = []
            text_idx = 0

            def safe_score(metric: str, idx: int) -> float:
                values = question_sims.get(metric) or []
                if idx < len(values):
                    return round(float(values[idx]), 3)
                return 0.0

            for name, text in answers.items():
                metric_scores = [
                    safe_score("TF-IDF", text_idx),
                    safe_score("Soft Cosine", text_idx),
                    safe_score("Sentence-BERT", text_idx),
                ]
                rows.append({
                    'Type': 'Answer',
                    'Model/Method': name,
                    'Length (chars)': len(text),
                    'Words': len(text.split()),
                    'TF-IDF Sim': metric_scores[0],
                    'Soft Cosine Sim': metric_scores[1],
                    'SBERT Sim': metric_scores[2],
                    'Avg Similarity': round(np.mean(metric_scores), 3),
                })
                text_idx += 1

            for method, text in summaries.items():
                metric_scores = [
                    safe_score("TF-IDF", text_idx),
                    safe_score("Soft Cosine", text_idx),
                    safe_score("Sentence-BERT", text_idx),
                ]
                rows.append({
                    'Type': 'Summary',
                    'Model/Method': method,
                    'Length (chars)': len(text),
                    'Words': len(text.split()),
                    'TF-IDF Sim': metric_scores[0],
                    'Soft Cosine Sim': metric_scores[1],
                    'SBERT Sim': metric_scores[2],
                    'Avg Similarity': round(np.mean(metric_scores), 3),
                })
                text_idx += 1

            comparison_df = pd.DataFrame(rows)
            comparison_df = comparison_df.sort_values('Avg Similarity', ascending=False)
            st.dataframe(comparison_df, use_container_width=True)

            st.subheader("🏆 Best Performers")
            best_cols = st.columns(2)

            with best_cols[0]:
                best_answer_df = comparison_df[comparison_df['Type'] == 'Answer']
                if not best_answer_df.empty:
                    top_answer = best_answer_df.iloc[0]
                    st.metric(
                        "🥇 Best Answer",
                        top_answer['Model/Method'],
                        f"Similarity: {top_answer['Avg Similarity']}"
                    )

            with best_cols[1]:
                best_summary_df = comparison_df[comparison_df['Type'] == 'Summary']
                if not best_summary_df.empty:
                    top_summary = best_summary_df.iloc[0]
                    st.metric(
                        "🥇 Best Summary",
                        top_summary['Model/Method'],
                        f"Similarity: {top_summary['Avg Similarity']}"
                    )

            summary_language = st.session_state.results.get('summary_language')
            if summary_language:
                language_labels = {"en": "English", "hi": "Hindi"}
                st.caption(f"Summaries evaluated in {language_labels.get(summary_language, summary_language)}")
        else:
            st.info("Question-to-text similarity scores will appear here after running an analysis.")

        st.divider()

        # Response length comparison
        if len(answers) > 1:
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

        st.subheader("🎯 Model Performance Summary")

        performance_data = []
        for name, text in answers.items():
            performance_data.append({
                'Model': name,
                'Length (chars)': len(text),
                'Sentences (approx)': len(text.split('.')),
                'Words (approx)': len(text.split()),
                'Has Error': '[ERROR' in text or '[Error' in text
            })

        st.dataframe(pd.DataFrame(performance_data), use_container_width=True)

        if summaries:
            st.divider()
            st.subheader("📝 Summarization Overview")

            summary_rows = []
            for method, summary in summaries.items():
                summary_rows.append({
                    'Method': method,
                    'Length (chars)': len(summary),
                    'Words (approx)': len(summary.split()),
                    'Has Error': '[ERROR' in summary or '[Error' in summary
                })

            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True)
    else:
        st.warning("No analysis results available. Run analysis on the Home page first.")