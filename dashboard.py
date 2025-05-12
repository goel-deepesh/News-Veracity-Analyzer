
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import time

# Page configuration
st.set_page_config(
    page_title="News Veracity Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load model
@st.cache_resource
def load_model():
    # Your Hugging Face model repository
    model_name = "deepesh-goel/news-veracity-model" 
    
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        
    return tokenizer, model

# Calculate trust score (0-100)
def calculate_trust_score(fake_prob):
    return int((1 - fake_prob) * 100)

# Load the model (will be cached)
with st.spinner("Loading model... (this may take a minute on first run)"):
    tokenizer, model = load_model()

# Rest of your code remains the same...

# Sidebar for information
st.sidebar.title("About")
st.sidebar.write("""
This tool analyzes news articles and provides a trust score based on the content.
The system uses a machine learning model trained on thousands of real and fake news articles.
""")

st.sidebar.markdown("---")
st.sidebar.write("""
**How the Trust Score Works:**

The trust score ranges from 0-100:
- 80-100: Highly Trustworthy
- 60-80: Likely Trustworthy
- 40-60: Uncertain
- 20-40: Potentially Misleading
- 0-20: Highly Suspicious
""")

st.sidebar.markdown("---")
st.sidebar.write("""
**Note:** This tool is for research purposes only. Always verify information from multiple reliable sources.
""")

# Main content
st.title("News Veracity Analyzer")
st.write("Enter a news headline and content to analyze its trustworthiness.")

# User input
title = st.text_input("News Headline:", "")
content = st.text_area("News Content:", "", height=200)

# Analysis controls
col1, col2 = st.columns([1, 3])
with col1:
    analyze_button = st.button("Analyze", type="primary")
with col2:
    example_button = st.button("Load Example")

# Load example if requested
if example_button:
    title = "Scientists Discover Revolutionary Cancer Treatment"
    content = """In a breakthrough announcement yesterday, researchers at a leading medical institute revealed they have developed a new treatment that shows remarkable success in eliminating advanced-stage cancer cells without damaging healthy tissue. Clinical trials showed an unprecedented 95% success rate in terminal patients. Medical experts around the world are calling it the most significant advancement in cancer research in decades. The treatment combines targeted gene therapy with immune system modulation, effectively teaching the body to recognize and destroy cancer cells. Regulatory approval is expected within months."""
    st.session_state.example_loaded = True

# Analyze content
if analyze_button or (example_button and title and content):
    if title and content:
        with st.spinner("Analyzing content..."):
            # Add slight delay for UX
            time.sleep(1)
            
            # Combine inputs with separator
            combined_text = f"{title} [SEP] {content}"
            
            # Tokenize
            inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            fake_prob = probabilities[0, 1].item()
            real_prob = probabilities[0, 0].item()
            
            # Calculate trust score
            trust_score = calculate_trust_score(fake_prob)
            
            # Determine category and color
            if trust_score >= 80:
                category = "Highly Trustworthy"
                color = "#0f7921"  # Green
            elif trust_score >= 60:
                category = "Likely Trustworthy"
                color = "#7cb342"  # Light green
            elif trust_score >= 40:
                category = "Uncertain"
                color = "#ff9800"  # Amber
            elif trust_score >= 20:
                category = "Potentially Misleading"
                color = "#f57c00"  # Orange
            else:
                category = "Highly Suspicious"
                color = "#d32f2f"  # Red
                
            # Display results
            st.header("Analysis Results")
            
            # Split into columns
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Trust score display
                st.subheader("Trust Score")
                
                # Create a custom gauge chart
                fig, ax = plt.subplots(figsize=(4, 0.5))
                
                # Draw gauge background
                ax.barh([""], [100], color="#e0e0e0", height=0.3)
                
                # Draw trust score
                ax.barh([""], [trust_score], color=color, height=0.3)
                
                # Add trust score text
                ax.text(trust_score + 2, 0, f"{trust_score}", 
                        va='center', ha='left', fontweight='bold')
                
                # Set limits and remove axes
                ax.set_xlim(0, 105)
                ax.set_ylim(-0.5, 0.5)
                ax.axis('off')
                
                # Display gauge
                st.pyplot(fig)
                
                # Display category
                st.markdown(f"<h3 style='color:{color}'>{category}</h3>", unsafe_allow_html=True)
                
                # Display probabilities
                st.write(f"Real probability: {real_prob:.2f}")
                st.write(f"Fake probability: {fake_prob:.2f}")
            
            with col2:
                # Analysis details
                st.subheader("Interpretation")
                
                if trust_score >= 80:
                    st.write("""
                    This content appears to be highly trustworthy based on our analysis. 
                    It exhibits patterns consistent with legitimate news articles from reputable sources.
                    """)
                    
                elif trust_score >= 60:
                    st.write("""
                    This content generally appears trustworthy, though it may contain some elements 
                    that differ from typical legitimate news. Consider verifying key claims.
                    """)
                    
                elif trust_score >= 40:
                    st.write("""
                    Our analysis is inconclusive. This content contains a mix of patterns associated 
                    with both legitimate and potentially misleading news. Verify from trusted sources.
                    """)
                    
                elif trust_score >= 20:
                    st.write("""
                    This content contains patterns often associated with misleading information.
                    Exercise caution and verify key claims from multiple reliable sources.
                    """)
                    
                else:
                    st.write("""
                    This content strongly exhibits patterns associated with misinformation.
                    The claims made should be treated with significant skepticism and
                    thoroughly verified through trusted sources.
                    """)
                
                st.write("---")
                st.write("""
                **Remember**: This tool provides an automated assessment based on patterns
                learned from many examples. It cannot guarantee accuracy. Always think critically
                and verify information from multiple trusted sources.
                """)
        
    else:
        st.error("Please enter both a headline and content for analysis.")

# Footer
st.markdown("---")
st.caption("© 2025 News Veracity Analyzer - For research and educational purposes only")
