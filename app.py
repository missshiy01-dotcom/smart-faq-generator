"""
Smart FAQ Generator - Latest Gemini API (2024)
Using the NEW google-genai SDK and Gemini 2.5 Flash

Installation:
pip install google-genai streamlit PyPDF2
"""

import streamlit as st
import PyPDF2
import json
import re
from datetime import datetime
import time
from typing import List, Dict
from google import genai

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart FAQ Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .free-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 20px;
        border-radius: 10px;
        color: #155724;
        font-weight: 500;
        margin: 20px 0;
    }
    .faq-card {
        background: white;
        padding: 25px;
        border-left: 5px solid #667eea;
        margin: 20px 0;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .faq-card:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transform: translateY(-3px);
    }
    .progress-text {
        font-size: 1.1rem;
        color: #667eea;
        font-weight: 600;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GEMINI API FUNCTIONS (NEW SDK)
# ============================================================

@st.cache_resource
def initialize_gemini_client(api_key: str):
    """Initialize Gemini client with the NEW SDK"""
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini: {e}")
        return None

def test_gemini_connection(client, model_name: str = "gemini-2.0-flash-exp") -> bool:
    """Test if Gemini API is working"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Hello"
        )
        return response.text is not None
    except Exception as e:
        st.error(f"❌ Connection test failed: {str(e)[:200]}")
        return False

def generate_with_gemini(client, prompt: str, model_name: str = "gemini-2.0-flash-exp") -> str:
    """
    Generate content using the NEW Gemini API
    """
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "temperature": 0.7,
                "max_output_tokens": 2500,
                "top_p": 0.95,
            }
        )
        
        if response.text:
            return response.text
        else:
            return ""
            
    except Exception as e:
        st.error(f"❌ Gemini API Error: {str(e)[:200]}")
        return ""

# ============================================================
# TEXT EXTRACTION FUNCTIONS
# ============================================================

def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        total_pages = len(pdf_reader.pages)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
            
            progress_bar.progress((i + 1) / total_pages)
            status_text.markdown(f"<p class='progress-text'>📖 Reading page {i+1}/{total_pages}...</p>", 
                               unsafe_allow_html=True)
        
        progress_bar.empty()
        status_text.empty()
        
        return text.strip()
        
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""

def extract_text_from_txt(file) -> str:
    """Extract text from TXT/MD file"""
    try:
        text = file.read().decode('utf-8', errors='ignore')
        return text.strip()
    except Exception as e:
        st.error(f"❌ Error reading text file: {e}")
        return ""

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ============================================================
# TEXT CHUNKING FUNCTIONS
# ============================================================

def smart_chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into chunks with smart boundary detection"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_length = len(sentence_words)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            overlap_sentences = []
            overlap_length = 0
            
            for s in reversed(current_chunk):
                s_length = len(s.split())
                if overlap_length + s_length > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_length += s_length
            
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# ============================================================
# FAQ GENERATION FUNCTIONS
# ============================================================

def generate_faqs_from_chunk(
    client,
    chunk: str,
    chunk_num: int,
    total_chunks: int,
    num_questions: int = 5,
    model_name: str = "gemini-2.0-flash-exp"
) -> List[Dict[str, str]]:
    """Generate FAQ pairs from a text chunk using Gemini"""
    
    prompt = f"""You are an expert at creating educational FAQ content.

Generate exactly {num_questions} high-quality question-answer pairs from the following text (chunk {chunk_num}/{total_chunks}).

TEXT:
\"\"\"
{chunk}
\"\"\"

REQUIREMENTS:
1. Generate EXACTLY {num_questions} question-answer pairs
2. Questions should be natural and specific to the content
3. Answers must be complete and informative (2-4 sentences)
4. Cover different aspects of the text
5. Base answers ONLY on the provided text

OUTPUT: Return ONLY a valid JSON array with this structure:
[
  {{
    "question": "What is the main topic?",
    "answer": "The main topic is... [complete answer]"
  }}
]

CRITICAL: Return ONLY the JSON array, no markdown, no extra text."""

    try:
        response_text = generate_with_gemini(client, prompt, model_name)
        
        if not response_text:
            return []
        
        # Clean response
        response_text = response_text.strip()
        
        # Remove markdown code blocks
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]
        
        response_text = response_text.strip()
        
        # Parse JSON
        faqs = json.loads(response_text)
        
        # Validate structure
        if isinstance(faqs, list):
            valid_faqs = []
            for faq in faqs:
                if isinstance(faq, dict) and 'question' in faq and 'answer' in faq:
                    if faq['question'].strip() and faq['answer'].strip():
                        valid_faqs.append({
                            'question': faq['question'].strip(),
                            'answer': faq['answer'].strip()
                        })
            return valid_faqs
        
        return []
        
    except json.JSONDecodeError as e:
        st.warning(f"⚠️ Chunk {chunk_num}: JSON parsing error")
        return []
    except Exception as e:
        st.error(f"❌ Chunk {chunk_num}: {str(e)[:100]}")
        return []

def deduplicate_faqs(faqs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicate FAQs"""
    seen_questions = set()
    unique_faqs = []
    
    for faq in faqs:
        q_normalized = faq['question'].lower().strip().rstrip('?').rstrip('.')
        if q_normalized not in seen_questions:
            seen_questions.add(q_normalized)
            unique_faqs.append(faq)
    
    return unique_faqs

# ============================================================
# EXPORT FUNCTIONS
# ============================================================

def export_to_json(faqs: List[Dict[str, str]], filename: str) -> str:
    """Export FAQs to JSON"""
    data = {
        "document": filename,
        "generated_at": datetime.now().isoformat(),
        "total_faqs": len(faqs),
        "model": "Google Gemini 2.0 Flash",
        "faqs": faqs
    }
    return json.dumps(data, indent=2, ensure_ascii=False)

def export_to_markdown(faqs: List[Dict[str, str]], filename: str) -> str:
    """Export FAQs to Markdown"""
    md = f"# 📚 FAQs - {filename}\n\n"
    md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
    md += f"**Total Questions:** {len(faqs)}  \n\n---\n\n"
    
    for i, faq in enumerate(faqs, 1):
        md += f"## {i}. {faq['question']}\n\n"
        md += f"**Answer:** {faq['answer']}\n\n---\n\n"
    
    return md

def export_to_html(faqs: List[Dict[str, str]], filename: str) -> str:
    """Export FAQs to HTML"""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FAQs - {filename}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 50px;
        }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        .faq {{
            margin: 25px 0;
            padding: 25px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            border-left: 6px solid #667eea;
            transition: transform 0.3s;
        }}
        .faq:hover {{ transform: translateY(-5px); box-shadow: 0 8px 20px rgba(0,0,0,0.1); }}
        .question {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 12px;
        }}
        .answer {{ color: #34495e; line-height: 1.8; font-size: 1.05em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 {filename}</h1>
        <p style="text-align:center;color:#888;margin-bottom:30px;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Total: {len(faqs)} FAQs
        </p>
"""
    
    for i, faq in enumerate(faqs, 1):
        html += f"""
        <div class="faq">
            <div class="question">Q{i}: {faq['question']}</div>
            <div class="answer">{faq['answer']}</div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    return html

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    st.markdown('<p class="main-header">📚 Smart FAQ Generator</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <span class="free-badge">✨ 100% FREE ✨</span>
        <p style='color: #666; margin-top: 10px;'>Powered by Google Gemini 2.0 Flash</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown("### 🔑 Gemini API Key")
        st.markdown("""
        **Get FREE API key:**
        1. Visit [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
        2. Click "Create API Key"
        3. Copy & paste below
        """)
        
        api_key = st.text_input("Enter API Key", type="password")
        
        # Model selection
        model_name = st.selectbox(
            "🤖 Model",
            [
                "gemini-2.5-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro"
            ],
            help="Official Gemini models"
        )
        
        if api_key:
            client = initialize_gemini_client(api_key)
            if client:
                with st.spinner("Testing connection..."):
                    if test_gemini_connection(client, model_name):
                        st.success(f"✅ {model_name} ready!")
                    else:
                        st.error("❌ Connection failed. Try another model.")
                        client = None
            else:
                client = None
        else:
            st.warning("⚠️ Enter API key")
            client = None
        
        st.markdown("---")
        st.subheader("📊 Settings")
        
        chunk_size = st.slider("Chunk Size", 500, 3000, 2000, 100)
        overlap = st.slider("Overlap", 0, 500, 200, 50)
        questions_per_chunk = st.slider("Questions/Chunk", 3, 8, 5)
        
        if 'faqs' in st.session_state:
            st.markdown("---")
            st.metric("Total FAQs", len(st.session_state.faqs))
            if st.button("🗑️ Clear"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
    
    # Main area
    st.markdown("## 📤 Upload Document")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'md']
        )
    
    with col2:
        if uploaded_file:
            st.metric("📄 File", uploaded_file.name[:20])
            st.metric("📦 Size", f"{uploaded_file.size/1024:.1f} KB")
    
    # Generate button
    if uploaded_file and client:
        st.markdown("---")
        
        if st.button("🚀 Generate FAQs", type="primary", use_container_width=True):
            
            st.markdown("## 🔄 Processing...")
            
            # Extract text
            with st.spinner("Extracting text..."):
                if uploaded_file.name.endswith('.pdf'):
                    text = extract_text_from_pdf(uploaded_file)
                else:
                    text = extract_text_from_txt(uploaded_file)
                
                if text:
                    text = clean_text(text)
            
            if not text:
                st.error("❌ No text extracted")
                return
            
            st.markdown(f"""
            <div class="success-box">
                ✅ Extracted {len(text):,} characters / {len(text.split()):,} words
            </div>
            """, unsafe_allow_html=True)
            
            # Chunk text
            with st.spinner("Chunking..."):
                chunks = smart_chunk_text(text, chunk_size, overlap)
            
            st.markdown(f"""
            <div class="success-box">
                ✅ Created {len(chunks)} chunks
            </div>
            """, unsafe_allow_html=True)
            
            # Generate FAQs
            st.markdown("### 🤖 Generating FAQs...")
            
            all_faqs = []
            progress_bar = st.progress(0)
            status = st.empty()
            
            for i, chunk in enumerate(chunks):
                status.markdown(
                    f"<p class='progress-text'>Processing chunk {i+1}/{len(chunks)}...</p>",
                    unsafe_allow_html=True
                )
                
                faqs = generate_faqs_from_chunk(
                    client, chunk, i+1, len(chunks),
                    questions_per_chunk, model_name
                )
                
                if faqs:
                    all_faqs.extend(faqs)
                    st.toast(f"✅ Chunk {i+1}: {len(faqs)} FAQs", icon="✅")
                
                progress_bar.progress((i + 1) / len(chunks))
                
                if i < len(chunks) - 1:
                    time.sleep(1)
            
            status.empty()
            progress_bar.empty()
            
            if all_faqs:
                all_faqs = deduplicate_faqs(all_faqs)
                
                st.markdown(f"""
                <div class="success-box">
                    🎉 Generated {len(all_faqs)} unique FAQs!
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.faqs = all_faqs
                st.session_state.doc_name = uploaded_file.name
                st.balloons()
            else:
                st.error("❌ No FAQs generated")
    
    # Display results
    if 'faqs' in st.session_state and st.session_state.faqs:
        st.markdown("---")
        st.header(f"📋 Results ({len(st.session_state.faqs)} FAQs)")
        
        # Export buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📄 JSON",
                export_to_json(st.session_state.faqs, st.session_state.doc_name),
                f"{st.session_state.doc_name}_faqs.json",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                "📝 Markdown",
                export_to_markdown(st.session_state.faqs, st.session_state.doc_name),
                f"{st.session_state.doc_name}_faqs.md",
                use_container_width=True
            )
        
        with col3:
            st.download_button(
                "🌐 HTML",
                export_to_html(st.session_state.faqs, st.session_state.doc_name),
                f"{st.session_state.doc_name}_faqs.html",
                use_container_width=True
            )
        
        # Preview
        st.markdown("---")
        st.subheader("📖 Preview")
        
        for i, faq in enumerate(st.session_state.faqs, 1):
            with st.expander(f"**Q{i}: {faq['question']}**"):
                st.markdown(f"**Answer:** {faq['answer']}")
    
    elif not api_key:
        st.info("👈 Enter your Gemini API key to start!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p><strong>Smart FAQ Generator</strong> | Course Project 1.2</p>
        <p>Powered by Google Gemini 2.0 • Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()