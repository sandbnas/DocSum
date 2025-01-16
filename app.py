import streamlit as st
import plotly.express as px
import os
from datetime import datetime
import time
from summarizer import DocumentSummarizer, RecursiveCharacterTextSplitter
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
import concurrent.futures
import chromadb
from chromadb.config import Settings
import atexit

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'app.log',
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger('chromadb').setLevel(logging.DEBUG)

# Initialize ChromaDB
if not os.path.exists("./chroma_db"):
    os.makedirs("./chroma_db")

chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(
        anonymized_telemetry=False
    )
)
def cleanup_chroma():
    try:
        chroma_client.reset()
    except Exception as e:
        logger.error(f"Error cleaning up ChromaDB: {e}")

atexit.register(cleanup_chroma)


# Set page configuration
st.set_page_config(
    page_title="DocSum - Advanced Document Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stProgress .st-bo {
        background-color: #00a0a0;
    }
    .stats-container {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state and temp directory
if 'summarizer' not in st.session_state:
    logger.info("Initializing summarizer...")
    try:
        st.session_state.summarizer = DocumentSummarizer()
        logger.info("Summarizer initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing summarizer: {str(e)}", exc_info=True)
        st.error("Failed to initialize summarizer. Please check the logs.")

if 'history' not in st.session_state:
    logger.debug("Initializing history")
    st.session_state.history = []

# Create temp directory if it doesn't exist
temp_dir = "temp_uploads"
os.makedirs(temp_dir, exist_ok=True)

# Sidebar
with st.sidebar:
    st.title("DocSum Settings")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Model settings
    st.subheader("Model Settings")
    model = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo-16k"], help="Select the model to use for summarization")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1, help="Higher values make the output more creative")
    
    # Advanced settings
    st.subheader("Advanced Settings")
    chunk_size = st.number_input("Chunk Size", 500, 3000, 1500, 100, help="Size of text chunks for processing")
    chunk_overlap = st.number_input("Chunk Overlap", 50, 1000, 300, 50, help="Overlap between text chunks")
    
    # Cache settings
    st.subheader("Cache Settings")
    cache_enabled = st.toggle("Enable Cache", True, help="Enable/disable summary caching")
    if cache_enabled:
        cache_days = st.number_input("Cache Expiry (days)", 1, 30, 7, 1, help="Number of days before cache expires")

# Main content
st.title("📄 DocSum - Advanced Document Summarizer")
st.markdown("Transform your documents into concise, intelligent summaries using state-of-the-art AI.")

# File upload
uploaded_file = st.file_uploader("Upload your document", type=['txt', 'pdf', 'docx'], 
                                help="Supported formats: PDF, DOCX, TXT")

def timeout_handler(func, args=(), timeout=300):
    """Handle function timeout"""
    result = None
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args)
            result = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return {"error": "Operation timed out"}
    except Exception as e:
        return {"error": str(e)}
    return result

def main():
    try:
        if 'summarizer' not in st.session_state:
            st.session_state.summarizer = DocumentSummarizer()

        # Add session timeout handling
        if 'last_activity' not in st.session_state:
            st.session_state.last_activity = time.time()
        
        # Check session timeout (30 minutes)
        if time.time() - st.session_state.last_activity > 1800:
            # Reset session
            st.session_state.clear()
            st.session_state.summarizer = DocumentSummarizer()
        
        st.session_state.last_activity = time.time()
        
        if uploaded_file is not None:
            try:
                logger.info(f"Processing file: {uploaded_file.name}")
                # Save uploaded file temporarily
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                logger.debug(f"File saved to temporary path: {temp_path}")
                
                with st.spinner("Processing document..."):
                    try:
                        # Update summarizer settings
                        logger.debug(f"Updating settings - Model: {model}, Temperature: {temperature}")
                        st.session_state.summarizer.llm.temperature = temperature
                        st.session_state.summarizer.llm.model_name = model
                        
                        # Create new text splitter with updated settings
                        logger.debug(f"Creating text splitter - Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
                        st.session_state.summarizer.text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            length_function=len,
                            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                        )
                        
                        # Generate summary
                        logger.info("Starting document summarization")
                        result = timeout_handler(
                            st.session_state.summarizer.summarize,
                            args=(temp_path,),
                            timeout=600
                        )
                        logger.info("Summarization completed")
                        
                        if 'error' in result and result['error']:
                            logger.error(f"Error in summarization result: {result['error']}")
                            st.error(f"Error processing document: {result['error']}")
                        else:
                            logger.info("Displaying results")
                            # Display summary
                            st.markdown("### Document Summary")
                            st.write(result.get("summary", "No summary available"))
                            
                            # Add to history with proper dict handling
                            metadata = {}
                            if isinstance(result, dict):
                                metadata = result.get('metadata', {})
                                if isinstance(metadata, dict):
                                    history_item = {
                                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        'filename': uploaded_file.name,
                                        'processing_time': float(metadata.get('processing_time') or 0),
                                        'total_tokens': int(metadata.get('total_tokens') or 0),
                                        'total_cost': float(metadata.get('total_cost') or 0),
                                        'chunks_processed': int(metadata.get('chunks_processed') or 0)
                                    }
                                    st.session_state.history.append(history_item)
                                    
                                    # Display metrics with proper type checking and default values
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        proc_time = float(metadata.get('processing_time') or 0)
                                        st.metric("Processing Time", f"{proc_time:.2f}s")
                                    with col2:
                                        tokens = int(metadata.get('total_tokens') or 0)
                                        st.metric("Total Tokens", tokens)
                                    with col3:
                                        cost = float(metadata.get('total_cost') or 0)
                                        st.metric("Cost", f"${cost:.4f}")
                                    with col4:
                                        chunks = int(metadata.get('chunks_processed') or 0)
                                        st.metric("Chunks", chunks)
                
                    except Exception as e:
                        logger.error(f"Error during summarization: {str(e)}", exc_info=True)
                        st.error(f"Error during summarization: {str(e)}")
                        
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    logger.error("Failed to remove temporary file", exc_info=True)
                    
            except Exception as e:
                logger.error(f"Error uploading file: {str(e)}", exc_info=True)
                st.error(f"Error uploading file: {str(e)}")
                
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please refresh the page.")

if __name__ == "__main__":
    main()

# History section
if st.session_state.history:
    st.subheader("Processing History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
    
    # Fix column names for plotting
    col1, col2 = st.columns(2)
    with col1:
        if 'total_tokens' in history_df.columns:
            fig_tokens = px.line(history_df, x='timestamp', y='total_tokens', 
                               title='Token Usage Over Time')
            st.plotly_chart(fig_tokens)
    with col2:
        if 'total_cost' in history_df.columns:
            fig_cost = px.line(history_df, x='timestamp', y='total_cost', 
                             title='Cost Over Time')
            st.plotly_chart(fig_cost)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ using LangChain and Streamlit</p>
</div>
""", unsafe_allow_html=True)