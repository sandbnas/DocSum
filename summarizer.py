import os
import pandas as pd
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any, Union, Mapping, Tuple
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.documents import Document
import logging
from logging.handlers import RotatingFileHandler
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient, Collection
from chromadb.types import Metadata

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = RotatingFileHandler(
        'app.log',
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5,
        encoding='utf-8'
    )
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

# Load environment variables
load_dotenv()

class DocumentSummarizer:
    def __init__(self, cache_dir: str = ".cache"):
        logger.info("Initializing DocumentSummarizer")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create chroma directory
        chroma_dir = os.path.join(cache_dir, "chroma")
        os.makedirs(chroma_dir, exist_ok=True)
        
        # Custom prompts for better summarization
        self.map_prompt_template = """Write a detailed summary of the following text, focusing on the main points and key details:
        "{text}"
        DETAILED SUMMARY:"""
        
        self.combine_prompt_template = """Combine these summaries into a comprehensive, well-structured final summary:
        "{text}"
        FINAL SUMMARY:"""
        
        try:
            # Initialize LLM with better parameters
            logger.debug("Initializing ChatOpenAI")
            self.llm = ChatOpenAI(
                model="gpt-4",  # Use 'model' instead of 'model_name'
                temperature=0.3,
                max_retries=5,  # Increased retries
                timeout=600  # Timeout in seconds
            )
            
            # Adjust text splitter for better performance
            logger.debug("Initializing text splitter")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Reduced chunk size
                chunk_overlap=100,  # Reduced overlap
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # Initialize ChromaDB with correct settings
            self.chroma_client = PersistentClient(
                path=chroma_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection for document embeddings
            self.collection = self.chroma_client.get_or_create_collection(
                name="document_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize embeddings with better parameters
            logger.debug("Initializing OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(
                timeout=600,  # Increased timeout
                max_retries=5  # Increased retries
            )
            
            logger.info("DocumentSummarizer initialized successfully")
        except Exception as e:
            logger.error("Error initializing DocumentSummarizer", exc_info=True)
            raise

    def _get_cache_path(self, file_path: str) -> Path:
        """Generate a unique cache file path based on file content hash"""
        try:
            logger.debug(f"Generating cache path for {file_path}")
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return self.cache_dir / f"{file_hash}.json"
        except Exception as e:
            logger.error("Error generating cache path", exc_info=True)
            raise

    def _get_cached_summary(self, file_path: str) -> Optional[Dict]:
        """Retrieve cached summary if available and not expired"""
        try:
            logger.debug(f"Checking cache for {file_path}")
            cache_path = self._get_cache_path(file_path)
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
                    # Cache expires after 7 days
                    cache_date = datetime.fromisoformat(cache['timestamp'])
                    if (datetime.now() - cache_date).days < 7:
                        logger.info("Found valid cached summary")
                        return cache
            logger.info("No valid cached summary found")
            return None
        except Exception as e:
            logger.error("Error checking cache", exc_info=True)
            raise

    def _save_to_cache(self, cache_path: Path, cache_data: Dict[str, Any]) -> None:
        """Save data to cache file."""
        try:
            logger.debug(f"Saving to cache: {cache_path}")
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.error("Error saving to cache", exc_info=True)
            raise

    def load_document(self, file_path: str):
        """Enhanced document loading with better error handling"""
        try:
            logger.info(f"Loading document: {file_path}")
            _, file_extension = os.path.splitext(file_path)
            
            if file_extension.lower() == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension.lower() == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension.lower() in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            logger.debug("Loading document content")
            documents = loader.load()
            
            # Create metadata
            metadata = {
                'filename': os.path.basename(file_path),
                'file_type': file_extension,
                'file_size': os.path.getsize(file_path),
                'created_at': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                'modified_at': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            }
            
            logger.info("Document loaded successfully")
            return documents, metadata
            
        except Exception as e:
            logger.error("Error loading document", exc_info=True)
            raise Exception(f"Error loading document: {str(e)}")

    def process_document(self, file_path: str) -> List[Document]:
        """Process a document and split it into chunks."""
        try:
            logger.info(f"Processing document: {file_path}")
            file_extension = Path(file_path).suffix.lower()
            
            logger.debug(f"File extension: {file_extension}")
            if file_extension == '.txt':
                loader = TextLoader(file_path)
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.doc', '.docx']:
                loader = Docx2txtLoader(file_path)
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            logger.debug("Loading document")
            document = self.load_document(file_path)
            chunks = self.text_splitter.split_documents(document[0])
            
            # Prepare documents for ChromaDB
            texts = [chunk.page_content for chunk in chunks]
            metadatas: List[Metadata] = [
                {str(k): str(v) for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))} 
                for chunk in chunks
            ]
            ids = [str(i) for i in range(len(chunks))]
            
            # Add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(chunks)} chunks to ChromaDB")
            return chunks
            
        except Exception as e:
            logger.error("Error processing document", exc_info=True)
            raise Exception(f"Error loading document: {str(e)}")

    def summarize(self, file_path: str) -> Dict[str, Any]:
        """Summarize a document with improved error handling and progress tracking."""
        try:
            logger.info(f"Starting summarization for: {file_path}")
            
            # Generate cache key
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            cache_path = self.cache_dir / f"{file_hash}.json"

            # Check cache
            if cache_path.exists():
                logger.info("Found cached summary")
                with open(cache_path, 'r') as f:
                    return json.load(f)

            start_time = datetime.now()
            
            # Load and process document with progress tracking
            logger.debug("Processing document")
            texts = self.process_document(file_path)
            
            if not texts:
                logger.error("No text content found in document")
                raise ValueError("No text content found in document")

            # Add chunk size validation
            if len(texts) > 50:  # Limit maximum chunks
                logger.warning("Document too large, truncating to first 50 chunks")
                texts = texts[:50]
            
            # Add progress tracking
            total_chunks = len(texts)
            processed_chunks = 0
            
            # Process chunks with better error handling
            summaries = []
            for chunk in texts:
                try:
                    with get_openai_callback() as cb:
                        result = self.llm.invoke(chunk.page_content)
                        summaries.append(result)
                    processed_chunks += 1
                    logger.info(f"Processed chunk {processed_chunks}/{total_chunks}")
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}", exc_info=True)
                    continue  # Skip failed chunks instead of crashing
            
            # Only proceed if we have some successful summaries
            if not summaries:
                raise ValueError("Failed to generate any summaries")
                
            # Combine summaries with error handling
            try:
                final_summary = self._combine_summaries(summaries)
            except Exception as e:
                logger.error(f"Error combining summaries: {str(e)}", exc_info=True)
                final_summary = "Error: Failed to combine summaries. " + str(e)

            # Create vector store for semantic search
            logger.debug("Creating vector store")
            vectorstore = Chroma.from_documents(documents=list(texts), embedding=self.embeddings)

            # Create map-reduce chain for summarization
            logger.debug("Creating summarization chains")
            map_prompt = PromptTemplate(
                template=self.map_prompt_template,
                input_variables=["text"]
            )
            combine_prompt = PromptTemplate(
                template=self.combine_prompt_template,
                input_variables=["text"]
            )
            
            map_chain = LLMChain(llm=self.llm, prompt=map_prompt)
            reduce_chain = LLMChain(llm=self.llm, prompt=combine_prompt)
            
            # Initialize chains with better error handling
            combine_documents_chain = StuffDocumentsChain(
                llm_chain=reduce_chain, 
                document_variable_name="text"
            )
            
            reduce_documents_chain = MapReduceDocumentsChain(
                llm_chain=map_chain,
                reduce_documents_chain=combine_documents_chain,
                document_variable_name="text",
                return_intermediate_steps=False  # Changed to False to reduce memory usage
            )

            # Process with OpenAI callback for token tracking
            logger.info("Starting summarization process")
            with get_openai_callback() as cb:
                try:
                    logger.debug("Invoking reduce_documents_chain")
                    result = reduce_documents_chain.invoke({"input_documents": texts})
                    logger.info("Summarization completed successfully")
                    logger.debug(f"Result type: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
                except Exception as e:
                    logger.error("Error during summarization", exc_info=True)
                    raise Exception(f"Error during summarization: {str(e)}")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Prepare result with metadata
            summary_data = {
                "summary": result["output_text"],
                "metadata": {
                    "filename": Path(file_path).name,
                    "file_size": Path(file_path).stat().st_size,
                    "processing_time": duration,
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost,
                    "chunks_processed": len(texts)
                }
            }

            # Cache the result
            logger.debug("Saving results to cache")
            self._save_to_cache(cache_path, summary_data)
            logger.info("Summarization process completed")
            return summary_data

        except Exception as e:
            logger.error("Error in summarization process", exc_info=True)
            error_msg = f"Error in summarization process: {str(e)}"
            return {"error": error_msg, "summary": None, "metadata": None}

    def _combine_summaries(self, summaries: List[Any]) -> str:
        """Combine multiple summaries into a single coherent text."""
        try:
            # Convert AIMessage objects to strings if necessary
            processed_summaries = []
            for summary in summaries:
                if hasattr(summary, 'content'):
                    processed_summaries.append(str(summary.content))
                else:
                    processed_summaries.append(str(summary))
            
            combined_text = " ".join(processed_summaries)
            result = self.llm.invoke(f"Combine these summaries into a coherent text: {combined_text}")
            
            # Handle different return types
            if hasattr(result, 'content'):
                return str(result.content)
            elif isinstance(result, str):
                return result
            elif isinstance(result, list):
                return " ".join(str(item) for item in result)
            elif isinstance(result, dict) and 'content' in result:
                return str(result['content'])
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Error combining summaries: {str(e)}", exc_info=True)
            raise

    # Add cleanup method
    def cleanup(self):
        try:
            # Clean up any resources
            if hasattr(self, 'llm'):
                del self.llm
            if hasattr(self, 'embeddings'):
                del self.embeddings
            # Clean up ChromaDB resources
            if hasattr(self, 'collection'):
                self.collection.delete()
            if hasattr(self, 'chroma_client'):
                self.chroma_client.reset()
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}", exc_info=True)

    # Add method for semantic search
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using ChromaDB."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            documents = results.get('documents')
            metadatas = results.get('metadatas')
            
            if not documents or not metadatas or not len(documents) or not len(metadatas):
                return []
                
            return [
                {"document": doc, "metadata": meta} 
                for doc, meta in zip(documents[0], metadatas[0]) 
                if doc is not None and meta is not None
            ]
        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            return []

def main():
    logger.info("Starting main function")
    summarizer = DocumentSummarizer()
    
    print("Advanced Document Summarization Tool")
    print("=" * 50)
    print("\nFeatures:")
    print("- Intelligent caching")
    print("- Advanced text processing")
    print("- Detailed analytics")
    print("- GPT-4 powered summarization")
    print("\nSupported formats: PDF, DOCX, TXT")
    
    while True:
        file_path = input("\nEnter document path (or 'quit' to exit): ")
        
        if file_path.lower() == 'quit':
            break
        
        if not os.path.exists(file_path):
            print("Error: File not found. Please check the path and try again.")
            continue
        
        print("\nProcessing document...")
        result = summarizer.summarize(file_path)
        
        if 'error' in result:
            print(f"\nError: {result['error']}")
            continue
        
        print("\nSummary:")
        print("=" * 50)
        print(result['summary'])
        print("\nDocument Statistics:")
        print("=" * 50)
        for key, value in result['metadata'].items():
            print(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()