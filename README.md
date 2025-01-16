# 📄 DocSum: Advanced Document Summarizer

> Transform your documents into intelligent summaries using state-of-the-art AI technology

![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Overview

DocSum is a powerful document summarization tool that uses artificial intelligence to create concise, intelligent summaries of your documents. Whether you have long reports, academic papers, or technical documents, DocSum can help you extract the key information quickly and efficiently.

## ✨ Key Features

### 📚 Document Support
- **Multiple Formats**: Support for PDF, Word (DOCX), and Text files
- **Smart Processing**: Intelligent chunking of large documents
- **Unicode Support**: Handles documents in various languages and encodings

### 🤖 AI-Powered Summarization
- **Advanced AI**: Uses GPT-4 for high-quality summaries
- **Customizable**: Adjust summary length and style
- **Context-Aware**: Maintains document context and flow

### 💾 Smart Caching
- **Efficient**: Saves processed summaries for quick access
- **Space-Efficient**: MD5-based document hashing
- **Configurable**: Adjustable cache duration

### 📊 Analytics & Insights
- **Cost Tracking**: Monitor API usage costs
- **Token Analytics**: Detailed token usage statistics
- **Processing Metrics**: Track processing time and efficiency

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Internet connection
- Basic understanding of:
  - Python programming ([Python docs](https://docs.python.org/3/))
  - AI/ML concepts ([OpenAI guides](https://platform.openai.com/docs/guides/overview))

### Core Components
- **LangChain**: Framework for building LLM applications
  - [Documentation](https://python.langchain.com/docs/get_started)
  - [Components Guide](https://python.langchain.com/docs/modules)
- **OpenAI API**: Powers the AI summarization
  - [API Reference](https://platform.openai.com/docs/api-reference)
  - [Best Practices](https://platform.openai.com/docs/guides/best-practices)
- **Streamlit**: Creates the web interface
  - [Documentation](https://docs.streamlit.io)
  - [Components](https://docs.streamlit.io/library/api-reference)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/doc-sum.git
   cd doc-sum
   ```

2. **Set Up Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

## 💻 Usage

### Environment Setup Tips
- Ensure your Python environment is correctly configured
- Check your OpenAI API key has sufficient credits
- Monitor your API usage at [OpenAI Dashboard](https://platform.openai.com/usage)

### Web Interface (Recommended)

1. **Start the Application**
   ```bash
   streamlit run app.py
   ```

2. **Access the Interface**
   - Open your browser
   - Go to `http://localhost:8501`
   - You'll see the DocSum interface

3. **Using the Interface**
   - Upload your document using the file uploader
   - Configure settings in the sidebar (optional)
   - Click "Generate Summary"
   - View your summary and analytics

### Command Line Interface

1. **Run the CLI Version**
   ```bash
   python summarizer.py
   ```

2. **Follow the Prompts**
   - Enter the path to your document
   - Choose whether to use cached version
   - View the summary and statistics

### Advanced Configuration

#### Token Management
- **Token Limits**:
  - GPT-4: 8,192 tokens per request
  - GPT-3.5-Turbo-16k: 16,384 tokens per request
- **Cost Considerations**:
  - Monitor token usage in analytics
  - Adjust chunk size for cost optimization
  
#### Performance Tuning
- **Chunk Size**: 
  - Smaller (500-1000): Better for precise summaries
  - Larger (2000-3000): Better for context retention
- **Temperature Settings**:
  - 0.0-0.3: More factual, consistent
  - 0.4-0.7: Balanced creativity
  - 0.8-1.0: More creative, varied

## ⚙️ Configuration Options

### Model Settings
- **Temperature**: Controls creativity (0.0 - 1.0)
  - Lower: More focused, factual
  - Higher: More creative, varied
- **Model Selection**: Choose between
  - GPT-4 (Best quality)
  - GPT-3.5-Turbo (Faster, more economical)

### Processing Settings
- **Chunk Size**: Control text splitting (500-3000 chars)
- **Chunk Overlap**: Maintain context (50-1000 chars)
- **Cache Duration**: Set cache expiry (1-30 days)

## 📊 Understanding the Output

### Summary Section
- **Main Summary**: Concise overview of the document
- **Key Points**: Important information highlighted
- **Structure**: Maintains document's logical flow

### Analytics Dashboard
- **Processing Time**: Time taken to generate summary
- **Token Usage**: Number of tokens processed
- **Cost Analysis**: API usage cost
- **Compression Ratio**: Summary efficiency metrics

## 🛠️ Technical Architecture

### Component Overview
```
DocSum/
├── app.py           # Web interface
├── summarizer.py    # Core logic
├── requirements.txt # Dependencies
└── .env            # Configuration
```

### Process Flow
1. Document Upload → 2. Text Extraction → 3. Chunk Processing → 4. AI Summarization → 5. Results Display

## 🔄 Planned Improvements

### 1. Security Enhancements
- **API Key Management**: Secure key storage and rotation
- **User Authentication**: Access control and user management
- **Rate Limiting**: Prevent abuse and control costs
- **Data Encryption**: Secure document storage

### 2. Performance Optimization
- **Streaming Responses**: Real-time summary generation
- **Batch Processing**: Handle large documents efficiently
- **Background Tasks**: Asynchronous processing
- **Memory Management**: Efficient resource utilization

### 3. Feature Additions
- **Multi-language Support**: Detect and translate summaries
- **Export Options**: PDF, Word, and other formats
- **Custom Templates**: Summary style customization
- **Collaboration Tools**: Share and comment on summaries

### 4. Technical Improvements
- **Testing Framework**: Unit and integration tests
- **Monitoring System**: Performance and error tracking
- **Dependency Management**: Regular updates and security
- **Container Support**: Docker deployment options

### 5. UI/UX Enhancements
- **Theme Support**: Dark/light mode
- **Mobile Optimization**: Responsive design
- **Keyboard Shortcuts**: Improved accessibility
- **Interactive Tutorials**: User onboarding

## 🎓 Lessons Learned

### Working with ChromaDB Types (v0.6.3+)

When working with ChromaDB's collection operations (like `add`), you might encounter type-related issues. Here's a beginner-friendly guide to handle them:

#### The Problem
```python
# ❌ This might cause type errors
metadatas = [
    {k: v for k, v in chunk.metadata.items()}
    for chunk in chunks
]
```

#### The Solution
```python
# ✅ Import the correct type
from chromadb.types import Metadata

# ✅ Use the proper type annotation
metadatas: List[Metadata] = [
    {str(k): str(v) for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))} 
    for chunk in chunks
]
```

#### Why This Works
1. **Proper Types**: Using ChromaDB's own `Metadata` type ensures compatibility
2. **Type Safety**: Converting keys and values to strings prevents type mismatches
3. **Value Filtering**: Only allowing basic types (str, int, float, bool) prevents serialization issues

#### Key Takeaways for Beginners
1. Always import types from the correct module (`chromadb.types`)
2. Convert metadata values to strings to ensure compatibility
3. Filter out complex types that can't be stored in ChromaDB
4. Use proper type hints to catch issues early

This approach makes your code more reliable and easier to maintain while working with ChromaDB's vector store functionality.

### Working with LLM Models (LangChain + OpenAI)

#### Understanding Prompt Templates
When working with Language Models through LangChain, proper prompt engineering is crucial. Here's a beginner-friendly guide:

1. **Basic Prompt Structure**
```python
# ❌ Incorrect: No input variable
template = """Summarize this text."""

# ✅ Correct: Clear structure with input variable
template = """Write a detailed summary of the following text:
"{text}"
SUMMARY:"""
```

2. **Input Variables**
```python
# ❌ Incorrect: Missing input_variables
prompt = PromptTemplate(
    template="Summarize: {text}"
)

# ✅ Correct: Properly defined input_variables
prompt = PromptTemplate(
    template="Summarize: {text}",
    input_variables=["text"]
)
```

#### Chain Configuration
When using LangChain's document chains, ensure consistency between prompts and chain configuration:

1. **Document Variable Names**
```python
# ❌ Incorrect: Mismatched variables
prompt_template = "Process this: {content}"  # Uses 'content'
chain = StuffDocumentsChain(
    document_variable_name="text"  # But chain expects 'text'
)

# ✅ Correct: Consistent variable names
prompt_template = "Process this: {text}"  # Uses 'text'
chain = StuffDocumentsChain(
    document_variable_name="text"  # Chain also uses 'text'
)
```

2. **Map-Reduce Chains**
```python
# Example of a proper map-reduce setup
map_prompt = PromptTemplate(
    template="Summarize this part: {text}",
    input_variables=["text"]
)

combine_prompt = PromptTemplate(
    template="Combine these summaries: {text}",
    input_variables=["text"]
)

# Both chains use consistent variable names
map_chain = LLMChain(llm=llm, prompt=map_prompt)
combine_chain = StuffDocumentsChain(
    llm_chain=LLMChain(llm=llm, prompt=combine_prompt),
    document_variable_name="text"
)
```

#### Best Practices

1. **Prompt Design**
   - Use clear, specific instructions
   - Include example format in the prompt
   - Add section markers (e.g., "SUMMARY:", "OUTPUT:")
   - Use consistent variable names throughout

2. **Chain Configuration**
   - Match prompt variables with chain expectations
   - Use the same document variable name consistently
   - Initialize prompts before creating chains
   - Test with small inputs first

3. **Error Prevention**
   - Always specify input_variables in PromptTemplate
   - Keep prompt templates in a centralized location
   - Use type hints for better error catching
   - Add proper error handling around LLM calls

4. **Debugging Tips**
   - Print prompt templates to verify structure
   - Log actual values being passed to chains
   - Test prompts directly before using in chains
   - Use smaller chunks for initial testing

#### Common Pitfalls to Avoid

1. **Template Variables**
   - Not defining all used variables
   - Mismatched variable names
   - Missing input_variables declaration
   - Inconsistent variable naming

2. **Chain Setup**
   - Mismatched document_variable_name
   - Incorrect chain type for use case
   - Missing or incorrect prompt templates
   - Improper chain nesting

3. **Error Handling**
   - Not catching LLM-specific exceptions
   - Missing timeout handling
   - No fallback for failed calls
   - Insufficient logging

Remember: The key to successful LLM integration is consistency in variable naming and proper prompt structure. Always test your prompts independently before integrating them into chains.

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- 📧 Email: 
- 💬 Discord: 
- 📚 Documentation:

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/doc-sum&type=Date)](https://star-history.com/#yourusername/doc-sum&Date)

## 🔧 Troubleshooting

### Common Issues
1. **API Key Errors**
   - Verify key in `.env` file
   - Check API key permissions
   - Ensure sufficient API credits

2. **Memory Issues**
   - Reduce chunk size for large documents
   - Monitor system RAM usage
   - Use recommended file sizes (<10MB)

3. **Processing Errors**
   - Check file encoding (UTF-8 recommended)
   - Verify file permissions
   - Review log files for details

### Performance Optimization
- Use caching for repeated summaries
- Adjust chunk settings for your needs
- Monitor analytics for bottlenecks

## 📚 Additional Resources

### Learning Materials
- [LangChain Course](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)

### Community Support
- [LangChain Discord](https://discord.gg/6adMQxSpJS)
- [Streamlit Forums](https://discuss.streamlit.io)
- [OpenAI Community](https://community.openai.com)

---

<p align="center">Made with ❤️ by the DocSum Team</p>
