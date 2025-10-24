# LangChain Text Splitters

A comprehensive demonstration of different text splitting techniques using LangChain for document processing and analysis in AI/ML applications.

## Overview

Text splitters are essential components in building RAG (Retrieval-Augmented Generation) systems and LLM applications. They break down large documents into smaller, manageable chunks while preserving semantic meaning and context. This project demonstrates three different approaches to text splitting using LangChain.

## Why Text Splitting?

- **Token Limits**: LLMs have input token limitations
- **Better Retrieval**: Smaller chunks improve semantic search accuracy
- **Context Management**: Maintain relevant context within each chunk
- **Performance**: Optimize processing speed and efficiency

## Project Structure

### 1. Length-Based Splitter (`1_length_based_splitter.py`)

Demonstrates the basic **CharacterTextSplitter** approach:
- Splits text based on character count
- Uses a simple separator (empty string in this case)
- Fixed chunk size: 100 characters
- No overlap between chunks

**Use Case**: Simple text splitting where structure preservation is not critical.

```python
text_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)
```

### 2. Document Loader with Splitter (`2_doc_loader_with_splitter.py`)

Shows how to combine document loading with text splitting:
- Uses **PyPDFLoader** to load PDF documents
- Applies **CharacterTextSplitter** to the loaded documents
- Chunk size: 200 characters
- Demonstrates processing real-world documents

**Use Case**: Processing PDF documents for RAG systems or document analysis.

```python
loader = PyPDFLoader("indian_economy_report_2025.pdf")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
result = text_splitter.split_documents(docs)
```

### 3. Structure-Based Splitter (`3_text_structure_based_splitter.py`)

Demonstrates the **RecursiveCharacterTextSplitter** - the most intelligent approach:
- Respects natural text boundaries (paragraphs, lines, sentences, words)
- Uses hierarchical separators: `["\n\n", "\n", " ", ""]`
- Chunk size: 300 characters
- Preserves document structure and readability

**Use Case**: When maintaining semantic coherence and natural text structure is important.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""]
)
```

## Setup and Installation

### Prerequisites

- Python 3.12
- Conda (recommended) or pip

### Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd 8_LangChain_Text_Splitters
   ```

2. **Create and activate the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate for_genai
   ```

   Or update an existing environment:
   ```bash
   conda env update -f environment.yml
   ```

3. **Set up environment variables**:
   - Create a `.env` file (if not present)
   - Add any required API keys

## Dependencies

Key packages used in this project:
- **langchain**: Core framework for LLM applications
- **langchain-community**: Community-contributed integrations
- **pypdf**: PDF processing
- **python-dotenv**: Environment variable management

Additional AI framework integrations included:
- OpenAI
- Anthropic (Claude)
- Google Generative AI
- Hugging Face

See `environment.yml` for the complete list of dependencies.

## Usage

Run any of the examples:

```bash
# Example 1: Basic character-based splitting
python 1_length_based_splitter.py

# Example 2: PDF document splitting
python 2_doc_loader_with_splitter.py

# Example 3: Structure-aware splitting
python 3_text_structure_based_splitter.py
```

## Key Concepts

### Chunk Size
The maximum number of characters in each chunk. Larger chunks retain more context but may exceed token limits.

### Chunk Overlap
The number of characters that overlap between consecutive chunks. Helps maintain context across chunk boundaries (set to 0 in these examples).

### Separators
Characters or strings used to split text. The `RecursiveCharacterTextSplitter` tries separators in order, using the first one that results in chunks below the size limit.

## Best Practices

1. **Choose the right splitter**:
   - Use `CharacterTextSplitter` for simple, uniform splitting
   - Use `RecursiveCharacterTextSplitter` for preserving document structure
   - Consider specialized splitters for code, markdown, or other formats

2. **Set appropriate chunk sizes**:
   - Consider your LLM's context window
   - Balance between context preservation and chunk size
   - Test different sizes for your specific use case

3. **Use chunk overlap**:
   - Add overlap (e.g., 10-20% of chunk size) for better context continuity
   - Essential for semantic search and QA applications

4. **Test with real data**:
   - Evaluate chunk quality with actual documents
   - Verify that semantic meaning is preserved

## Example Output

When running `3_text_structure_based_splitter.py`, you'll see:
- Number of chunks created
- The content of each chunk, respecting paragraph boundaries

## Resources

- [LangChain Documentation](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [LangChain Text Splitters Guide](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter)

## Author

Srinivas Jagadeesh
