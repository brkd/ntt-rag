import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.documents import Document
from vectorstore.vectorstore import VectorStoreBuilder


@pytest.fixture
def mock_embedding_model():
    """Mock the HuggingFace embeddings model"""
    with patch('vectorstore.vectorstore.HuggingFaceEmbeddings') as mock:
        mock.return_value = Mock()
        yield mock


@pytest.fixture
def mock_chroma():
    """Mock the Chroma vector store"""
    with patch('vectorstore.vectorstore.Chroma') as mock:
        mock_instance = Mock()
        mock_instance.add_documents = Mock(return_value=None)
        mock_instance.asimilarity_search_with_score = AsyncMock(
            return_value=[
                (Document(page_content="Test content 1"), 0.95),
                (Document(page_content="Test content 2"), 0.85)
            ]
        )
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def vector_store_builder(mock_embedding_model, mock_chroma):
    """Create a VectorStoreBuilder instance with mocked dependencies"""
    return VectorStoreBuilder(
        collection_name="test_collection",
        host="localhost",
        port=8000,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )


def test_initialization(vector_store_builder, mock_embedding_model, mock_chroma):
    """Test that VectorStoreBuilder initializes correctly"""
    mock_embedding_model.assert_called_once_with(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    mock_chroma.assert_called_once()
    call_kwargs = mock_chroma.call_args.kwargs
    assert call_kwargs['collection_name'] == "test_collection"
    assert call_kwargs['host'] == "localhost"
    assert call_kwargs['port'] == 8000


def test_add_documents(vector_store_builder):
    """Test adding documents to the vector store"""
    documents = [
        Document(page_content="Document 1", metadata={"source": "test1"}),
        Document(page_content="Document 2", metadata={"source": "test2"})
    ]
    
    vector_store_builder.add(documents)
    
    vector_store_builder.vector_store.add_documents.assert_called_once_with(
        documents=documents
    )



@pytest.mark.asyncio
async def test_search(vector_store_builder):
    """Test searching for similar documents"""
    query = "test query"
    k = 2
    
    results = await vector_store_builder.search(query, k=k)
    
    vector_store_builder.vector_store.asimilarity_search_with_score.assert_called_once_with(
        query, k=k
    )
    
    assert len(results) == 2
    assert isinstance(results[0][0], Document)
    assert isinstance(results[0][1], float)
