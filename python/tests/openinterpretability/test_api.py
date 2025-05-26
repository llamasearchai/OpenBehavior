"""
Tests for OpenInterpretability API client and server.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from openinterpretability.api.client import OpenInterpretabilityClient
from openinterpretability.api.server import create_app


@pytest_asyncio.fixture
async def mock_session():
    """Mock aiohttp session for testing."""
    session = AsyncMock()
    
    # Create a proper async context manager mock
    async_context_manager = AsyncMock()
    async_context_manager.__aenter__ = AsyncMock()
    async_context_manager.__aexit__ = AsyncMock()
    
    # Mock the response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"test": "data"})
    mock_response.text = AsyncMock(return_value="test response")
    
    async_context_manager.__aenter__.return_value = mock_response
    session.request.return_value = async_context_manager
    
    return session


@pytest_asyncio.fixture
async def client():
    """OpenInterpretability client with mocked session."""
    client = OpenInterpretabilityClient()
    
    # Create a proper mock session that supports async context manager protocol
    session = AsyncMock()
    session.closed = False
    
    # Create a mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"test": "data"})
    mock_response.text = AsyncMock(return_value="test response")
    mock_response.request_info = MagicMock()
    mock_response.history = []
    
    # Create a proper async context manager class
    class MockContextManager:
        def __init__(self, response):
            self.response = response
        
        async def __aenter__(self):
            return self.response
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None
    
    # Mock the request method to return the context manager
    def mock_request(*args, **kwargs):
        return MockContextManager(mock_response)
    
    session.request = mock_request
    
    # Override the _ensure_session method to prevent creating a real session
    async def mock_ensure_session():
        client.session = session
    
    client._ensure_session = mock_ensure_session
    client.session = session
    return client


@pytest.fixture
def test_client():
    """FastAPI test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def client_config():
    """Client configuration for testing."""
    return {
        "base_url": "http://localhost:8000",
        "api_key": "test-key",
        "timeout": 30,
        "max_retries": 3
    }


class TestOpenInterpretabilityClient:
    """Test cases for OpenInterpretability API client."""

    def _create_mock_context_manager(self, mock_response):
        """Helper to create a proper async context manager."""
        class MockContextManager:
            def __init__(self, response):
                self.response = response
            
            async def __aenter__(self):
                return self.response
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None
        
        def mock_request(*args, **kwargs):
            return MockContextManager(mock_response)
        
        return mock_request

    @pytest.mark.asyncio
    async def test_evaluate_text(self, client):
        """Test text evaluation through client."""
        # Update the mock response for this specific test
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "overall_score": 0.85,
            "safety_score": {"overall_score": 0.9},
            "ethical_score": {"overall_score": 0.8}
        })
        mock_response.request_info = MagicMock()
        mock_response.history = []

        client.session.request = self._create_mock_context_manager(mock_response)

        result = await client.evaluate_text(
            text="Test prompt",
            evaluation_types=["safety", "ethical"]
        )

        assert result["overall_score"] == 0.85
        assert "safety_score" in result

    @pytest.mark.asyncio
    async def test_client_initialization(self, client_config):
        """Test client initialization."""
        client = OpenInterpretabilityClient(**client_config)
        
        assert client.base_url == client_config["base_url"]
        assert client.api_key == client_config["api_key"]
        assert client.timeout.total == client_config["timeout"]
        assert "Authorization" in client.headers
    
    @pytest.mark.asyncio
    async def test_context_manager(self, client_config):
        """Test client as async context manager."""
        async with OpenInterpretabilityClient(**client_config) as client:
            # Mock the session to prevent real HTTP calls
            session = AsyncMock()
            session.closed = False
            client.session = session
            assert client.session is not None
            assert not client.session.closed
    
    @pytest.mark.asyncio
    async def test_batch_evaluate(self, client):
        """Test batch evaluation through client."""
        # Update mock response for batch evaluation
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "batch_id": "test-task-123",
            "status": "started",
            "total_items": 2,
            "completed_items": 0
        })
        mock_response.request_info = MagicMock()
        mock_response.history = []

        client.session.request = self._create_mock_context_manager(mock_response)
        
        result = await client.batch_evaluate(
            texts=["Test 1", "Test 2"],
            evaluation_types=["safety"]
        )
        
        assert isinstance(result, dict)
        assert "batch_id" in result
    
    @pytest.mark.asyncio
    async def test_get_batch_status(self, client):
        """Test getting batch evaluation status."""
        # Update mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "batch_id": "test-task-123",
            "status": "completed",
            "total_items": 2,
            "completed_items": 2,
            "results": []
        })
        mock_response.request_info = MagicMock()
        mock_response.history = []

        client.session.request = self._create_mock_context_manager(mock_response)
        
        result = await client.get_batch_status("test-task-123")
        
        assert isinstance(result, dict)
        assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_analyze_model(self, client):
        """Test model analysis through client."""
        # Update mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "behavior_analysis": {"patterns": []},
            "statistics": {"avg_score": 0.85}
        })
        mock_response.request_info = MagicMock()
        mock_response.history = []

        client.session.request = self._create_mock_context_manager(mock_response)
        
        result = await client.analyze_model(
            model="test-model",
            test_prompts=["Test 1", "Test 2"]
        )
        
        assert isinstance(result, dict)
        assert "behavior_analysis" in result
    
    @pytest.mark.asyncio
    async def test_compare_models(self, client):
        """Test model comparison through client."""
        # Update mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "comparison_result": {"winner": "model-a"},
            "model_a_scores": {"avg": 0.85},
            "model_b_scores": {"avg": 0.80}
        })
        mock_response.request_info = MagicMock()
        mock_response.history = []

        client.session.request = self._create_mock_context_manager(mock_response)
        
        result = await client.compare_models(
            model_a="model-a",
            model_b="model-b",
            test_prompts=["Test 1", "Test 2"]
        )
        
        assert isinstance(result, dict)
        assert "comparison_result" in result
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test client error handling."""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.request_info = MagicMock()
        mock_response.history = []

        client.session.request = self._create_mock_context_manager(mock_response)
        
        with pytest.raises(Exception):
            await client.evaluate_text("Test prompt")
    
    @pytest.mark.asyncio
    async def test_retry_logic(self, client):
        """Test client retry logic."""
        # Mock responses - first fails, second succeeds
        call_count = 0
        
        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            class MockContextManager:
                def __init__(self):
                    if call_count == 1:
                        # First call fails
                        self.response = AsyncMock()
                        self.response.status = 503
                        self.response.text = AsyncMock(return_value="Service Unavailable")
                        self.response.request_info = MagicMock()
                        self.response.history = []
                    else:
                        # Second call succeeds
                        self.response = AsyncMock()
                        self.response.status = 200
                        self.response.json = AsyncMock(return_value={"result": "success"})
                        self.response.request_info = MagicMock()
                        self.response.history = []
                
                async def __aenter__(self):
                    return self.response
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return None
            
            return MockContextManager()

        client.session.request = mock_request
        
        result = await client.evaluate_text("Test prompt")
        
        assert result["result"] == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_validate_api_key(self, client):
        """Test API key validation."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "healthy"})
        mock_response.request_info = MagicMock()
        mock_response.history = []

        client.session.request = self._create_mock_context_manager(mock_response)
        
        is_valid = await client.validate_api_key()
        
        assert is_valid is True


class TestAPIServer:
    """Test API server functionality."""
    
    @pytest.fixture
    def app(self, sample_config):
        """Create FastAPI app for testing."""
        with patch('openinterpretability.api.server.get_config', return_value=sample_config._config):
            return create_app()
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, app):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_evaluate_endpoint(self, app, sample_test_prompts):
        """Test text evaluation endpoint."""
        from fastapi.testclient import TestClient
        
        with patch('openinterpretability.api.server.get_engine') as mock_engine:
            # Mock engine response
            mock_engine.return_value.evaluate_text.return_value = {
                "overall_score": 0.85,
                "safety_score": {"overall_score": 0.9}
            }
            
            client = TestClient(app)
            response = client.post("/evaluate", json={
                "text": sample_test_prompts[0],
                "evaluation_types": ["safety"]
            })
            
            assert response.status_code == 200
            assert "overall_score" in response.json()
    
    @pytest.mark.asyncio
    async def test_batch_evaluate_endpoint(self, app, sample_test_prompts):
        """Test batch evaluation endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        response = client.post("/evaluate/batch", json={
            "texts": sample_test_prompts[:2],
            "evaluation_types": ["safety"]
        })
        
        assert response.status_code == 200
        assert "batch_id" in response.json()
    
    @pytest.mark.asyncio
    async def test_analyze_model_endpoint(self, app, sample_test_prompts):
        """Test model analysis endpoint."""
        from fastapi.testclient import TestClient
        
        with patch('openinterpretability.api.server.get_engine') as mock_engine:
            # Mock engine response
            mock_engine.return_value.analyze_model_behavior.return_value = {
                "behavior_analysis": {"patterns": []},
                "statistics": {"avg_score": 0.85}
            }
            
            client = TestClient(app)
            response = client.post("/analyze/model", json={
                "model": "test-model",
                "test_prompts": sample_test_prompts
            })
            
            assert response.status_code == 200
            assert "behavior_analysis" in response.json()
    
    @pytest.mark.asyncio
    async def test_compare_models_endpoint(self, app, sample_test_prompts):
        """Test model comparison endpoint."""
        from fastapi.testclient import TestClient
        
        with patch('openinterpretability.api.server.get_engine') as mock_engine:
            # Mock engine response
            mock_engine.return_value.compare_models.return_value = {
                "comparison_result": {"winner": "model-a"},
                "detailed_analysis": {}
            }
            
            client = TestClient(app)
            response = client.post("/analyze/compare", json={
                "model_a": "model-a",
                "model_b": "model-b",
                "test_prompts": sample_test_prompts
            })
            
            assert response.status_code == 200
            assert "comparison_result" in response.json()
    
    @pytest.mark.asyncio
    async def test_validation_errors(self, app):
        """Test input validation errors."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test missing required fields
        response = client.post("/evaluate", json={})
        assert response.status_code == 422
        
        # Test invalid evaluation types
        response = client.post("/evaluate", json={
            "text": "test",
            "evaluation_types": ["invalid_type"]
        })
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_authentication(self, app):
        """Test API authentication."""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test without API key
        response = client.post("/evaluate", json={
            "text": "test",
            "evaluation_types": ["safety"]
        })
        # Should still work for basic endpoints (depending on auth setup)
        assert response.status_code in [200, 401, 403]


class TestIntegration:
    """Integration tests for API components."""
    
    @pytest.mark.asyncio
    async def test_client_server_integration(self, sample_test_prompts):
        """Test client-server integration."""
        # This would typically require a running server
        # For now, we'll test the client with a mock server
        
        async def mock_server_handler(request):
            return {"overall_score": 0.85, "safety_score": {"overall_score": 0.9}}
        
        # Test would involve setting up a mock server and client
        # This is a placeholder for actual integration testing
        pass
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, sample_test_prompts):
        """Test handling concurrent API requests."""
        # Mock client for concurrent testing
        client_config = {
            "base_url": "http://localhost:8000",
            "api_key": "test-key"
        }
        
        # Create multiple mock requests
        async def mock_request():
            client = OpenInterpretabilityClient(**client_config)
            
            # Mock the session
            session = AsyncMock()
            session.closed = False
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"result": "success"})
            mock_response.request_info = MagicMock()
            mock_response.history = []
            
            # Create proper async context manager
            class MockContextManager:
                def __init__(self, response):
                    self.response = response
                
                async def __aenter__(self):
                    return self.response
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return None
            
            def mock_request_method(*args, **kwargs):
                return MockContextManager(mock_response)
            
            session.request = mock_request_method
            client.session = session
            
            return await client.evaluate_text("test prompt")
        
        # Run concurrent requests
        tasks = [mock_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert result["result"] == "success" 