"""Tests for OpenInterpretability CLI components."""

import pytest
import tempfile
import json
from unittest.mock import Mock, patch, AsyncMock
from click.testing import CliRunner
from pathlib import Path

from openinterpretability.cli.main import cli


class TestCLI:
    """Test CLI functionality."""
    
    @pytest.fixture
    def runner(self):
        """CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def temp_config(self, temp_dir):
        """Create temporary config file."""
        config = {
            "model": {
                "provider": "openai",
                "name": "gpt-4",
                "api_key": "test-key"
            },
            "data": {
                "template_dir": str(temp_dir / "templates"),
                "output_dir": str(temp_dir / "output")
            }
        }
        config_file = temp_dir / "config.yaml"
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        return str(config_file)
    
    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "OpenInterpretability CLI" in result.output
    
    def test_version_command(self, runner):
        """Test version command."""
        result = runner.invoke(cli, ['version'])
        assert result.exit_code == 0
        assert "OpenInterpretability" in result.output
    
    @patch('openinterpretability.cli.main.get_engine')
    def test_evaluate_command(self, mock_get_engine, runner, temp_config):
        """Test evaluate command."""
        # Mock engine
        mock_engine = AsyncMock()
        mock_engine.evaluate_text.return_value = {
            "overall_score": 0.85,
            "safety_score": {"overall_score": 0.9}
        }
        mock_get_engine.return_value = mock_engine
        
        result = runner.invoke(cli, [
            '--config', temp_config,
            'evaluate',
            '--text', 'Test prompt',
            '--types', 'safety,ethical'
        ])
        
        assert result.exit_code == 0
        assert "Overall Score" in result.output
    
    @patch('openinterpretability.cli.main.get_engine')
    def test_batch_evaluate_command(self, mock_get_engine, runner, temp_config, temp_dir):
        """Test batch evaluate command."""
        # Create test file with prompts
        prompts_file = temp_dir / "prompts.txt"
        with open(prompts_file, 'w') as f:
            f.write("Test prompt 1\n")
            f.write("Test prompt 2\n")
        
        # Mock engine
        mock_engine = AsyncMock()
        mock_engine.batch_evaluate.return_value = [
            {"overall_score": 0.85, "prompt": "Test prompt 1"},
            {"overall_score": 0.90, "prompt": "Test prompt 2"}
        ]
        mock_get_engine.return_value = mock_engine
        
        result = runner.invoke(cli, [
            '--config', temp_config,
            'batch-evaluate',
            '--input', str(prompts_file),
            '--types', 'safety'
        ])
        
        assert result.exit_code == 0
        assert "Batch evaluation completed" in result.output
    
    @patch('openinterpretability.cli.main.get_engine')
    def test_analyze_command(self, mock_get_engine, runner, temp_config):
        """Test analyze command."""
        # Mock engine
        mock_engine = AsyncMock()
        mock_engine.analyze_model_behavior.return_value = {
            "behavior_analysis": {"patterns": []},
            "statistics": {"avg_score": 0.85}
        }
        mock_get_engine.return_value = mock_engine
        
        result = runner.invoke(cli, [
            '--config', temp_config,
            'analyze',
            '--model', 'gpt-4',
            '--prompts', 'Test 1', 'Test 2',
            '--depth', 'standard'
        ])
        
        assert result.exit_code == 0
        assert "Model analysis completed" in result.output
    
    @patch('openinterpretability.cli.main.get_engine')
    def test_compare_command(self, mock_get_engine, runner, temp_config):
        """Test compare command."""
        # Mock engine
        mock_engine = AsyncMock()
        mock_engine.compare_models.return_value = {
            "comparison_result": {"winner": "model-a"},
            "model_a_scores": {"avg": 0.85},
            "model_b_scores": {"avg": 0.80}
        }
        mock_get_engine.return_value = mock_engine
        
        result = runner.invoke(cli, [
            '--config', temp_config,
            'compare',
            '--model-a', 'gpt-4',
            '--model-b', 'gpt-3.5-turbo',
            '--prompts', 'Test 1', 'Test 2'
        ])
        
        assert result.exit_code == 0
        assert "Model comparison completed" in result.output
    
    def test_config_validation(self, runner):
        """Test config validation."""
        # Test with non-existent config file
        result = runner.invoke(cli, [
            '--config', '/non/existent/config.yaml',
            'version'
        ])
        
        assert result.exit_code != 0
        assert "Configuration file not found" in result.output
    
    @patch('openinterpretability.cli.main.get_client')
    def test_interactive_mode(self, mock_get_client, runner, temp_config):
        """Test interactive mode."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.evaluate_text.return_value = {
            "overall_score": 0.85
        }
        mock_get_client.return_value = mock_client
        
        # Simulate user input
        result = runner.invoke(cli, [
            '--config', temp_config,
            'interactive'
        ], input='Test prompt\nquit\n')
        
        assert result.exit_code == 0
    
    @patch('openinterpretability.cli.main.get_engine')
    def test_output_formats(self, mock_get_engine, runner, temp_config, temp_dir):
        """Test different output formats."""
        # Mock engine
        mock_engine = AsyncMock()
        mock_engine.evaluate_text.return_value = {
            "overall_score": 0.85,
            "safety_score": {"overall_score": 0.9}
        }
        mock_get_engine.return_value = mock_engine
        
        # Test JSON output
        result = runner.invoke(cli, [
            '--config', temp_config,
            'evaluate',
            '--text', 'Test prompt',
            '--output-format', 'json'
        ])
        
        assert result.exit_code == 0
        # Should be valid JSON
        json.loads(result.output.strip())
        
        # Test CSV output
        output_file = temp_dir / "output.csv"
        result = runner.invoke(cli, [
            '--config', temp_config,
            'evaluate',
            '--text', 'Test prompt',
            '--output-format', 'csv',
            '--output', str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
    
    def test_error_handling(self, runner, temp_config):
        """Test error handling in CLI."""
        # Test with invalid model
        with patch('openinterpretability.cli.main.get_engine') as mock_get_engine:
            mock_get_engine.side_effect = Exception("Model not found")
            
            result = runner.invoke(cli, [
                '--config', temp_config,
                'evaluate',
                '--text', 'Test prompt'
            ])
            
            assert result.exit_code != 0
            assert "Error" in result.output
    
    @patch('openinterpretability.cli.main.get_engine')
    def test_verbose_mode(self, mock_get_engine, runner, temp_config):
        """Test verbose mode."""
        # Mock engine
        mock_engine = AsyncMock()
        mock_engine.evaluate_text.return_value = {
            "overall_score": 0.85
        }
        mock_get_engine.return_value = mock_engine
        
        result = runner.invoke(cli, [
            '--config', temp_config,
            '--verbose',
            'evaluate',
            '--text', 'Test prompt'
        ])
        
        assert result.exit_code == 0
        # Verbose mode should show more output
        assert len(result.output) > 100


class TestConfigManagement:
    """Test configuration management in CLI."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_config_init(self, runner, temp_dir):
        """Test config initialization."""
        config_file = temp_dir / "new_config.yaml"
        
        result = runner.invoke(cli, [
            'config', 'init',
            '--output', str(config_file)
        ])
        
        assert result.exit_code == 0
        assert config_file.exists()
        
        # Check if config is valid
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        assert "model" in config
        assert "data" in config
    
    def test_config_validate(self, runner, temp_config):
        """Test config validation."""
        result = runner.invoke(cli, [
            'config', 'validate',
            '--config-file', temp_config
        ])
        
        assert result.exit_code == 0
        assert "valid" in result.output.lower()
    
    def test_config_show(self, runner, temp_config):
        """Test config display."""
        result = runner.invoke(cli, [
            '--config', temp_config,
            'config', 'show'
        ])
        
        assert result.exit_code == 0
        assert "model" in result.output


class TestDataManagement:
    """Test data management commands."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('openinterpretability.cli.main.DatabaseManager')
    def test_data_export(self, mock_db_manager, runner, temp_config, temp_dir):
        """Test data export."""
        # Mock database manager
        mock_db = Mock()
        mock_db.export_data.return_value = {"exported": 100}
        mock_db_manager.return_value = mock_db
        
        output_file = temp_dir / "export.json"
        
        result = runner.invoke(cli, [
            '--config', temp_config,
            'data', 'export',
            '--output', str(output_file)
        ])
        
        assert result.exit_code == 0
        mock_db.export_data.assert_called_once()
    
    @patch('openinterpretability.cli.main.DatabaseManager')
    def test_data_import(self, mock_db_manager, runner, temp_config, temp_dir):
        """Test data import."""
        # Create test data file
        data_file = temp_dir / "import.json"
        test_data = {"evaluations": [{"id": 1, "score": 0.85}]}
        with open(data_file, 'w') as f:
            json.dump(test_data, f)
        
        # Mock database manager
        mock_db = Mock()
        mock_db.import_data.return_value = {"imported": 1}
        mock_db_manager.return_value = mock_db
        
        result = runner.invoke(cli, [
            '--config', temp_config,
            'data', 'import',
            '--input', str(data_file)
        ])
        
        assert result.exit_code == 0
        mock_db.import_data.assert_called_once()
    
    @patch('openinterpretability.cli.main.DatabaseManager')
    def test_data_stats(self, mock_db_manager, runner, temp_config):
        """Test data statistics."""
        # Mock database manager
        mock_db = Mock()
        mock_db.get_statistics.return_value = {
            "total_evaluations": 150,
            "avg_score": 0.85
        }
        mock_db_manager.return_value = mock_db
        
        result = runner.invoke(cli, [
            '--config', temp_config,
            'data', 'stats'
        ])
        
        assert result.exit_code == 0
        assert "150" in result.output
        assert "0.85" in result.output 