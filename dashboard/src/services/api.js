/**
 * API service for communicating with OpenBehavior backend
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

const apiRequest = async (endpoint, options = {}) => {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, config);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
        response.status
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    
    // Network or other errors
    throw new ApiError(`Network error: ${error.message}`, 0);
  }
};

export const evaluationApi = {
  // Evaluate single text
  evaluateText: async (data) => {
    return apiRequest('/evaluate', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  // Start batch evaluation
  startBatchEvaluation: async (data) => {
    return apiRequest('/evaluate/batch', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  // Get batch evaluation status
  getBatchStatus: async (taskId) => {
    return apiRequest(`/evaluate/batch/${taskId}`);
  },

  // Get evaluation statistics
  getEvaluationStats: async (filters = {}) => {
    const params = new URLSearchParams(filters).toString();
    return apiRequest(`/evaluations/stats?${params}`);
  },

  // Get recent evaluations
  getRecentEvaluations: async (limit = 50) => {
    return apiRequest(`/evaluations/recent?limit=${limit}`);
  },
};

export const promptApi = {
  // Test a prompt
  testPrompt: async (data) => {
    return apiRequest('/prompts/test', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  // Get available templates
  getTemplates: async () => {
    return apiRequest('/prompts/templates');
  },

  // Get specific template
  getTemplate: async (templateId) => {
    return apiRequest(`/prompts/templates/${templateId}`);
  },

  // Create new template
  createTemplate: async (data) => {
    return apiRequest('/prompts/templates', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  // Update template
  updateTemplate: async (templateId, data) => {
    return apiRequest(`/prompts/templates/${templateId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  // Delete template
  deleteTemplate: async (templateId) => {
    return apiRequest(`/prompts/templates/${templateId}`, {
      method: 'DELETE',
    });
  },

  // Optimize prompt
  optimizePrompt: async (data) => {
    return apiRequest('/prompts/optimize', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },
};

export const modelApi = {
  // Get model usage statistics
  getUsageStats: async () => {
    return apiRequest('/models/usage');
  },

  // Get available models
  getAvailableModels: async () => {
    return apiRequest('/models/available');
  },
};

export const configApi = {
  // Get current configuration
  getConfig: async () => {
    return apiRequest('/config');
  },

  // Update configuration
  updateConfig: async (data) => {
    return apiRequest('/config', {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },
};

export const healthApi = {
  // Health check
  checkHealth: async () => {
    return apiRequest('/health');
  },
};

// Helper function for handling long-running tasks
export const pollTaskStatus = async (taskId, pollInterval = 2000, timeout = 300000) => {
  const startTime = Date.now();
  
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        if (Date.now() - startTime > timeout) {
          reject(new Error('Task polling timeout'));
          return;
        }

        const status = await evaluationApi.getBatchStatus(taskId);
        
        if (status.status === 'completed') {
          resolve(status);
        } else if (status.status === 'failed') {
          reject(new Error(status.error || 'Task failed'));
        } else {
          // Still running, continue polling
          setTimeout(poll, pollInterval);
        }
      } catch (error) {
        reject(error);
      }
    };

    poll();
  });
};

export default {
  evaluationApi,
  promptApi,
  modelApi,
  configApi,
  healthApi,
  pollTaskStatus,
};