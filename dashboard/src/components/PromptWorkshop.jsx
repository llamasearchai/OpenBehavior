/**
 * Prompt engineering and testing workshop component
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Input,
  Select,
  Slider,
  Switch,
  Space,
  Tabs,
  Table,
  Tag,
  Alert,
  Spin,
  Typography,
  Tooltip,
  Modal,
  Form,
  Upload,
  message
} from 'antd';
import {
  PlayCircleOutlined,
  SaveOutlined,
  UploadOutlined,
  DownloadOutlined,
  ExperimentOutlined,
  BulbOutlined,
  SettingOutlined
} from '@ant-design/icons';
import { Controlled as CodeMirror } from 'react-codemirror2';
import 'codemirror/lib/codemirror.css';
import 'codemirror/theme/material.css';
import 'codemirror/mode/markdown/markdown';

import { promptApi } from '../services/api';
import { usePromptTemplates } from '../hooks/usePromptTemplates';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;

const PromptWorkshop = () => {
  const [activeTab, setActiveTab] = useState('editor');
  const [currentTemplate, setCurrentTemplate] = useState('');
  const [variables, setVariables] = useState({});
  const [testResult, setTestResult] = useState(null);
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  
  // Model settings
  const [selectedModel, setSelectedModel] = useState('gpt-4');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [topP, setTopP] = useState(1.0);
  
  // Optimization settings
  const [optimizationRunning, setOptimizationRunning] = useState(false);
  const [optimizationResults, setOptimizationResults] = useState([]);
  
  const {
    templates,
    loading: templatesLoading,
    createTemplate,
    updateTemplate,
    deleteTemplate
  } = usePromptTemplates();

  const extractVariables = (template) => {
    const regex = /\{([^}]+)\}/g;
    const vars = [];
    let match;
    
    while ((match = regex.exec(template)) !== null) {
      if (!vars.includes(match[1])) {
        vars.push(match[1]);
      }
    }
    
    return vars;
  };

  const handleTemplateChange = (value) => {
    setCurrentTemplate(value);
    
    // Extract variables and update state
    const extractedVars = extractVariables(value);
    const newVariables = { ...variables };
    
    extractedVars.forEach(varName => {
      if (!newVariables[varName]) {
        newVariables[varName] = '';
      }
    });
    
    setVariables(newVariables);
  };

  const handleTestPrompt = async () => {
    if (!currentTemplate.trim()) return;

    setTesting(true);
    try {
      const result = await promptApi.testPrompt({
        template: currentTemplate,
        variables,
        model: selectedModel,
        temperature,
        max_tokens: maxTokens
      });
      
      setTestResult(result);
    } catch (error) {
      message.error('Failed to test prompt');
      console.error('Test prompt error:', error);
    } finally {
      setTesting(false);
    }
  };

  const handleSaveTemplate = async () => {
    if (!currentTemplate.trim()) return;

    setSaving(true);
    try {
      await createTemplate({
        name: `Template ${Date.now()}`,
        template: currentTemplate,
        variables: Object.keys(variables),
        metadata: {
          model: selectedModel,
          temperature,
          max_tokens: maxTokens
        }
      });
      
      message.success('Template saved successfully');
    } catch (error) {
      message.error('Failed to save template');
      console.error('Save template error:', error);
    } finally {
      setSaving(false);
    }
  };

  const runOptimization = async () => {
    if (!currentTemplate.trim()) return;

    setOptimizationRunning(true);
    try {
      // This would call an optimization endpoint
      const results = await promptApi.optimizePrompt({
        template: currentTemplate,
        variables,
        iterations: 20,
        metrics: ['coherence', 'relevance', 'safety']
      });
      
      setOptimizationResults(results);
      message.success('Optimization completed');
    } catch (error) {
      message.error('Optimization failed');
      console.error('Optimization error:', error);
    } finally {
      setOptimizationRunning(false);
    }
  };

  const renderEditorTab = () => (
    <Row gutter={[16, 16]}>
      <Col span={16}>
        <Card title="Prompt Template Editor">
          <Space direction="vertical" style={{ width: '100%' }}>
            <div style={{ border: '1px solid #d9d9d9', borderRadius: '6px' }}>
              <CodeMirror
                value={currentTemplate}
                onBeforeChange={(editor, data, value) => {
                  handleTemplateChange(value);
                }}
                options={{
                  mode: 'markdown',
                  theme: 'material',
                  lineNumbers: true,
                  lineWrapping: true,
                  height: 'auto'
                }}
              />
            </div>
            
            {/* Variable Inputs */}
            {Object.keys(variables).length > 0 && (
              <Card size="small" title="Template Variables">
                <Row gutter={[8, 8]}>
                  {Object.keys(variables).map(varName => (
                    <Col span={12} key={varName}>
                      <Input
                        addonBefore={varName}
                        value={variables[varName]}
                        onChange={(e) => setVariables({
                          ...variables,
                          [varName]: e.target.value
                        })}
                        placeholder={`Enter value for ${varName}`}
                      />
                    </Col>
                  ))}
                </Row>
              </Card>
            )}
            
            {/* Action Buttons */}
            <Space>
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                loading={testing}
                onClick={handleTestPrompt}
                disabled={!currentTemplate.trim()}
              >
                Test Prompt
              </Button>
              
              <Button
                icon={<SaveOutlined />}
                loading={saving}
                onClick={handleSaveTemplate}
                disabled={!currentTemplate.trim()}
              >
                Save Template
              </Button>
              
              <Button
                icon={<ExperimentOutlined />}
                loading={optimizationRunning}
                onClick={runOptimization}
                disabled={!currentTemplate.trim()}
              >
                Optimize
              </Button>
            </Space>
          </Space>
        </Card>
      </Col>
      
      <Col span={8}>
        <Space direction="vertical" style={{ width: '100%' }}>
          {/* Model Settings */}
          <Card title="Model Settings" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>Model:</Text>
                <Select
                  value={selectedModel}
                  onChange={setSelectedModel}
                  style={{ width: '100%', marginTop: 4 }}
                >
                  <Option value="gpt-4">GPT-4</Option>
                  <Option value="gpt-3.5-turbo">GPT-3.5 Turbo</Option>
                  <Option value="claude-3-opus">Claude 3 Opus</Option>
                  <Option value="claude-3-sonnet">Claude 3 Sonnet</Option>
                </Select>
              </div>
              
              <div>
                <Text strong>Temperature: {temperature}</Text>
                <Slider
                  min={0}
                  max={2}
                  step={0.1}
                  value={temperature}
                  onChange={setTemperature}
                />
              </div>
              
              <div>
                <Text strong>Max Tokens: {maxTokens}</Text>
                <Slider
                  min={100}
                  max={4000}
                  step={100}
                  value={maxTokens}
                  onChange={setMaxTokens}
                />
              </div>
              
              <div>
                <Text strong>Top-P: {topP}</Text>
                <Slider
                  min={0}
                  max={1}
                  step={0.1}
                  value={topP}
                  onChange={setTopP}
                />
              </div>
            </Space>
          </Card>
          
          {/* Saved Templates */}
          <Card title="Saved Templates" size="small">
            {templatesLoading ? (
              <Spin />
            ) : (
              <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                {templates.map(template => (
                  <div
                    key={template.id}
                    style={{
                      padding: '8px',
                      cursor: 'pointer',
                      borderBottom: '1px solid #f0f0f0'
                    }}
                    onClick={() => {
                      setCurrentTemplate(template.template);
                      handleTemplateChange(template.template);
                    }}
                  >
                    <Text strong>{template.name}</Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {template.template.substring(0, 50)}...
                    </Text>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </Space>
      </Col>
    </Row>
  );

  const renderTestResultsTab = () => (
    <Row gutter={[16, 16]}>
      <Col span={12}>
        <Card title="Generated Response">
          {testResult ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <TextArea
                value={testResult.response}
                rows={15}
                readOnly
                style={{ fontFamily: 'monospace' }}
              />
              
              <Card size="small" title="Metadata">
                <Text>Model: {testResult.metadata?.model}</Text><br />
                <Text>Generation Time: {testResult.metadata?.generation_time?.toFixed(2)}s</Text><br />
                <Text>Response Length: {testResult.metadata?.response_length} chars</Text><br />
                <Text>Tokens Used: {testResult.metadata?.total_tokens}</Text>
              </Card>
            </Space>
          ) : (
            <Alert
              message="No test results yet"
              description="Run a prompt test to see results here"
              type="info"
            />
          )}
        </Card>
      </Col>
      
      <Col span={12}>
        <Card title="Formatted Prompt">
          {currentTemplate && Object.keys(variables).length > 0 ? (
            <TextArea
              value={Object.keys(variables).reduce(
                (template, varName) => template.replace(
                  new RegExp(`\\{${varName}\\}`, 'g'),
                  variables[varName] || `{${varName}}`
                ),
                currentTemplate
              )}
              rows={15}
              readOnly
              style={{ fontFamily: 'monospace' }}
            />
          ) : (
            <Alert
              message="Template preview will appear here"
              description="Enter a template with variables to see the formatted result"
              type="info"
            />
          )}
        </Card>
      </Col>
    </Row>
  );

  const renderOptimizationTab = () => (
    <Row gutter={[16, 16]}>
      <Col span={24}>
        <Card title="Prompt Optimization Results">
          {optimizationResults.length > 0 ? (
            <Table
              dataSource={optimizationResults}
              rowKey="iteration"
              pagination={false}
              columns={[
                {
                  title: 'Iteration',
                  dataKey: 'iteration',
                  width: 100
                },
                {
                  title: 'Template Variant',
                  dataKey: 'template',
                  ellipsis: true,
                  render: (template) => (
                    <Tooltip title={template}>
                      <Text code>{template.substring(0, 100)}...</Text>
                    </Tooltip>
                  )
                },
                {
                  title: 'Coherence Score',
                  dataKey: 'coherence_score',
                  width: 120,
                  render: (score) => (
                    <Tag color={score > 0.8 ? 'green' : score > 0.6 ? 'orange' : 'red'}>
                      {(score || 0).toFixed(3)}
                    </Tag>
                  )
                },
                {
                  title: 'Relevance Score',
                  dataKey: 'relevance_score',
                  width: 120,
                  render: (score) => (
                    <Tag color={score > 0.8 ? 'green' : score > 0.6 ? 'orange' : 'red'}>
                      {(score || 0).toFixed(3)}
                    </Tag>
                  )
                },
                {
                  title: 'Safety Score',
                  dataKey: 'safety_score',
                  width: 120,
                  render: (score) => (
                    <Tag color={score > 0.8 ? 'green' : score > 0.6 ? 'orange' : 'red'}>
                      {(score || 0).toFixed(3)}
                    </Tag>
                  )
                },
                {
                  title: 'Overall Score',
                  dataKey: 'overall_score',
                  width: 120,
                  render: (score) => (
                    <Tag color={score > 0.8 ? 'green' : score > 0.6 ? 'orange' : 'red'}>
                      <strong>{(score || 0).toFixed(3)}</strong>
                    </Tag>
                  )
                },
                {
                  title: 'Actions',
                  width: 100,
                  render: (_, record) => (
                    <Button
                      type="link"
                      size="small"
                      onClick={() => {
                        setCurrentTemplate(record.template);
                        handleTemplateChange(record.template);
                        setActiveTab('editor');
                      }}
                    >
                      Use This
                    </Button>
                  )
                }
              ]}
            />
          ) : (
            <Alert
              message="No optimization results yet"
              description="Run prompt optimization to see results here"
              type="info"
              icon={<BulbOutlined />}
            />
          )}
        </Card>
      </Col>
    </Row>
  );

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>Prompt Workshop</Title>
        <Text type="secondary">
          Engineer, test, and optimize prompts for better model behavior
        </Text>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="Editor" key="editor">
          {renderEditorTab()}
        </TabPane>
        
        <TabPane tab="Test Results" key="results">
          {renderTestResultsTab()}
        </TabPane>
        
        <TabPane tab="Optimization" key="optimization">
          {renderOptimizationTab()}
        </TabPane>
      </Tabs>
    </div>
  );
};

export default PromptWorkshop;