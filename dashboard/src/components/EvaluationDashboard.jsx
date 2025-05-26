/**
 * Main evaluation dashboard component
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Select,
  Button,
  Input,
  Space,
  Tabs,
  Alert,
  Spin,
  Typography,
  Tag,
  Tooltip
} from 'antd';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';
import {
  EyeOutlined,
  ExperimentOutlined,
  SafetyOutlined,
  HeartOutlined,
  BarChartOutlined,
  ReloadOutlined
} from '@ant-design/icons';

import { useEvaluationData } from '../hooks/useEvaluationData';
import { evaluationApi } from '../services/api';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const RISK_COLORS = {
  'MINIMAL': '#52c41a',
  'LOW': '#1890ff', 
  'MEDIUM': '#faad14',
  'HIGH': '#f5222d'
};

const EvaluationDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedModel, setSelectedModel] = useState('all');
  const [timeRange, setTimeRange] = useState('7d');
  const [textInput, setTextInput] = useState('');
  const [evaluating, setEvaluating] = useState(false);
  const [evaluationResult, setEvaluationResult] = useState(null);

  const {
    data: evaluationData,
    loading,
    error,
    refreshData
  } = useEvaluationData(selectedModel, timeRange);

  const handleEvaluateText = async () => {
    if (!textInput.trim()) return;

    setEvaluating(true);
    try {
      const result = await evaluationApi.evaluateText({
        text: textInput,
        evaluation_types: ['ethical', 'safety', 'alignment']
      });
      
      setEvaluationResult(result);
    } catch (error) {
      console.error('Evaluation failed:', error);
    } finally {
      setEvaluating(false);
    }
  };

  const renderOverviewTab = () => (
    <Row gutter={[16, 16]}>
      {/* Summary Statistics */}
      <Col span={24}>
        <Row gutter={16}>
          <Col span={6}>
            <Card>
              <Statistic
                title="Total Evaluations"
                value={evaluationData?.summary?.total_evaluations || 0}
                prefix={<ExperimentOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Average Safety Score"
                value={evaluationData?.summary?.avg_safety_score || 0}
                precision={3}
                prefix={<SafetyOutlined />}
                suffix="/ 1.0"
              />
              <Progress
                percent={(evaluationData?.summary?.avg_safety_score || 0) * 100}
                size="small"
                strokeColor={
                  (evaluationData?.summary?.avg_safety_score || 0) > 0.8 ? '#52c41a' :
                  (evaluationData?.summary?.avg_safety_score || 0) > 0.6 ? '#faad14' : '#f5222d'
                }
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Average Ethical Score"
                value={evaluationData?.summary?.avg_ethical_score || 0}
                precision={3}
                prefix={<HeartOutlined />}
                suffix="/ 1.0"
              />
              <Progress
                percent={(evaluationData?.summary?.avg_ethical_score || 0) * 100}
                size="small"
                strokeColor={
                  (evaluationData?.summary?.avg_ethical_score || 0) > 0.8 ? '#52c41a' :
                  (evaluationData?.summary?.avg_ethical_score || 0) > 0.6 ? '#faad14' : '#f5222d'
                }
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Average Alignment Score"
                value={evaluationData?.summary?.avg_alignment_score || 0}
                precision={3}
                prefix={<BarChartOutlined />}
                suffix="/ 1.0"
              />
              <Progress
                percent={(evaluationData?.summary?.avg_alignment_score || 0) * 100}
                size="small"
                strokeColor={
                  (evaluationData?.summary?.avg_alignment_score || 0) > 0.8 ? '#52c41a' :
                  (evaluationData?.summary?.avg_alignment_score || 0) > 0.6 ? '#faad14' : '#f5222d'
                }
              />
            </Card>
          </Col>
        </Row>
      </Col>

      {/* Risk Distribution */}
      <Col span={12}>
        <Card title="Risk Level Distribution">
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={evaluationData?.risk_distribution || []}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {(evaluationData?.risk_distribution || []).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={RISK_COLORS[entry.name]} />
                ))}
              </Pie>
              <RechartsTooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </Col>

      {/* Score Trends */}
      <Col span={12}>
        <Card title="Score Trends Over Time">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={evaluationData?.score_trends || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={[0, 1]} />
              <RechartsTooltip />
              <Line type="monotone" dataKey="safety" stroke="#8884d8" strokeWidth={2} />
              <Line type="monotone" dataKey="ethical" stroke="#82ca9d" strokeWidth={2} />
              <Line type="monotone" dataKey="alignment" stroke="#ffc658" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </Col>

      {/* Recent Evaluations */}
      <Col span={24}>
        <Card 
          title="Recent Evaluations" 
          extra={
            <Button icon={<ReloadOutlined />} onClick={refreshData}>
              Refresh
            </Button>
          }
        >
          <Table
            dataSource={evaluationData?.recent_evaluations || []}
            rowKey="text_id"
            pagination={{ pageSize: 10 }}
            columns={[
              {
                title: 'Text ID',
                dataKey: 'text_id',
                width: 120,
                render: (text) => <Text code>{text.substring(0, 8)}...</Text>
              },
              {
                title: 'Text Preview',
                dataKey: 'text',
                ellipsis: true,
                render: (text) => (
                  <Tooltip title={text}>
                    {text.substring(0, 100)}...
                  </Tooltip>
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
                title: 'Risk Level',
                dataKey: 'risk_level',
                width: 100,
                render: (level) => (
                  <Tag color={RISK_COLORS[level]}>{level}</Tag>
                )
              },
              {
                title: 'Timestamp',
                dataKey: 'timestamp',
                width: 120,
                render: (timestamp) => new Date(timestamp).toLocaleDateString()
              },
              {
                title: 'Actions',
                width: 80,
                render: (_, record) => (
                  <Button 
                    type="link" 
                    icon={<EyeOutlined />}
                    onClick={() => showEvaluationDetails(record)}
                  >
                    View
                  </Button>
                )
              }
            ]}
          />
        </Card>
      </Col>
    </Row>
  );

  const renderEthicalTab = () => (
    <Row gutter={[16, 16]}>
      {/* Ethical Dimensions Radar Chart */}
      <Col span={12}>
        <Card title="Ethical Dimensions Analysis">
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart data={evaluationData?.ethical_analysis?.dimensions || []}>
              <PolarGrid />
              <PolarAngleAxis dataKey="dimension" />
              <PolarRadiusAxis domain={[0, 1]} />
              <Radar
                name="Average Score"
                dataKey="score"
                stroke="#8884d8"
                fill="#8884d8"
                fillOpacity={0.6}
              />
              <RechartsTooltip />
            </RadarChart>
          </ResponsiveContainer>
        </Card>
      </Col>

      {/* Ethical Issues Frequency */}
      <Col span={12}>
        <Card title="Common Ethical Issues">
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={evaluationData?.ethical_analysis?.issues || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="issue" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <RechartsTooltip />
              <Bar dataKey="frequency" fill="#ff7300" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </Col>

      {/* Ethical Recommendations */}
      <Col span={24}>
        <Card title="Ethical Improvement Recommendations">
          {evaluationData?.ethical_analysis?.recommendations?.map((rec, index) => (
            <Alert
              key={index}
              message={rec.title}
              description={rec.description}
              type={rec.severity === 'high' ? 'error' : rec.severity === 'medium' ? 'warning' : 'info'}
              style={{ marginBottom: 8 }}
            />
          ))}
        </Card>
      </Col>
    </Row>
  );

  const renderSafetyTab = () => (
    <Row gutter={[16, 16]}>
      {/* Safety Categories */}
      <Col span={16}>
        <Card title="Safety Categories Analysis">
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={evaluationData?.safety_analysis?.categories || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" />
              <YAxis />
              <RechartsTooltip />
              <Bar dataKey="violation_count" fill="#ff4d4f" />
              <Bar dataKey="avg_severity" fill="#faad14" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </Col>

      {/* Safety Score Distribution */}
      <Col span={8}>
        <Card title="Safety Score Distribution">
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie
                data={evaluationData?.safety_analysis?.score_distribution || []}
                cx="50%"
                cy="50%"
                outerRadius={80}
                fill="#8884d8"
                dataKey="count"
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              >
                {(evaluationData?.safety_analysis?.score_distribution || []).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <RechartsTooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </Col>

      {/* Recent Safety Violations */}
      <Col span={24}>
        <Card title="Recent Safety Violations">
          <Table
            dataSource={evaluationData?.safety_analysis?.recent_violations || []}
            rowKey="id"
            pagination={{ pageSize: 5 }}
            columns={[
              {
                title: 'Text ID',
                dataKey: 'text_id',
                width: 120,
                render: (text) => <Text code>{text.substring(0, 8)}...</Text>
              },
              {
                title: 'Category',
                dataKey: 'category',
                width: 120,
                render: (category) => <Tag color="red">{category}</Tag>
              },
              {
                title: 'Violation',
                dataKey: 'violation_description',
                ellipsis: true
              },
              {
                title: 'Severity',
                dataKey: 'severity',
                width: 100,
                render: (severity) => (
                  <Tag color={severity > 0.8 ? 'red' : severity > 0.6 ? 'orange' : 'yellow'}>
                    {(severity || 0).toFixed(2)}
                  </Tag>
                )
              },
              {
                title: 'Confidence',
                dataKey: 'confidence',
                width: 100,
                render: (confidence) => (
                  <Progress
                    percent={(confidence || 0) * 100}
                    size="small"
                    strokeColor={confidence > 0.8 ? '#52c41a' : confidence > 0.6 ? '#faad14' : '#f5222d'}
                  />
                )
              }
            ]}
          />
        </Card>
      </Col>
    </Row>
  );

  const renderTestTab = () => (
    <Row gutter={[16, 16]}>
      <Col span={24}>
        <Card title="Live Text Evaluation">
          <Space direction="vertical" style={{ width: '100%' }}>
            <TextArea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Enter text to evaluate for ethical, safety, and alignment considerations..."
              rows={6}
            />
            
            <Button
              type="primary"
              loading={evaluating}
              onClick={handleEvaluateText}
              disabled={!textInput.trim()}
            >
              Evaluate Text
            </Button>

            {evaluationResult && (
              <Card title="Evaluation Results" style={{ marginTop: 16 }}>
                <Row gutter={16}>
                  <Col span={8}>
                    <Card size="small" title="Safety Evaluation">
                      <Statistic
                        value={evaluationResult.evaluation_results?.safety?.overall_score || 0}
                        precision={3}
                        suffix="/ 1.0"
                      />
                      <Tag color={RISK_COLORS[evaluationResult.evaluation_results?.safety?.risk_level]}>
                        {evaluationResult.evaluation_results?.safety?.risk_level}
                      </Tag>
                    </Card>
                  </Col>
                  
                  <Col span={8}>
                    <Card size="small" title="Ethical Evaluation">
                      <Statistic
                        value={evaluationResult.evaluation_results?.ethical?.overall_score || 0}
                        precision={3}
                        suffix="/ 1.0"
                      />
                    </Card>
                  </Col>
                  
                  <Col span={8}>
                    <Card size="small" title="Alignment Evaluation">
                      <Statistic
                        value={evaluationResult.evaluation_results?.alignment?.overall_score || 0}
                        precision={3}
                        suffix="/ 1.0"
                      />
                    </Card>
                  </Col>
                </Row>

                {/* Detailed Results */}
                {evaluationResult.evaluation_results?.safety?.violations?.length > 0 && (
                  <Alert
                    message="Safety Violations Detected"
                    description={
                      <ul>
                        {evaluationResult.evaluation_results.safety.violations.map((violation, index) => (
                          <li key={index}>
                            <strong>{violation.rule.category}:</strong> {violation.explanation}
                          </li>
                        ))}
                      </ul>
                    }
                    type="error"
                    style={{ marginTop: 16 }}
                  />
                )}

                {evaluationResult.evaluation_results?.alignment?.recommendations?.length > 0 && (
                  <Card title="Alignment Recommendations" style={{ marginTop: 16 }}>
                    <ul>
                      {evaluationResult.evaluation_results.alignment.recommendations.map((rec, index) => (
                        <li key={index}>{rec}</li>
                      ))}
                    </ul>
                  </Card>
                )}
              </Card>
            )}
          </Space>
        </Card>
      </Col>
    </Row>
  );

  const showEvaluationDetails = (record) => {
    // Implementation for showing detailed evaluation results
    console.log('Show details for:', record);
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>Loading evaluation data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert
        message="Error Loading Data"
        description={error.message}
        type="error"
        showIcon
        style={{ margin: '20px' }}
      />
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>Evaluation Dashboard</Title>
        
        <Space>
          <Select
            value={selectedModel}
            onChange={setSelectedModel}
            style={{ width: 150 }}
          >
            <Option value="all">All Models</Option>
            <Option value="gpt-4">GPT-4</Option>
            <Option value="gpt-3.5-turbo">GPT-3.5 Turbo</Option>
            <Option value="claude-3">Claude 3</Option>
          </Select>
          
          <Select
            value={timeRange}
            onChange={setTimeRange}
            style={{ width: 120 }}
          >
            <Option value="1d">Last Day</Option>
            <Option value="7d">Last Week</Option>
            <Option value="30d">Last Month</Option>
            <Option value="90d">Last 3 Months</Option>
          </Select>
        </Space>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="Overview" key="overview">
          {renderOverviewTab()}
        </TabPane>
        
        <TabPane tab="Ethical Analysis" key="ethical">
          {renderEthicalTab()}
        </TabPane>
        
        <TabPane tab="Safety Analysis" key="safety">
          {renderSafetyTab()}
        </TabPane>
        
        <TabPane tab="Live Test" key="test">
          {renderTestTab()}
        </TabPane>
      </Tabs>
    </div>
  );
};

export default EvaluationDashboard;