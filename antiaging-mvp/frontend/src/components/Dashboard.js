#PLACEHOLDER CODE #1

import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Brain, 
  Calendar,
  Info,
  AlertCircle,
  CheckCircle
} from 'lucide-react';
import { dashboardAPI, predictionsAPI } from '../services/api';
import toast from 'react-hot-toast';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const Dashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [selectedSHAPFeature, setSelectedSHAPFeature] = useState(null);

  useEffect(() => {
    loadDashboardData();
    loadPredictions();
  }, []);

  const loadDashboardData = async () => {
    try {
      const response = await dashboardAPI.getDashboardData();
      setDashboardData(response.data);
    } catch (error) {
      console.error('Error loading dashboard data:', error);
      toast.error('Failed to load dashboard data');
    }
  };

  const loadPredictions = async () => {
    try {
      const response = await predictionsAPI.getHistory();
      setPredictions(response.data.slice(0, 10)); // Last 10 predictions
    } catch (error) {
      console.error('Error loading predictions:', error);
    } finally {
      setLoading(false);
    }
  };

  const generatePrediction = async () => {
    setPredicting(true);
    try {
      const response = await predictionsAPI.predict();
      toast.success('New aging prediction generated!');
      
      // Reload data
      loadDashboardData();
      loadPredictions();
    } catch (error) {
      toast.error('Failed to generate prediction. Please ensure you have uploaded genetic data and habits.');
    } finally {
      setPredicting(false);
    }
  };

  const getAgingTrendData = () => {
    if (!predictions.length) return null;

    const sortedPredictions = [...predictions].sort(
      (a, b) => new Date(a.created_at) - new Date(b.created_at)
    );

    return {
      labels: sortedPredictions.map((_, index) => `Prediction ${index + 1}`),
      datasets: [
        {
          label: 'Biological Age',
          data: sortedPredictions.map(p => p.biological_age),
          borderColor: 'rgb(99, 102, 241)',
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
          tension: 0.4,
        },
        {
          label: 'Chronological Age',
          data: sortedPredictions.map(p => p.chronological_age),
          borderColor: 'rgb(156, 163, 175)',
          backgroundColor: 'rgba(156, 163, 175, 0.1)',
          borderDash: [5, 5],
          tension: 0.4,
        },
      ],
    };
  };

  const getSHAPData = () => {
    if (!dashboardData?.latest_prediction?.shap_values) return null;

    const shapValues = dashboardData.latest_prediction.shap_values;
    const features = Object.keys(shapValues);
    const values = Object.values(shapValues);

    return {
      labels: features.map(f => f.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())),
      datasets: [
        {
          label: 'SHAP Values',
          data: values,
          backgroundColor: values.map(v => v > 0 ? 'rgba(239, 68, 68, 0.8)' : 'rgba(34, 197, 94, 0.8)'),
          borderColor: values.map(v => v > 0 ? 'rgb(239, 68, 68)' : 'rgb(34, 197, 94)'),
          borderWidth: 1,
        },
      ],
    };
  };

  const getHealthScoreData = () => {
    if (!dashboardData?.latest_prediction) return null;

    const prediction = dashboardData.latest_prediction;
    const agingDiff = prediction.biological_age - prediction.chronological_age;
    
    // Convert aging difference to health score (0-100)
    const healthScore = Math.max(0, Math.min(100, 75 - (agingDiff * 5)));

    return {
      labels: ['Health Score', 'Remaining'],
      datasets: [
        {
          data: [healthScore, 100 - healthScore],
          backgroundColor: [
            healthScore >= 70 ? '#10B981' : healthScore >= 50 ? '#F59E0B' : '#EF4444',
            '#E5E7EB'
          ],
          borderWidth: 0,
        },
      ],
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: false,
      },
    },
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const value = context.parsed.y;
            const impact = value > 0 ? 'increases' : 'decreases';
            return `${context.label}: ${impact} aging by ${Math.abs(value).toFixed(3)}`;
          }
        }
      }
    },
    scales: {
      x: {
        ticks: {
          maxRotation: 45,
        },
      },
      y: {
        title: {
          display: true,
          text: 'Impact on Aging',
        },
      },
    },
    onClick: (event, elements) => {
      if (elements.length > 0) {
        const index = elements[0].index;
        const shapData = getSHAPData();
        if (shapData) {
          setSelectedSHAPFeature({
            feature: shapData.labels[index],
            value: shapData.datasets[0].data[index],
          });
        }
      }
    },
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
    },
    cutout: '70%',
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  const latestPrediction = dashboardData?.latest_prediction;
  const agingDiff = latestPrediction ? 
    latestPrediction.biological_age - latestPrediction.chronological_age : 0;

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Aging Analytics Dashboard</h1>
          <p className="mt-2 text-gray-600">
            Track your biological age and get insights into your aging process
          </p>
        </div>

        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Calendar className="h-8 w-8 text-indigo-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Latest Prediction</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {latestPrediction ? 
                    `${latestPrediction.biological_age} years` : 
                    'No predictions'
                  }
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                {agingDiff > 0 ? (
                  <TrendingUp className="h-8 w-8 text-red-500" />
                ) : (
                  <TrendingDown className="h-8 w-8 text-green-500" />
                )}
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Age Difference</p>
                <p className={`text-2xl font-semibold ${agingDiff > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {agingDiff > 0 ? '+' : ''}{agingDiff.toFixed(1)} years
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Activity className="h-8 w-8 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Confidence</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {latestPrediction ? 
                    `${(latestPrediction.confidence_score * 100).toFixed(1)}%` : 
                    'N/A'
                  }
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Brain className="h-8 w-8 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total Predictions</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {predictions.length}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="mb-8">
          <button
            onClick={generatePrediction}
            disabled={predicting || !dashboardData?.profile_complete}
            className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {predicting ? (
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>Generating Prediction...</span>
              </div>
            ) : (
              'Generate New Prediction'
            )}
          </button>
          
          {!dashboardData?.profile_complete && (
            <p className="mt-2 text-sm text-amber-600 flex items-center">
              <AlertCircle className="h-4 w-4 mr-1" />
              Complete your profile and upload genetic data to generate predictions
            </p>
          )}
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Aging Trend Chart */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Aging Trend</h3>
            <div className="h-64">
              {getAgingTrendData() ? (
                <Line data={getAgingTrendData()} options={chartOptions} />
              ) : (
                <div className="flex items-center justify-center h-full text-gray-500">
                  No prediction history available
                </div>
              )}
            </div>
          </div>

          {/* Health Score */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Health Score</h3>
            <div className="h-64 flex items-center justify-center">
              {getHealthScoreData() ? (
                <div className="relative">
                  <Doughnut data={getHealthScoreData()} options={doughnutOptions} />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-gray-900">
                        {Math.round(getHealthScoreData().datasets[0].data[0])}
                      </div>
                      <div className="text-sm text-gray-500">Health Score</div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-gray-500">No health data available</div>
              )}
            </div>
          </div>
        </div>

        {/* SHAP Values Chart */}
        {getSHAPData() && (
          <div className="bg-white rounded-lg shadow p-6 mb-8">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">Feature Impact Analysis</h3>
              <div className="flex items-center text-sm text-gray-500">
                <Info className="h-4 w-4 mr-1" />
                Click bars to see details
              </div>
            </div>
            <div className="h-64">
              <Bar data={getSHAPData()} options={barChartOptions} />
            </div>
            <div className="mt-4 text-xs text-gray-600">
              <p>Green bars indicate factors that decrease your biological age (good for longevity)</p>
              <p>Red bars indicate factors that increase your biological age (accelerate aging)</p>
            </div>
          </div>
        )}

        {/* SHAP Feature Detail Modal */}
        {selectedSHAPFeature && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900">Feature Impact</h3>
                <button
                  onClick={() => setSelectedSHAPFeature(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ×
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-gray-900">{selectedSHAPFeature.feature}</h4>
                  <p className={`text-lg font-semibold ${
                    selectedSHAPFeature.value > 0 ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {selectedSHAPFeature.value > 0 ? 'Increases' : 'Decreases'} aging by{' '}
                    {Math.abs(selectedSHAPFeature.value).toFixed(3)}
                  </p>
                </div>
                
                <div className="bg-blue-50 border border-blue-200 rounded p-4">
                  <p className="text-sm text-blue-800">
                    This feature {selectedSHAPFeature.value > 0 ? 'negatively' : 'positively'} impacts 
                    your biological age prediction. Consider focusing on improving this aspect of your 
                    health for better aging outcomes.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Profile Status */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Profile Status</h3>
          <div className="space-y-3">
            <div className="flex items-center">
              {dashboardData?.profile_complete ? (
                <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
              ) : (
                <AlertCircle className="h-5 w-5 text-amber-500 mr-2" />
              )}
              <span className="text-sm text-gray-700">
                Profile completion: {dashboardData?.profile_complete ? 'Complete' : 'Incomplete'}
              </span>
            </div>
            
            <div className="flex items-center">
              <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
              <span className="text-sm text-gray-700">
                Habits recorded: {dashboardData?.habits_count || 0} entries
              </span>
            </div>
            
            <div className="flex items-center">
              {latestPrediction ? (
                <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
              ) : (
                <AlertCircle className="h-5 w-5 text-amber-500 mr-2" />
              )}
              <span className="text-sm text-gray-700">
                Latest prediction: {latestPrediction ? 
                  new Date(latestPrediction.created_at).toLocaleDateString() : 
                  'None'
                }
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

#PLACEHOLDER CODE #2

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Bar } from 'react-chartjs-2';
import 'chart.js/auto';

const Dashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        const res = await axios.get('/api/predict/');
        setData(res.data);
      } catch (err) {
        console.error('Fetch failed', err);
      } finally {
        setLoading(false);
      }
    };
    fetchPrediction();
  }, []);

  if (loading) return <p>Loading...</p>;
  if (!data) return <p>No data</p>;

  const chartData = {
    labels: Object.keys(data.explanations),
    datasets: [{ label: 'SHAP Contribution', data: Object.values(data.explanations), backgroundColor: 'rgba(75,192,192,0.6)' }]
  };

  return (
    <div>
      <h2>Prediction: {data.prediction}</h2>
      <Bar data={chartData} />
      {/* Add tooltips for explanations */}
      <p>Disclaimer: This is not medical advice.</p>
    </div>
  );
};

export default Dashboard;