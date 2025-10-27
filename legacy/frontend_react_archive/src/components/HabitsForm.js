#PLACEHOLDER CODE #1

import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import { Save, Activity, Moon, Brain, Apple, Cigarette, Wine } from 'lucide-react';
import { habitsAPI } from '../services/api';
import toast from 'react-hot-toast';

const habitsSchema = yup.object({
  exercise_frequency: yup
    .number()
    .required('Exercise frequency is required')
    .min(0, 'Cannot be negative')
    .max(7, 'Cannot exceed 7 days'),
  sleep_hours: yup
    .number()
    .required('Sleep hours is required')
    .min(3, 'Minimum 3 hours')
    .max(12, 'Maximum 12 hours'),
  stress_level: yup
    .number()
    .required('Stress level is required')
    .min(1, 'Minimum level is 1')
    .max(10, 'Maximum level is 10'),
  diet_quality: yup
    .number()
    .required('Diet quality is required')
    .min(1, 'Minimum rating is 1')
    .max(10, 'Maximum rating is 10'),
  smoking: yup.boolean(),
  alcohol_consumption: yup
    .number()
    .required('Alcohol consumption is required')
    .min(0, 'Cannot be negative')
    .max(50, 'Maximum 50 drinks per week'),
});

const HabitsForm = ({ onSaveSuccess }) => {
  const [loading, setLoading] = useState(false);
  const [existingHabits, setExistingHabits] = useState(null);

  const {
    register,
    handleSubmit,
    formState: { errors, isDirty },
    setValue,
    watch,
    reset
  } = useForm({
    resolver: yupResolver(habitsSchema),
    defaultValues: {
      exercise_frequency: 3,
      sleep_hours: 7.5,
      stress_level: 5,
      diet_quality: 5,
      smoking: false,
      alcohol_consumption: 0,
    }
  });

  const watchedValues = watch();

  useEffect(() => {
    loadExistingHabits();
  }, []);

  const loadExistingHabits = async () => {
    try {
      const response = await habitsAPI.getHabits();
      if (response.data.length > 0) {
        const latest = response.data[0]; // Get most recent habits
        setExistingHabits(latest);
        
        // Populate form with existing data
        Object.keys(latest).forEach(key => {
          if (key !== 'id' && key !== 'user' && key !== 'recorded_date') {
            setValue(key, latest[key]);
          }
        });
      }
    } catch (error) {
      console.error('Error loading habits:', error);
    }
  };

  const onSubmit = async (data) => {
    setLoading(true);
    try {
      const response = await habitsAPI.createHabits(data);
      toast.success('Habits saved successfully!');
      
      if (onSaveSuccess) {
        onSaveSuccess(response.data);
      }
      
      setExistingHabits(response.data);
    } catch (error) {
      toast.error('Failed to save habits. Please try again.');
      console.error('Save habits error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStressLevelColor = (level) => {
    if (level <= 3) return 'text-green-600';
    if (level <= 6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getDietQualityColor = (quality) => {
    if (quality >= 8) return 'text-green-600';
    if (quality >= 6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Lifestyle Habits</h2>
        <p className="text-gray-600">
          Track your daily habits to get personalized aging predictions and recommendations.
        </p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-8">
        {/* Exercise */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-indigo-600" />
            <h3 className="text-lg font-medium text-gray-900">Exercise</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Exercise Frequency (days per week)
              </label>
              <input
                type="number"
                {...register('exercise_frequency')}
                min="0"
                max="7"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
              {errors.exercise_frequency && (
                <p className="mt-1 text-sm text-red-600">{errors.exercise_frequency.message}</p>
              )}
            </div>

            <div className="flex items-center justify-center">
              <div className="text-center">
                <div className="text-3xl font-bold text-indigo-600">
                  {watchedValues.exercise_frequency || 0}
                </div>
                <div className="text-sm text-gray-500">days/week</div>
              </div>
            </div>
          </div>
        </div>

        {/* Sleep */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <Moon className="h-5 w-5 text-indigo-600" />
            <h3 className="text-lg font-medium text-gray-900">Sleep</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Average Sleep Hours per Night
              </label>
              <input
                type="number"
                step="0.5"
                {...register('sleep_hours')}
                min="3"
                max="12"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
              {errors.sleep_hours && (
                <p className="mt-1 text-sm text-red-600">{errors.sleep_hours.message}</p>
              )}
            </div>

            <div className="flex items-center justify-center">
              <div className="text-center">
                <div className="text-3xl font-bold text-indigo-600">
                  {watchedValues.sleep_hours || 0}
                </div>
                <div className="text-sm text-gray-500">hours/night</div>
              </div>
            </div>
          </div>
        </div>

        {/* Stress Level */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <Brain className="h-5 w-5 text-indigo-600" />
            <h3 className="text-lg font-medium text-gray-900">Stress Level</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Stress Level (1-10, where 10 is very stressed)
              </label>
              <input
                type="range"
                {...register('stress_level')}
                min="1"
                max="10"
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Low (1)</span>
                <span>High (10)</span>
              </div>
              {errors.stress_level && (
                <p className="mt-1 text-sm text-red-600">{errors.stress_level.message}</p>
              )}
            </div>

            <div className="flex items-center justify-center">
              <div className="text-center">
                <div className={`text-3xl font-bold ${getStressLevelColor(watchedValues.stress_level)}`}>
                  {watchedValues.stress_level || 1}
                </div>
                <div className="text-sm text-gray-500">stress level</div>
              </div>
            </div>
          </div>
        </div>

        {/* Diet Quality */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <Apple className="h-5 w-5 text-indigo-600" />
            <h3 className="text-lg font-medium text-gray-900">Diet Quality</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Diet Quality (1-10, where 10 is excellent)
              </label>
              <input
                type="range"
                {...register('diet_quality')}
                min="1"
                max="10"
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Poor (1)</span>
                <span>Excellent (10)</span>
              </div>
              {errors.diet_quality && (
                <p className="mt-1 text-sm text-red-600">{errors.diet_quality.message}</p>
              )}
            </div>

            <div className="flex items-center justify-center">
              <div className="text-center">
                <div className={`text-3xl font-bold ${getDietQualityColor(watchedValues.diet_quality)}`}>
                  {watchedValues.diet_quality || 1}
                </div>
                <div className="text-sm text-gray-500">diet rating</div>
              </div>
            </div>
          </div>
        </div>

        {/* Smoking and Alcohol */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Smoking */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Cigarette className="h-5 w-5 text-indigo-600" />
              <h3 className="text-lg font-medium text-gray-900">Smoking</h3>
            </div>
            
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                {...register('smoking')}
                className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
              />
              <label className="text-sm text-gray-700">
                I smoke regularly
              </label>
            </div>
          </div>

          {/* Alcohol */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Wine className="h-5 w-5 text-indigo-600" />
              <h3 className="text-lg font-medium text-gray-900">Alcohol Consumption</h3>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Drinks per week
              </label>
              <input
                type="number"
                {...register('alcohol_consumption')}
                min="0"
                max="50"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
              {errors.alcohol_consumption && (
                <p className="mt-1 text-sm text-red-600">{errors.alcohol_consumption.message}</p>
              )}
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-end space-x-4">
          {existingHabits && (
            <div className="flex items-center text-sm text-gray-500">
              Last updated: {new Date(existingHabits.recorded_date).toLocaleDateString()}
            </div>
          )}
          
          <button
            type="submit"
            disabled={loading || !isDirty}
            className="flex items-center space-x-2 px-6 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
            ) : (
              <Save className="h-4 w-4" />
            )}
            <span>{loading ? 'Saving...' : 'Save Habits'}</span>
          </button>
        </div>
      </form>
    </div>
  );
};

export default HabitsForm;

#PLACEHOLDER CODE #2

import React from 'react';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import { Tabs, Tab } from 'react-bootstrap';
import axios from 'axios';

const schema = yup.object({
  exercises_per_week: yup.number().min(0).max(7).required(),
  daily_calories: yup.number().min(1000).max(5000).required(),
  // Add others similarly
}).required();

const HabitsForm = () => {
  const { register, handleSubmit, formState: { errors } } = useForm({ resolver: yupResolver(schema) });

  const onSubmit = async data => {
    try {
      await axios.post('/api/submit-habits/', data);
      alert('Habits submitted');
    } catch (err) {
      console.error('Submission failed', err);
    }
  };

  return (
    <Tabs defaultActiveKey="nutrition" id="habits-tabs">
      <Tab eventKey="nutrition" title="Nutrition">
        <form onSubmit={handleSubmit(onSubmit)}>
          <input {...register('daily_calories')} placeholder="Daily Calories" />
          <p>{errors.daily_calories?.message}</p>
          {/* Add more fields */}
          <button type="submit">Submit</button>
        </form>
      </Tab>
      {/* Add tabs for exercise, sleep, etc. */}
    </Tabs>
  );
};

export default HabitsForm;