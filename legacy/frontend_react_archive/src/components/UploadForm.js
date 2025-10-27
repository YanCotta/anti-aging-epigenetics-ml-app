#PLACEHOLDER CODE #1

import React, { useState, useCallback } from 'react';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import { Upload, FileText, CheckCircle, AlertCircle, X } from 'lucide-react';
import { uploadFile, geneticAPI } from '../services/api';
import toast from 'react-hot-toast';

const uploadSchema = yup.object({
  genetic_file: yup
    .mixed()
    .required('Please select a file')
    .test('fileSize', 'File size must be less than 10MB', (value) => {
      return value && value[0] && value[0].size <= 10 * 1024 * 1024;
    })
    .test('fileType', 'Only CSV, TXT, and VCF files are allowed', (value) => {
      if (!value || !value[0]) return false;
      const allowedTypes = ['.csv', '.txt', '.vcf'];
      const fileName = value[0].name.toLowerCase();
      return allowedTypes.some(type => fileName.endsWith(type));
    }),
});

const UploadForm = ({ onUploadSuccess }) => {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
    watch,
    reset
  } = useForm({
    resolver: yupResolver(uploadSchema)
  });

  const watchedFile = watch('genetic_file');

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setValue('genetic_file', e.dataTransfer.files);
    }
  }, [setValue]);

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setValue('genetic_file', e.target.files);
    }
  };

  const onSubmit = async (data) => {
    if (!selectedFile) {
      toast.error('Please select a file to upload');
      return;
    }

    try {
      setUploading(true);
      setUploadProgress(0);

      const response = await uploadFile(selectedFile, setUploadProgress);
      
      setUploadComplete(true);
      toast.success('Genetic data uploaded successfully!');
      
      if (onUploadSuccess) {
        onUploadSuccess(response.data);
      }

      // Reset form after successful upload
      setTimeout(() => {
        reset();
        setSelectedFile(null);
        setUploadComplete(false);
        setUploadProgress(0);
      }, 2000);

    } catch (error) {
      toast.error('Upload failed. Please try again.');
      console.error('Upload error:', error);
    } finally {
      setUploading(false);
    }
  };

  const removeFile = () => {
    setSelectedFile(null);
    setValue('genetic_file', null);
    setUploadProgress(0);
    setUploadComplete(false);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-md p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Genetic Data</h2>
        <p className="text-gray-600">
          Upload your genetic or epigenetic data file. Supported formats: CSV, TXT, VCF (max 10MB)
        </p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        {/* File Upload Area */}
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive
              ? 'border-indigo-500 bg-indigo-50'
              : uploadComplete
              ? 'border-green-500 bg-green-50'
              : errors.genetic_file
              ? 'border-red-500 bg-red-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            {...register('genetic_file')}
            onChange={handleFileSelect}
            accept=".csv,.txt,.vcf"
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            disabled={uploading}
          />

          {selectedFile ? (
            <div className="space-y-4">
              <div className="flex items-center justify-center space-x-2">
                {uploadComplete ? (
                  <CheckCircle className="h-8 w-8 text-green-500" />
                ) : (
                  <FileText className="h-8 w-8 text-gray-400" />
                )}
              </div>
              
              <div className="bg-white rounded-lg p-4 border">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <p className="font-medium text-gray-900 truncate">
                      {selectedFile.name}
                    </p>
                    <p className="text-sm text-gray-500">
                      {formatFileSize(selectedFile.size)}
                    </p>
                  </div>
                  {!uploading && !uploadComplete && (
                    <button
                      type="button"
                      onClick={removeFile}
                      className="ml-2 p-1 text-gray-400 hover:text-gray-600"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  )}
                </div>

                {uploading && (
                  <div className="mt-3">
                    <div className="flex justify-between text-sm text-gray-600 mb-1">
                      <span>Uploading...</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  </div>
                )}

                {uploadComplete && (
                  <div className="mt-3 flex items-center text-green-600 text-sm">
                    <CheckCircle className="h-4 w-4 mr-1" />
                    Upload complete!
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex justify-center">
                <Upload className="h-12 w-12 text-gray-400" />
              </div>
              <div>
                <p className="text-lg font-medium text-gray-900">
                  Drop your file here or click to browse
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  CSV, TXT, VCF files up to 10MB
                </p>
              </div>
            </div>
          )}
        </div>

        {errors.genetic_file && (
          <div className="flex items-center space-x-2 text-red-600 text-sm">
            <AlertCircle className="h-4 w-4" />
            <span>{errors.genetic_file.message}</span>
          </div>
        )}

        {/* Upload Button */}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={!selectedFile || uploading || uploadComplete}
            className="px-6 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {uploading ? (
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>Uploading...</span>
              </div>
            ) : uploadComplete ? (
              'Upload Complete'
            ) : (
              'Upload File'
            )}
          </button>
        </div>

        {/* File Format Information */}
        <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
          <h3 className="font-medium text-blue-900 mb-2">Supported File Formats:</h3>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• <strong>CSV:</strong> Comma-separated values with genetic markers</li>
            <li>• <strong>TXT:</strong> Tab-delimited or space-separated genetic data</li>
            <li>• <strong>VCF:</strong> Variant Call Format files</li>
          </ul>
          <p className="text-xs text-blue-700 mt-2">
            Note: Your data is processed securely and used only for aging predictions.
          </p>
        </div>
      </form>
    </div>
  );
};

export default UploadForm;

#PLACEHOLDER CODE #2

import React, { useState } from 'react';
import axios from 'axios';

const UploadForm = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = e => setFile(e.target.files[0]);

  const handleSubmit = async e => {
    e.preventDefault();
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      await axios.post('/api/upload-genetic/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      alert('Upload successful');
    } catch (err) {
      setError(err.response?.data?.error || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="file" onChange={handleFileChange} accept=".csv" />
      <button type="submit" disabled={loading}>Upload Genetic Data</button>
      {error && <p>{error}</p>}
      {loading && <p>Loading...</p>}
    </form>
  );
};

export default UploadForm;