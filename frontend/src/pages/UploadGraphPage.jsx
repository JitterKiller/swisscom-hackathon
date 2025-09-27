import React, { useState, useCallback } from 'react'
import { Upload, FileText, AlertCircle, CheckCircle, Loader2 } from 'lucide-react'
import { graphAPI } from '../api/graphAPI'

const UploadGraphPage = () => {
  const [isDragOver, setIsDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const [error, setError] = useState(null)

  const allowedTypes = [
    'text/csv',
    '.csv'
  ]

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const validateFile = (file) => {
    if (!file) return false

    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('Le fichier est trop volumineux (max 10MB)')
      return false
    }

    // Check file type
    if (!allowedTypes.some(type =>
      file.type === type ||
      file.name.toLowerCase().endsWith(type.replace('.', ''))
    )) {
      setError('Type de fichier non supporté. Utilisez uniquement CSV')
      return false
    }

    return true
  }

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileUpload(files[0])
    }
  }, [])

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFileUpload(file)
    }
  }

  const handleFileUpload = async (file) => {
    if (!validateFile(file)) return

    try {
      setUploading(true)
      setError(null)
      setUploadResult(null)

      const result = await graphAPI.uploadGraph(file)
      setUploadResult(result)
    } catch (err) {
      setError(err.message || 'Erreur lors de l\'upload')
      console.error('Upload error:', err)
    } finally {
      setUploading(false)
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Upload de Graphe
        </h1>
        <p className="text-gray-600">
          Importez vos données de graphe pour l'analyse d'anomalies
        </p>
      </div>

      {/* Upload Area */}
      <div className="card mb-8">
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            isDragOver
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {uploading ? (
            <div className="flex flex-col items-center">
              <Loader2 className="w-12 h-12 animate-spin text-blue-500 mb-4" />
              <p className="text-lg font-medium text-gray-700 mb-2">
                Upload en cours...
              </p>
              <p className="text-gray-500">
                Veuillez patienter pendant le traitement du fichier
              </p>
            </div>
          ) : (
            <>
              <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-lg font-medium text-gray-700 mb-2">
                Glissez-déposez votre fichier ici
              </p>
              <p className="text-gray-500 mb-4">
                Ou cliquez pour sélectionner un fichier
              </p>

              <input
                type="file"
                id="file-upload"
                className="hidden"
                accept=".csv"
                onChange={handleFileSelect}
              />

              <label
                htmlFor="file-upload"
                className="btn-primary cursor-pointer inline-flex items-center gap-2"
              >
                <FileText className="w-4 h-4" />
                Sélectionner un fichier
              </label>

              <div className="mt-4 text-sm text-gray-500">
                <p>Formats supportés: CSV uniquement</p>
                <p>Taille maximale: 10 MB</p>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
          <div>
            <p className="font-medium text-red-800">Erreur d'upload</p>
            <p className="text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Upload Result */}
      {uploadResult && (
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <CheckCircle className="w-6 h-6 text-green-500" />
            <h2 className="text-xl font-semibold">Upload réussi</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium text-gray-700 mb-2">Informations du fichier</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Nom:</span>
                  <span className="font-mono">{uploadResult.filename}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Taille:</span>
                  <span>{formatFileSize(uploadResult.file_size)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Type:</span>
                  <span>{uploadResult.file_type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Upload ID:</span>
                  <span className="font-mono text-xs">{uploadResult.upload_id}</span>
                </div>
              </div>
            </div>

            <div>
              <h3 className="font-medium text-gray-700 mb-2">Statistiques du graphe</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Nœuds:</span>
                  <span className="font-semibold">{uploadResult.stats?.nodes || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Arêtes:</span>
                  <span className="font-semibold">{uploadResult.stats?.edges || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Densité:</span>
                  <span>{uploadResult.stats?.density?.toFixed(4) || 'N/A'}</span>
                </div>
              </div>
            </div>
          </div>

          {uploadResult.preview && uploadResult.file_type === 'csv' && (
            <div className="mt-6">
              <h3 className="font-medium text-gray-700 mb-2">Aperçu des données (CSV)</h3>
              <div className="bg-gray-50 rounded-lg p-4 max-h-40 overflow-y-auto">
                <pre className="text-xs text-gray-700 whitespace-pre-wrap">
                  {uploadResult.preview.map((row, index) =>
                    `${index + 1}: ${Object.entries(row).map(([k, v]) => `${k}=${v}`).join(', ')}\n`
                  ).join('')}
                </pre>
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  )
}

export default UploadGraphPage
