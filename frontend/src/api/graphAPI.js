import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:1337/api'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', response.status, response.config.url)
    return response.data
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message)

    if (error.code === 'ECONNABORTED') {
      throw new Error('Timeout de la requête')
    }

    if (error.response?.status === 404) {
      throw new Error('Endpoint non trouvé')
    }

    if (error.response?.status === 500) {
      throw new Error('Erreur serveur interne')
    }

    throw error.response?.data || error
  }
)

export const graphAPI = {
  // Get available graphs
  getAvailableGraphs: async () => {
    try {
      const response = await api.get('/graphs')
      return response
    } catch (error) {
      console.error('Error fetching available graphs:', error)
      // Return default graph if API fails
      return [
        { id: 'default', name: 'edge_events_clean', model: 'STRIPE Light' }
      ]
    }
  },

  // Get graph data for visualization
  getGraphData: async (graphId = 'default') => {
    try {
      const response = await api.get(`/graph-data/${graphId}`)
      return response
    } catch (error) {
      console.error('Error fetching graph data:', error)
      throw error
    }
  },

  // Predict anomaly for edge
  predictAnomaly: async (srcNode, dstNode, label = 'DEPENDS_ON', eventType = 'add', graphId = 'default') => {
    try {
      const response = await api.post('/predict', {
        src: srcNode,
        dst: dstNode,
        label: label,
        event_type: eventType,
        graph_id: graphId,
      })
      return response
    } catch (error) {
      console.error('Error predicting anomaly:', error)
      throw error
    }
  },

  // Upload graph file
  uploadGraph: async (file) => {
    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await api.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      return response
    } catch (error) {
      console.error('Error uploading graph:', error)
      throw error
    }
  },

  // Get upload status
  getUploadStatus: async (uploadId) => {
    try {
      const response = await api.get(`/upload-status/${uploadId}`)
      return response
    } catch (error) {
      console.error('Error getting upload status:', error)
      throw error
    }
  }
}
