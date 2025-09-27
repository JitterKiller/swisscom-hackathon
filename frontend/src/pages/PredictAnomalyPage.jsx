import React, { useState, useEffect, useRef } from 'react'
import { Loader2, AlertCircle, CheckCircle, ChevronDown } from 'lucide-react'
import GraphVisualization from '../components/GraphVisualization'
import { graphAPI } from '../api/graphAPI'

const PredictAnomalyPage = () => {
  const [graphData, setGraphData] = useState(null)
  const [availableGraphs, setAvailableGraphs] = useState([])
  const [selectedGraph, setSelectedGraph] = useState('default')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [predictionResult, setPredictionResult] = useState(null)

  // Form state
  const [srcNode, setSrcNode] = useState('')
  const [dstNode, setDstNode] = useState('')
  const [selectedLabel, setSelectedLabel] = useState('DEPENDS_ON')
  const [selectedEventType, setSelectedEventType] = useState('add')

  // Refs for input fields
  const srcInputRef = useRef(null)
  const dstInputRef = useRef(null)

  useEffect(() => {
    loadAvailableGraphs()
    loadGraphData('default') // Load default graph initially
  }, [])

  const loadAvailableGraphs = async () => {
    try {
      const graphs = await graphAPI.getAvailableGraphs()
      setAvailableGraphs(graphs)
    } catch (err) {
      console.error('Error loading available graphs:', err)
    }
  }

  const loadGraphData = async (graphId = selectedGraph) => {
    try {
      setLoading(true)
      console.log('PredictAnomalyPage: Loading graph data for', graphId)
      const data = await graphAPI.getGraphData(graphId)
      console.log('PredictAnomalyPage: Graph data loaded', data)
      setGraphData(data)
    } catch (err) {
      setError('Erreur lors du chargement des données du graphe')
      console.error('Error loading graph data:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleGraphChange = (graphId) => {
    console.log('PredictAnomalyPage: Graph changed to', graphId)
    setSelectedGraph(graphId)
    loadGraphData(graphId) // Pass the graphId directly
  }

  const handlePredictAnomaly = async () => {
    if (!srcNode || !dstNode) {
      setError('Veuillez entrer les nœuds source et destination')
      return
    }

    try {
      setPredictionLoading(true)
      setError(null)
      const result = await graphAPI.predictAnomaly(srcNode, dstNode, selectedLabel, selectedEventType, selectedGraph)
      setPredictionResult(result)
    } catch (err) {
      setError('Erreur lors de la prédiction')
      console.error('Error predicting anomaly:', err)
    } finally {
      setPredictionLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handlePredictAnomaly()
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex items-center gap-3">
          <Loader2 className="w-6 h-6 animate-spin" />
          <span>Chargement du graphe...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Prédiction d'Anomalies d'Arêtes
        </h1>
        <p className="text-gray-600">
          Analysez les connexions entre les nœuds pour détecter les comportements anormaux
        </p>
      </div>

      {/* Graph Visualization */}
      <div className="mb-8">
        <div className="card">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Visualisation du Graphe</h2>
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Graphe:</span>
              <div className="relative">
                <select
                  value={selectedGraph}
                  onChange={(e) => handleGraphChange(e.target.value)}
                  className="appearance-none bg-white border border-gray-300 rounded-lg px-3 py-2 pr-8 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {availableGraphs.map((graph) => (
                    <option key={graph.id} value={graph.id}>
                      {graph.name} - {graph.model}
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-2 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none" />
              </div>
            </div>
          </div>
          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <span className="text-red-700">{error}</span>
            </div>
          )}
          <div className="h-[500px] bg-gray-50 rounded-lg border-2 border-dashed border-gray-300 overflow-hidden relative">
            {graphData ? (
              <GraphVisualization key={selectedGraph} data={graphData} />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                Aucun graphe à afficher
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Prediction Form */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Section */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Prédiction d'Anomalie</h2>

          <div className="space-y-4">
            <div>
              <label htmlFor="srcNode" className="block text-sm font-medium text-gray-700 mb-2">
                Nœud Source
              </label>
              <input
                ref={srcInputRef}
                id="srcNode"
                type="text"
                value={srcNode}
                onChange={(e) => setSrcNode(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ex: node-123"
                className="input-field"
              />
            </div>

            <div>
              <label htmlFor="dstNode" className="block text-sm font-medium text-gray-700 mb-2">
                Nœud Destination
              </label>
              <input
                ref={dstInputRef}
                id="dstNode"
                type="text"
                value={dstNode}
                onChange={(e) => setDstNode(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ex: node-456"
                className="input-field"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="label" className="block text-sm font-medium text-gray-700 mb-2">
                  Label
                </label>
                <div className="relative">
                  <select
                    id="label"
                    value={selectedLabel}
                    onChange={(e) => setSelectedLabel(e.target.value)}
                    className="appearance-none bg-white border border-gray-300 rounded-lg px-3 py-2 pr-8 w-full focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="HAS_PORT">HAS_PORT</option>
                    <option value="DEPENDS_ON">DEPENDS_ON</option>
                    <option value="INSTALLED_AT">INSTALLED_AT</option>
                    <option value="HAS">HAS</option>
                    <option value="REFERS_TO">REFERS_TO</option>
                  </select>
                  <ChevronDown className="absolute right-2 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none" />
                </div>
              </div>

              <div>
                <label htmlFor="eventType" className="block text-sm font-medium text-gray-700 mb-2">
                  Event Type
                </label>
                <div className="relative">
                  <select
                    id="eventType"
                    value={selectedEventType}
                    onChange={(e) => setSelectedEventType(e.target.value)}
                    className="appearance-none bg-white border border-gray-300 rounded-lg px-3 py-2 pr-8 w-full focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="add">Add</option>
                    <option value="delete">Delete</option>
                  </select>
                  <ChevronDown className="absolute right-2 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none" />
                </div>
              </div>
            </div>

            <button
              onClick={handlePredictAnomaly}
              disabled={predictionLoading || !srcNode || !dstNode}
              className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {predictionLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyse en cours...
                </>
              ) : (
                'Prédire l\'Anomalie'
              )}
            </button>
          </div>
        </div>

        {/* Results Section */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Résultats de la Prédiction</h2>

          {predictionResult ? (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                {predictionResult.is_anomaly ? (
                  <>
                    <AlertCircle className="w-5 h-5 text-red-500" />
                    <span className="text-red-700 font-medium">Anomalie Détectée</span>
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <span className="text-green-700 font-medium">Connexion Normale</span>
                  </>
                )}
              </div>

              <div className="bg-gray-50 rounded-lg p-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium text-gray-700">Score de Confiance:</span>
                    <div className="text-lg font-bold text-blue-600">
                      {(predictionResult.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Arête:</span>
                    <div className="font-mono text-sm bg-white px-2 py-1 rounded border">
                      {srcNode} → {dstNode}
                    </div>
                  </div>
                </div>
              </div>

              {predictionResult.details && (
                <div>
                  <h4 className="font-medium text-gray-700 mb-2">Détails de l'Analyse</h4>
                  <pre className="text-xs bg-gray-100 p-3 rounded overflow-x-auto">
                    {JSON.stringify(predictionResult.details, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              <AlertCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>Entrez les nœuds source et destination pour voir les résultats</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default PredictAnomalyPage
