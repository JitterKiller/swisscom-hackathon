import React, { useState, useEffect, useRef } from 'react'
import { Play, Pause, RotateCcw, CheckCircle, XCircle } from 'lucide-react'

const DatastreamPage = () => {
  const [initialData, setInitialData] = useState([])
  const [testData, setTestData] = useState([])
  const [displayedEvents, setDisplayedEvents] = useState([])
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(true)
  const intervalRef = useRef(null)

  // Load initial data and test data
  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      setLoading(true)

      // Load initial data (first 5 rows)
      const initialResponse = await fetch('http://localhost:1337/api/datastream/initial')
      if (initialResponse.ok) {
        const initialData = await initialResponse.json()
        setInitialData(initialData)
        setDisplayedEvents(initialData.slice(-5)) // Show last 5 events
      }

      // Load test data (first 1000 rows)
      const testResponse = await fetch('http://localhost:1337/api/datastream/test')
      if (testResponse.ok) {
        const testData = await testResponse.json()
        setTestData(testData)
      }
    } catch (error) {
      console.error('Error loading data:', error)
    } finally {
      setLoading(false)
    }
  }

  // Auto-play through test data
  useEffect(() => {
    if (isPlaying && currentIndex < testData.length) {
      intervalRef.current = setInterval(async () => {
        const nextIndex = currentIndex + 1
        if (nextIndex > testData.length) {
          setIsPlaying(false)
          return
        }

        // Get prediction for the current event
        try {
          const event = testData[currentIndex]
          const prediction = await getPredictionForEvent(event)
          const updatedEvent = {
            ...event,
            prediction: prediction.prediction,
            score: prediction.score,
            correct: prediction.correct
          }

          setDisplayedEvents(prev => {
            const updated = [...prev, updatedEvent]
            return updated.slice(-20) // Keep only last 20 events
          })
        } catch (error) {
          console.error('Error getting prediction:', error)
        }

        setCurrentIndex(nextIndex)
      }, 1000) // 1 second per event
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [isPlaying, currentIndex, testData.length, testData])

  // Get prediction for a single event
  const getPredictionForEvent = async (event) => {
    try {
      const response = await fetch('http://localhost:1337/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          src: event.source,
          dst: event.destination,
          label: event.label,
          event_type: event.event_type,
          graph_id: 'default'
        })
      })

      if (response.ok) {
        const result = await response.json()
        const prediction = result.prediction || 'NORMAL'
        const score = result.anomaly_score || 0.5
        const isAnomaly = prediction === 'ANOMALY'

        return {
          prediction: prediction,
          score: score,
          correct: isAnomaly === (event.true_label === 'ANOMALY')
        }
      } else {
        console.error('Prediction API error:', response.status)
        return {
          prediction: 'NORMAL',
          score: 0.5,
          correct: false
        }
      }
    } catch (error) {
      console.error('Error calling prediction API:', error)
      return {
        prediction: 'NORMAL',
        score: 0.5,
        correct: false
      }
    }
  }

  const handlePlay = () => {
    if (currentIndex >= testData.length) {
      setCurrentIndex(0)
      setDisplayedEvents(initialData.slice(-5))
    }
    setIsPlaying(true)
  }

  const handlePause = () => {
    setIsPlaying(false)
  }

  const handleReset = () => {
    setIsPlaying(false)
    setCurrentIndex(0)
    setDisplayedEvents(initialData.slice(-5))
    setPredictions([])
  }

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString()
  }

  const getStatusIcon = (correct) => {
    return correct ? (
      <CheckCircle className="w-4 h-4 text-green-500" />
    ) : (
      <XCircle className="w-4 h-4 text-red-500" />
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-2">Loading datastream...</span>
      </div>
    )
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Stream d'Événements d'Anomalies</h1>
        <p className="text-gray-600">
          Visualisation en temps réel des événements réseau avec détection d'anomalies (1000 événements de test)
        </p>
      </div>

      {/* Controls */}
      <div className="mb-6 flex items-center gap-4 bg-white p-4 rounded-lg shadow-sm border">
        <button
          onClick={handlePlay}
          disabled={isPlaying || currentIndex >= testData.length}
          className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Play className="w-4 h-4" />
          {currentIndex >= testData.length ? 'Reset & Play' : 'Play'}
        </button>

        <button
          onClick={handlePause}
          disabled={!isPlaying}
          className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Pause className="w-4 h-4" />
          Pause
        </button>

        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>

        <div className="text-sm text-gray-600">
          Index: {currentIndex}/{testData.length}
        </div>
      </div>

      {/* Events Table */}
      <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
        <div className="p-4 border-b">
          <h2 className="text-lg font-semibold">Événements Réseau (20 événements affichés)</h2>
          <p className="text-sm text-gray-600">
            Animation: Plus récents en haut • 20 derniers événements conservés
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Timestamp
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Source
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Destination
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Label
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type d'Événement
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Prédiction
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Vraie Valeur
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {displayedEvents.map((event, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {formatTimestamp(event.timestamp)}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900 font-mono">
                    {event.source}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900 font-mono">
                    {event.destination}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                      {event.label}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      event.event_type === 'add'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {event.event_type}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      event.prediction === 'ANOMALY'
                        ? 'bg-red-100 text-red-800'
                        : 'bg-green-100 text-green-800'
                    }`}>
                      {event.prediction}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      event.true_label === 'ANOMALY'
                        ? 'bg-red-100 text-red-800'
                        : 'bg-green-100 text-green-800'
                    }`}>
                      {event.true_label}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {getStatusIcon(event.correct)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default DatastreamPage
