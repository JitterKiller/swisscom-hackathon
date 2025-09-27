import React, { useState } from 'react'
import Sidebar from './components/Sidebar'
import PredictAnomalyPage from './pages/PredictAnomalyPage'
import UploadGraphPage from './pages/UploadGraphPage'
import DatastreamPage from './pages/DatastreamPage'

function App() {
  const [activePage, setActivePage] = useState('predict')

  const renderPage = () => {
    switch (activePage) {
      case 'predict':
        return <PredictAnomalyPage />
      case 'datastream':
        return <DatastreamPage />
      case 'upload':
        return <UploadGraphPage />
      default:
        return <PredictAnomalyPage />
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar activePage={activePage} setActivePage={setActivePage} />
      <main className="flex-1 overflow-auto">
        {renderPage()}
      </main>
    </div>
  )
}

export default App
