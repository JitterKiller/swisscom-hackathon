import React from 'react'
import { Activity, Upload, Zap, Database } from 'lucide-react'

const Sidebar = ({ activePage, setActivePage }) => {
  const menuItems = [
    {
      id: 'predict',
      label: 'Predict Edge Anomaly',
      icon: Activity,
      description: 'Analysez les anomalies d\'arêtes'
    },
    {
      id: 'datastream',
      label: 'Datastream Events',
      icon: Database,
      description: 'Visualisez le flux d\'événements'
    },
    {
      id: 'upload',
      label: 'Upload Graph',
      icon: Upload,
      description: 'Importez vos données de graphe'
    }
  ]

  return (
    <div className="w-80 bg-white shadow-lg border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900">Graph Analytics</h1>
            <p className="text-sm text-gray-500">Swisscom Challenge</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <div className="space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon
            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                className={`w-full sidebar-link ${
                  activePage === item.id ? 'active' : 'text-gray-700'
                }`}
              >
                <Icon className="w-5 h-5" />
                <div className="text-left">
                  <div className="font-medium">{item.label}</div>
                  <div className="text-xs text-gray-500">{item.description}</div>
                </div>
              </button>
            )
          })}
        </div>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200">
        <div className="text-xs text-gray-400 text-center">
          Graph Anomaly Detection v1.0
        </div>
      </div>
    </div>
  )
}

export default Sidebar
