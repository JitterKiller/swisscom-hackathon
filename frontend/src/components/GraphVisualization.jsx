import React, { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'

const GraphVisualization = ({ data }) => {
  const svgRef = useRef()
  const tooltipRef = useRef()
  const [selectedNode, setSelectedNode] = useState(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [timeRange, setTimeRange] = useState([0, 1000])
  const [performanceMode, setPerformanceMode] = useState(false)

  useEffect(() => {
    console.log('GraphVisualization: data changed', data)
    if (!data || !data.nodes || !data.edges) {
      console.log('GraphVisualization: invalid data')
      return
    }

    // Initialize time range from metadata
    if (data.metadata && data.metadata.time_range) {
      const [minTime, maxTime] = data.metadata.time_range
      console.log('GraphVisualization: time range', minTime, maxTime)
      setTimeRange([minTime, maxTime])
      setCurrentTime(minTime)

      // Check if we have meaningful time range (more than just one timestamp)
      if (minTime === maxTime) {
        console.log('GraphVisualization: Single timestamp detected, hiding time slider')
      }
    } else {
      console.log('GraphVisualization: No metadata or time_range')
    }

    drawGraph()
  }, [data])

  useEffect(() => {
    if (data && data.edges) {
      drawGraph()
    }
  }, [currentTime])

  const drawGraph = () => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove() // Clear previous content

    // Get container dimensions
    const container = svgRef.current.parentElement
    const containerWidth = container.clientWidth
    const containerHeight = container.clientHeight
    
    const width = containerWidth - 40  // Account for padding
    const height = containerHeight - 40 // Account for padding
    const margin = { top: 20, right: 20, bottom: 20, left: 20 }

    svg
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', [0, 0, width, height])

    // Create a tooltip div
    const tooltip = d3.select(tooltipRef.current)
      .style('position', 'absolute')
      .style('visibility', 'hidden')
      .style('background-color', 'white')
      .style('border', '1px solid #ddd')
      .style('border-radius', '4px')
      .style('padding', '8px')
      .style('font-size', '12px')
      .style('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
      .style('z-index', 1000)
      .style('max-width', '200px')
      .style('pointer-events', 'none')

    // Create node index map for edge references
    const nodeIndex = new Map()
    data.nodes.forEach((node, index) => {
      nodeIndex.set(node.id, index)
    })

    // Filter edges by current time
    const filteredEdges = data.edges.filter(edge =>
      !edge.timestamp || edge.timestamp <= currentTime
    )

    // Convert edge references to node indices
    const edgesWithIndices = filteredEdges.map(edge => ({
      ...edge,
      source: nodeIndex.get(edge.source),
      target: nodeIndex.get(edge.target)
    }))

    // Create optimized simulation with tighter node clustering
    const linkStrength = performanceMode ? 0.1 : 0.2
    const chargeStrength = performanceMode ? -100 : -200
    const centerStrength = performanceMode ? 0.05 : 0.1
    const alphaDecay = performanceMode ? 0.02 : 0.015
    const theta = performanceMode ? 0.8 : 0.7
    const linkDistance = Math.min(30, width / 20) // Adaptive distance based on container size

    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(edgesWithIndices).id((d, i) => i).distance(linkDistance).strength(linkStrength))
      .force('charge', d3.forceManyBody().strength(chargeStrength).theta(theta))
      .force('center', d3.forceCenter(width / 2, height / 2).strength(centerStrength))
      .force('collision', d3.forceCollide().radius(d => Math.min((d.size || 6) + 1, width / 50)).strength(performanceMode ? 0.7 : 0.9))
      .alphaDecay(alphaDecay)
      .velocityDecay(performanceMode ? 0.4 : 0.3)

    // Draw links with adaptive rendering
    const linkOpacity = performanceMode ? 0.2 : 0.4
    const linkWidth = performanceMode ? 0.5 : 1

    const link = svg.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(edgesWithIndices)
      .enter().append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', linkOpacity)
      .attr('stroke-width', linkWidth)

    // Draw nodes
    const node = svg.append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(data.nodes)
      .enter().append('circle')
      .attr('r', d => Math.min(d.size || 6, width / 80))
      .attr('fill', d => d.color || '#69b3a2')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0)
          d.fx = null
          d.fy = null
        })
      )
      .on('mouseover', (event, d) => {
        const containerRect = svgRef.current.getBoundingClientRect()
        const x = event.clientX - containerRect.left
        const y = event.clientY - containerRect.top
        
        tooltip
          .style('visibility', 'visible')
          .html(`
            <strong>${d.id}</strong><br/>
            Type: ${d.type || 'Unknown'}<br/>
            Degré: ${d.degree || 0}<br/>
            ${d.label ? `Label: ${d.label}` : ''}
          `)
          .style('left', Math.min(x + 15, containerRect.width - 200) + 'px')
          .style('top', Math.max(y - 15, 10) + 'px')
      })
      .on('mouseout', () => {
        tooltip.style('visibility', 'hidden')
      })
      .on('click', (event, d) => {
        setSelectedNode(d)
      })

    // Add labels (hide in performance mode)
    const label = svg.append('g')
      .attr('class', 'labels')
      .selectAll('text')
      .data(performanceMode ? [] : data.nodes) // Hide labels in performance mode
      .enter().append('text')
      .text(d => d.id)
      .attr('font-size', 12)
      .attr('dx', 12)
      .attr('dy', '.35em')
      .style('pointer-events', 'none')
      .style('font-weight', 'bold')

    // Simulation tick with performance optimizations
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y)

      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y)

      // Only update labels if not in performance mode
      if (!performanceMode) {
        label
          .attr('x', d => d.x)
          .attr('y', d => d.y)
      }
    })

    // Optimized zoom behavior with performance considerations
    const zoom = d3.zoom()
      .scaleExtent([0.1, 3]) // Reduced max zoom for performance
      .on('zoom', (event) => {
        // Use transform on container groups for better performance
        svg.select('g.nodes').attr('transform', event.transform)
        svg.select('g.links').attr('transform', event.transform)
        // Only transform labels if not in performance mode
        if (!performanceMode) {
          svg.select('g.labels').attr('transform', event.transform)
        }
      })

    svg.call(zoom)
  }

  return (
    <div className="relative w-full h-full flex flex-col">
      <div className="flex-1">
        {/* Performance mode toggle */}
        <div className="absolute top-2 left-2 z-10">
          <button
            onClick={() => setPerformanceMode(!performanceMode)}
            className={`px-3 py-1 text-xs rounded ${
              performanceMode
                ? 'bg-green-600 text-white'
                : 'bg-gray-600 text-white hover:bg-gray-700'
            }`}
          >
            {performanceMode ? '⚡ Performance' : '🎨 Qualité'}
          </button>
        </div>

        <svg ref={svgRef} className="w-full h-full" style={{maxWidth: '100%', maxHeight: '100%'}}></svg>
        <div ref={tooltipRef}></div>

        {/* Node details panel */}
        {selectedNode && (
          <div className="absolute top-4 right-4 bg-white border rounded-lg shadow-lg p-4 max-w-xs">
            <h3 className="font-semibold text-lg mb-2">{selectedNode.id}</h3>
            <div className="space-y-1 text-sm">
              <div><strong>Type:</strong> {selectedNode.type || 'Unknown'}</div>
              <div><strong>Degré:</strong> {selectedNode.degree || 0}</div>
              {selectedNode.label && <div><strong>Label:</strong> {selectedNode.label}</div>}
              {selectedNode.properties && Object.entries(selectedNode.properties).map(([key, value]) => (
                <div key={key}><strong>{key}:</strong> {String(value)}</div>
              ))}
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              className="mt-3 text-xs text-gray-500 hover:text-gray-700"
            >
              Fermer
            </button>
          </div>
        )}
      </div>

      {/* Time slider - only show if we have a meaningful time range */}
      {timeRange[0] !== timeRange[1] && (
        <div className="bg-gray-50 border-t border-gray-200 p-3 mt-auto">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-gray-700 min-w-[80px]">
              T: {currentTime}
            </span>
            <div className="flex-1">
              <input
                type="range"
                min={timeRange[0]}
                max={timeRange[1]}
                value={currentTime}
                onChange={(e) => setCurrentTime(parseInt(e.target.value))}
                className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer slider"
                style={{
                  background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${(currentTime - timeRange[0]) / (timeRange[1] - timeRange[0]) * 100}%, #e5e7eb ${(currentTime - timeRange[0]) / (timeRange[1] - timeRange[0]) * 100}%, #e5e7eb 100%)`
                }}
              />
            </div>
            <span className="text-sm text-gray-500 min-w-[40px] text-right">
              {timeRange[1]}
            </span>
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-2">
            <span>{timeRange[0]}</span>
            <span className="text-center">🕐 Évolution temporelle</span>
            <span>{timeRange[1]}</span>
          </div>
        </div>
      )}

      {/* Show message when no time variation */}
      {timeRange[0] === timeRange[1] && (
        <div className="bg-white border-t p-4">
          <div className="text-center text-sm text-gray-500">
            Données statiques (timestamp unique: {timeRange[0]})
          </div>
        </div>
      )}
    </div>
  )
}

export default GraphVisualization
