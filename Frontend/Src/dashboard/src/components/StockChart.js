import React, { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import './StockChart.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const StockChart = ({ data, ticker }) => {
  const chartRef = useRef(null);

  // Prepare chart data from stock data
  const prepareChartData = () => {
    if (!data || !data.dates || !data.prices) {
      return {
        labels: [],
        datasets: []
      };
    }

    return {
      labels: data.dates.map(date => new Date(date).toLocaleDateString()),
      datasets: [
        {
          label: `${ticker} Stock Price`,
          data: data.prices,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 3,
          fill: true,
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: '#3b82f6',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
        },
        // Add volume data if available
        ...(data.volume ? [{
          label: 'Volume',
          data: data.volume,
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderWidth: 2,
          fill: false,
          tension: 0.2,
          yAxisID: 'y1',
          pointRadius: 2,
          pointHoverRadius: 4,
        }] : [])
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20,
          font: {
            size: 12,
            weight: 'bold'
          }
        }
      },
      title: {
        display: true,
        text: `${ticker} - Historical Price Chart`,
        font: {
          size: 18,
          weight: 'bold'
        },
        padding: 20
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: '#3b82f6',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          label: (context) => {
            const value = context.parsed.y;
            if (context.datasetIndex === 0) {
              return `Price: $${value.toFixed(2)}`;
            } else {
              return `Volume: ${(value / 1000000).toFixed(2)}M`;
            }
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Date',
          font: {
            weight: 'bold'
          }
        },
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          maxTicksLimit: 10
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Price ($)',
          font: {
            weight: 'bold'
          }
        },
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          callback: function(value) {
            return '$' + value.toFixed(2);
          }
        }
      },
      ...(data?.volume ? {
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          title: {
            display: true,
            text: 'Volume',
            font: {
              weight: 'bold'
            }
          },
          grid: {
            drawOnChartArea: false,
          },
          ticks: {
            callback: function(value) {
              return (value / 1000000).toFixed(1) + 'M';
            }
          }
        }
      } : {})
    }
  };

  // Calculate price statistics
  const calculateStats = () => {
    if (!data || !data.prices || data.prices.length === 0) return null;
    
    const prices = data.prices;
    const currentPrice = prices[prices.length - 1];
    const previousPrice = prices[prices.length - 2];
    const change = currentPrice - previousPrice;
    const changePercent = (change / previousPrice) * 100;
    
    const highPrice = Math.max(...prices);
    const lowPrice = Math.min(...prices);
    
    return {
      current: currentPrice,
      change: change,
      changePercent: changePercent,
      high: highPrice,
      low: lowPrice
    };
  };

  const stats = calculateStats();

  return (
    <div className="stock-chart-container">
      <div className="chart-header">
        <h2 className="chart-title">
          📈 {ticker} Stock Analysis
        </h2>
        
        {stats && (
          <div className="price-stats">
            <div className="stat-item current-price">
              <span className="stat-label">Current Price</span>
              <span className="stat-value">${stats.current.toFixed(2)}</span>
            </div>
            
            <div className={`stat-item price-change ${stats.change >= 0 ? 'positive' : 'negative'}`}>
              <span className="stat-label">Change</span>
              <span className="stat-value">
                {stats.change >= 0 ? '+' : ''}${stats.change.toFixed(2)} 
                ({stats.change >= 0 ? '+' : ''}{stats.changePercent.toFixed(2)}%)
              </span>
            </div>
            
            <div className="stat-item">
              <span className="stat-label">High</span>
              <span className="stat-value">${stats.high.toFixed(2)}</span>
            </div>
            
            <div className="stat-item">
              <span className="stat-label">Low</span>
              <span className="stat-value">${stats.low.toFixed(2)}</span>
            </div>
          </div>
        )}
      </div>

      <div className="chart-wrapper">
        <Line 
          ref={chartRef}
          data={prepareChartData()} 
          options={chartOptions} 
        />
      </div>

      <div className="chart-info">
        <p>
          📊 This chart shows historical stock price data used for prediction analysis. 
          The AI model analyzes patterns, trends, and technical indicators from this data.
        </p>
      </div>
    </div>
  );
};

export default StockChart;
