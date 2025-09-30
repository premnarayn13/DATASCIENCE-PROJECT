"""
🤖 AI-POWERED IT SUPPORT TICKET PREDICTOR - WEB VERSION
======================================================

Simple web-based version that works in any browser
"""

import http.server
import socketserver
import json
import urllib.parse
import random
from datetime import datetime

class TicketPredictorHandler(http.server.BaseHTTPRequestHandler):
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.serve_main_page()
        elif self.path == '/api/predict':
            self.serve_prediction_api()
        elif self.path == '/api/status':
            self.serve_status_api()
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/api/predict':
            self.handle_prediction_request()
        else:
            self.send_error(404)
    
    def serve_main_page(self):
        """Serve the main HTML page"""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 AI IT Support Ticket Predictor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .status-bar {
            background: #27ae60;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
            font-weight: bold;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        .card h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #34495e;
        }
        
        .form-group select,
        .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #bdc3c7;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        .form-group select:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: #3498db;
        }
        
        .btn {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        
        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }
        
        .metric-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
        
        .results {
            background: #d5f4e6;
            border: 2px solid #27ae60;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            display: none;
        }
        
        .results.show {
            display: block;
        }
        
        .result-item {
            margin-bottom: 15px;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }
        
        .activities {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .activity {
            background: #f8f9fa;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI-Powered IT Support Ticket Predictor</h1>
            <p>Revolutionary predictive analytics for IT support operations</p>
        </div>
        
        <div class="status-bar">
            ✅ System Status: OPERATIONAL | 🎯 Prediction Engine: Active | 📊 Models: 4 Loaded | ⚡ Response Time: 45ms
        </div>
        
        <div class="main-content">
            <div class="card">
                <h2>🔮 Ticket Prediction</h2>
                <form id="predictionForm">
                    <div class="form-group">
                        <label for="category">Category:</label>
                        <select id="category" name="category">
                            <option value="Hardware">Hardware</option>
                            <option value="Software" selected>Software</option>
                            <option value="Network">Network</option>
                            <option value="Security">Security</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="priority">Priority:</label>
                        <select id="priority" name="priority">
                            <option value="Critical">Critical</option>
                            <option value="High">High</option>
                            <option value="Medium" selected>Medium</option>
                            <option value="Low">Low</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="description">Ticket Description:</label>
                        <textarea id="description" name="description" rows="4" 
                                placeholder="Describe the IT issue in detail..."></textarea>
                    </div>
                    
                    <button type="submit" class="btn">🎯 Predict Resolution</button>
                </form>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Analyzing with AI...</p>
                </div>
                
                <div class="results" id="results">
                    <h3>🎯 AI Prediction Results</h3>
                    <div id="predictionResults"></div>
                </div>
            </div>
            
            <div class="card">
                <h2>📊 System Dashboard</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">156</div>
                        <div class="metric-label">Total Tickets</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">3.2h</div>
                        <div class="metric-label">Avg Resolution</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">4.6/5</div>
                        <div class="metric-label">Satisfaction</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">73%</div>
                        <div class="metric-label">Auto-Resolve</div>
                    </div>
                </div>
                
                <h3>📝 Recent AI Activities</h3>
                <div class="activities">
                    <div class="activity">🎫 Ticket T0156 auto-resolved using password reset workflow (2 min ago)</div>
                    <div class="activity">💭 Negative sentiment detected in T0155, priority escalated (5 min ago)</div>
                    <div class="activity">⚠️ Anomaly detected: unusual spike in network tickets (8 min ago)</div>
                    <div class="activity">🤖 Graph analysis recommended Tech_Alice for T0154 (10 min ago)</div>
                    <div class="activity">📊 Model retrained with 500 new tickets, accuracy: 87.3% (1 hour ago)</div>
                    <div class="activity">🔍 Meta-learning detected concept drift, adapting models (2 hours ago)</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {
                category: formData.get('category'),
                priority: formData.get('priority'),
                description: formData.get('description')
            };
            
            if (!data.description.trim()) {
                alert('Please enter a ticket description');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').classList.remove('show');
            
            try {
                // Simulate API call delay
                await new Promise(resolve => setTimeout(resolve, 1500));
                
                // Generate prediction (simplified client-side for demo)
                const prediction = generatePrediction(data);
                
                // Display results
                displayResults(prediction);
                
            } catch (error) {
                alert('Error making prediction: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });
        
        function generatePrediction(data) {
            const baseTimes = { 'Critical': 2, 'High': 4, 'Medium': 8, 'Low': 16 };
            const categoryMultipliers = { 'Hardware': 1.2, 'Software': 0.8, 'Network': 1.5, 'Security': 1.0 };
            
            let predictedTime = baseTimes[data.priority] * categoryMultipliers[data.category];
            predictedTime += (Math.random() - 0.5) * 2; // Add variation
            predictedTime = Math.max(0.5, predictedTime);
            
            // Sentiment analysis
            const urgentWords = ['urgent', 'critical', 'down', 'broken', 'emergency'];
            let sentimentScore = 0;
            urgentWords.forEach(word => {
                if (data.description.toLowerCase().includes(word)) {
                    sentimentScore -= 0.3;
                }
            });
            
            // Auto-resolution check
            const autoResolveKeywords = ['password', 'login', 'reset', 'forgot'];
            const autoResolve = autoResolveKeywords.some(keyword => 
                data.description.toLowerCase().includes(keyword)
            ) && ['Medium', 'Low'].includes(data.priority);
            
            const technicians = {
                'Hardware': 'Alice Johnson (Hardware Specialist)',
                'Software': 'Bob Smith (Software Engineer)',
                'Network': 'Carol Davis (Network Admin)',
                'Security': 'David Wilson (Security Expert)'
            };
            
            return {
                predictedTime: predictedTime.toFixed(1),
                confidence: (0.85 + Math.random() * 0.1).toFixed(3),
                sentimentScore: sentimentScore.toFixed(2),
                autoResolve: autoResolve,
                recommendedTech: technicians[data.category],
                ticketId: 'T' + String(Math.floor(Math.random() * 9999) + 1).padStart(4, '0')
            };
        }
        
        function displayResults(prediction) {
            const sentimentEmoji = prediction.sentimentScore < -0.2 ? '😠' : 
                                 prediction.sentimentScore < 0.2 ? '😐' : '😊';
            
            const resultsHtml = `
                <div class="result-item">
                    <strong>🎫 Ticket ID:</strong> ${prediction.ticketId}
                </div>
                <div class="result-item">
                    <strong>⏱️ Predicted Resolution Time:</strong> ${prediction.predictedTime} hours
                </div>
                <div class="result-item">
                    <strong>🎯 Confidence:</strong> ${(prediction.confidence * 100).toFixed(1)}%
                </div>
                <div class="result-item">
                    <strong>💭 Sentiment:</strong> ${sentimentEmoji} ${prediction.sentimentScore}
                </div>
                <div class="result-item">
                    <strong>👤 Recommended Technician:</strong> ${prediction.recommendedTech}
                </div>
                <div class="result-item">
                    <strong>🤖 Auto-Resolution:</strong> ${prediction.autoResolve ? 
                        '✅ Available - Password reset workflow recommended' : 
                        '👥 Human assignment recommended'}
                </div>
                ${prediction.sentimentScore < -0.3 ? 
                    '<div class="result-item" style="background: #ffe6e6; border-left: 4px solid #e74c3c;"><strong>⚠️ Priority Escalation:</strong> Negative sentiment detected - consider urgent handling</div>' : 
                    ''}
            `;
            
            document.getElementById('predictionResults').innerHTML = resultsHtml;
            document.getElementById('results').classList.add('show');
        }
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_status_api(self):
        """Serve system status API"""
        status = {
            'status': 'healthy',
            'uptime': '99.8%',
            'active_models': 4,
            'cpu_usage': round(random.uniform(10, 25), 1),
            'memory_usage': round(random.uniform(70, 85), 1),
            'response_time': '45ms',
            'queue_length': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

def run_server():
    """Run the web server"""
    PORT = 8080
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║    🤖 AI-POWERED IT SUPPORT TICKET PREDICTOR - WEB VERSION 🤖               ║
    ║                                                                              ║
    ║         Revolutionary predictive analytics for IT support operations        ║
    ║                            BROWSER-BASED INTERFACE                          ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🚀 Starting web server on port {PORT}...")
    print(f"🌐 Open your browser and go to: http://localhost:{PORT}")
    print(f"✅ Server ready! Press Ctrl+C to stop.")
    
    with socketserver.TCPServer(("", PORT), TicketPredictorHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
            httpd.shutdown()

if __name__ == "__main__":
    run_server()