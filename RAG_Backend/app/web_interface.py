from flask import Flask, render_template, request, jsonify
from query_engine import RAGQueryEngine
import os

app = Flask(__name__)
rag_engine = None

def get_rag_engine():
    """Get or create RAG engine instance"""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGQueryEngine()
    return rag_engine

@app.route('/')
def index():
    """Main page with query interface"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Query Interface</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .query-form { margin-bottom: 30px; }
            textarea { width: 100%; height: 100px; margin: 10px 0; padding: 10px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .loading { color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Query Interface</h1>
            <div class="query-form">
                <h3>Ask a Question</h3>
                <form id="queryForm">
                    <textarea id="query" name="query" placeholder="Enter your question here..."></textarea>
                    <br>
                    <button type="submit">Submit Query</button>
                </form>
            </div>
            <div id="result" class="result" style="display: none;">
                <h3>Answer</h3>
                <div id="answer"></div>
                <h4>Sources</h4>
                <div id="sources"></div>
            </div>
            <div id="loading" class="loading" style="display: none;">
                Processing your query...
            </div>
        </div>

        <script>
            document.getElementById('queryForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const query = document.getElementById('query').value;
                if (!query.trim()) return;
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({query: query})
                    });
                    
                    const data = await response.json();
                    
                    // Display result
                    document.getElementById('answer').textContent = data.answer;
                    
                    // Display sources
                    const sourcesDiv = document.getElementById('sources');
                    sourcesDiv.innerHTML = '';
                    if (data.sources && data.sources.length > 0) {
                        const ul = document.createElement('ul');
                        data.sources.forEach(source => {
                            const li = document.createElement('li');
                            li.textContent = `File: ${source.file_name} - ${source.text_markdown?.substring(0, 100)}...`;
                            ul.appendChild(li);
                        });
                        sourcesDiv.appendChild(ul);
                    } else {
                        sourcesDiv.textContent = 'No sources found';
                    }
                    
                    document.getElementById('result').style.display = 'block';
                } catch (error) {
                    document.getElementById('answer').textContent = 'Error: ' + error.message;
                    document.getElementById('result').style.display = 'block';
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/query', methods=['POST'])
def query():
    """Handle query requests"""
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Get RAG engine and process query
        engine = get_rag_engine()
        result = engine.query(user_query)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port) 