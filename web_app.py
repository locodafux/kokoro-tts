#!/usr/bin/env python3
"""
Simple web interface for Kokoro TTS
"""

from flask import Flask, render_template_string, request, send_file
from tts_app import KokoroTTS
import os
import uuid

app = Flask(__name__)
tts = KokoroTTS(device='cpu')

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Kokoro TTS</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        textarea {
            width: 100%;
            height: 150px;
            margin: 10px 0;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        button {
            background: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #45a049;
        }
        .audio-player {
            margin-top: 20px;
            display: none;
        }
        .error {
            color: red;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Kokoro Text-to-Speech</h1>
        <form id="tts-form">
            <textarea id="text" placeholder="Enter text to convert to speech..."></textarea>
            <button type="submit">Generate Speech</button>
        </form>
        <div id="result" class="audio-player"></div>
        <div id="error" class="error"></div>
    </div>

    <script>
        document.getElementById('tts-form').onsubmit = async (e) => {
            e.preventDefault();
            const text = document.getElementById('text').value;
            if (!text.trim()) return;
            
            const resultDiv = document.getElementById('result');
            const errorDiv = document.getElementById('error');
            resultDiv.style.display = 'none';
            errorDiv.textContent = '';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });
                
                const data = await response.json();
                
                if (data.error) {
                    errorDiv.textContent = data.error;
                } else {
                    resultDiv.innerHTML = `
                        <audio controls autoplay>
                            <source src="/audio/${data.filename}" type="audio/mpeg">
                        </audio>
                        <br>
                        <a href="/download/${data.filename}" download>Download MP3</a>
                    `;
                    resultDiv.style.display = 'block';
                }
            } catch (error) {
                errorDiv.textContent = 'Error: ' + error.message;
            }
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return {'error': 'No text provided'}, 400
    
    try:
        # Generate unique filename
        filename = f"{uuid.uuid4().hex}.mp3"
        filepath = os.path.join('static', filename)
        
        # Create static directory if it doesn't exist
        os.makedirs('static', exist_ok=True)
        
        # Generate speech
        tts.text_to_speech(text, filepath)
        
        return {'filename': filename}
    
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/audio/<filename>')
def audio(filename):
    return send_file(f'static/{filename}', mimetype='audio/mpeg')

@app.route('/download/<filename>')
def download(filename):
    return send_file(f'static/{filename}', as_attachment=True, download_name=filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
