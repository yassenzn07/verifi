@app.route('/detect-ai', methods=['POST'])
def detect_ai():
    data = request.get_json()
    user_text = data.get('text', '')
    
    result = ai_detector(user_text)
    label = result[0]['label']
    confidence = round(result[0]['score'] * 100, 2)
    
    return jsonify({
        'status': 'success',
        'label': label,
        'confidence': f"{confidence}%"
    })