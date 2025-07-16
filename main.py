import os
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import re

# Only import from detector.py
from detector import (
    AITextDetector, 
    detect_with_all_models, 
    detect_with_selected_models, 
    detect_with_top_models,
    get_available_models as get_detection_models,
    get_ai_lines,
    get_ai_sentences,
    highlight_ai_text
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")

# Initialize detector service
ai_detector = AITextDetector()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "🚀 Humanize AI Server is running! (Detector only)",
        "features": {
            "ai_detection": True
        }
    })

@app.route('/detect', methods=['POST'])
def detect_ai_handler():
    """Main AI detection endpoint using ensemble method with enhanced options"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        data = request.get_json()
        text = data.get('text', '').strip()
        threshold = data.get('threshold', 0.7)
        models = data.get('models', None)
        use_all_models = data.get('use_all_models', False)
        top_n = data.get('top_n', None)
        criteria = data.get('criteria', 'performance')
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if len(text) < 20:
            return jsonify({"error": "Text must be at least 20 characters long"}), 400
        if len(text) > 50000:
            return jsonify({"error": "Text must be less than 50,000 characters"}), 400
        if use_all_models:
            result = detect_with_all_models(text)
        elif top_n and isinstance(top_n, int) and top_n > 0:
            result = detect_with_top_models(text, n=top_n, criteria=criteria)
        elif models and isinstance(models, list):
            result = detect_with_selected_models(text, models)
        else:
            result = ai_detector.detect_ensemble(text, models=models)
        is_ai = result['ensemble_ai_probability'] > threshold
        response = {
            "detection_result": result,
            "is_ai": is_ai,
            "threshold": threshold,
            "success": True
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing detection: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "success": False
        }), 500

@app.route('/detect_sentences', methods=['POST'])
def detect_sentences_handler():
    """Detect which specific sentences in text are AI-generated"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        data = request.get_json()
        text = data.get('text', '').strip()
        threshold = data.get('threshold', 0.6)
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if len(text) < 50:
            return jsonify({"error": "Text must be at least 50 characters long for sentence detection"}), 400
        if len(text) > 15000:
            return jsonify({"error": "Text must be less than 15,000 characters for sentence detection"}), 400
        detector = AITextDetector()
        result = detector.detect_ai_sentences(text, threshold)
        response = {
            "ai_detected_sentences": result['ai_detected_sentences'],
            "human_sentences": result['human_sentences'],
            "sentence_analysis": result['sentence_analysis'],
            "statistics": result['statistics'],
            "threshold_used": result['threshold_used'],
            "text_length": len(text),
            "success": True
        }
        logger.info(f"Sentence detection: {result['statistics']['ai_generated_sentences']}/{result['statistics']['total_sentences_analyzed']} sentences detected as AI")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in sentence detection: {str(e)}")
        return jsonify({
            "error": "Failed to detect AI sentences",
            "success": False
        }), 500

@app.route('/highlight_ai', methods=['POST'])
def highlight_ai_handler():
    """Highlight AI-detected portions in text"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        data = request.get_json()
        text = data.get('text', '').strip()
        threshold = data.get('threshold', 0.6)
        output_format = data.get('format', 'markdown')
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if output_format not in ['markdown', 'html', 'plain']:
            return jsonify({"error": "format must be 'markdown', 'html', or 'plain'"}), 400
        if len(text) < 50:
            return jsonify({"error": "Text must be at least 50 characters long for highlighting"}), 400
        if len(text) > 15000:
            return jsonify({"error": "Text must be less than 15,000 characters for highlighting"}), 400
        highlighted_text = highlight_ai_text(text, threshold, output_format)
        detector = AITextDetector()
        sentence_result = detector.detect_ai_sentences(text, threshold)
        response = {
            "original_text": text,
            "highlighted_text": highlighted_text,
            "output_format": output_format,
            "threshold_used": threshold,
            "ai_sentences_count": len(sentence_result['ai_detected_sentences']),
            "total_sentences": len(sentence_result['sentence_analysis']),
            "ai_percentage": sentence_result['statistics']['ai_percentage'],
            "text_length": len(text),
            "success": True
        }
        logger.info(f"Text highlighting completed: {len(sentence_result['ai_detected_sentences'])} AI sentences highlighted")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /highlight_ai: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Humanize AI Server (Detector only)...")
    app.run(debug=False, host='0.0.0.0', port=8080)