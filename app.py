"""
app.py - FIXED Flask UI for Multimodal RAG System with Accurate Results
"""

import os
import shutil
import time
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import uuid

# Import your pipeline components
from multimodal_unified_parallel import ParallelMasterPipeline
from ragchatbot import CrossModalRAGChatbot

# Upgrades: Redis caching, hallucination evaluation, hardware-aware model config
from src.cache.redis_cache import RAGCache
from src.evaluation.hallucination_eval import evaluate_response, render_eval_panel, check_retrieval_confidence
from src.optimization.model_loader import CONFIG, OptimizedCLIPEncoder

# Keep cache hit/miss DEBUG logs visible without making the whole app noisy
logging.basicConfig(level=logging.INFO)
logging.getLogger('src.cache.redis_cache').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.secret_key = 'change-this-secret-key-in-production'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed_output'
app.config['CLIP_MODEL_PATH'] = 'openai/clip-vit-base-patch32'

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'pdf', 'docx', 'doc',
    'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp',
    'mp3', 'wav', 'm4a', 'flac', 'ogg', 'opus', 'wma', 'aac'
}

# Global state
chatbot_instance = None
processing_lock = False
current_session_id = None

# Upgrade 1: Redis cache. Every method degrades to a cache-miss if Redis
# is unreachable, so the app still works with Redis down.
cache = RAGCache(host=os.environ.get('REDIS_HOST', 'localhost'), port=int(os.environ.get('REDIS_PORT', 6379)))

# Upgrade 3: hardware-aware model config + optional ONNX CLIP encoder.
# NOTE: optimized_encoder is intentionally NOT used for retrieval — the
# live FAISS index was built with vectorstore.py's UnifiedEmbeddingGenerator
# (sentence-transformers/clip-ViT-B-32), and swapping in a different base
# model's embeddings would silently corrupt similarity search. It's kept
# available here for future offline re-indexing work.
try:
    optimized_encoder = OptimizedCLIPEncoder()
except Exception as e:
    logger.warning(f"OptimizedCLIPEncoder unavailable ({e}); continuing without it")
    optimized_encoder = None

print(f"[STARTUP] Mode: {CONFIG['llm']} | Reranker: {CONFIG['reranker']} | Whisper: {CONFIG['whisper']} | Workers: {CONFIG['workers']}")
print(f"[STARTUP] Redis: {'connected' if cache.get_cache_stats().get('redis_available') else 'unavailable (running without cache)'}")
print(f"[STARTUP] ONNX CLIP encoder: {'ready' if optimized_encoder is not None else 'unavailable (fallback only, not used for retrieval)'}")

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Resume from a previously processed index if one exists on disk, so a
# server restart doesn't require re-uploading and re-processing everything.
_existing_index_path = os.path.join(app.config['PROCESSED_FOLDER'], 'multimodal_index.bin')
if os.path.exists(_existing_index_path):
    try:
        print(f"[STARTUP] Found existing index at {_existing_index_path} — loading...")
        chatbot_instance = CrossModalRAGChatbot(index_path=_existing_index_path)
        print(f"[STARTUP] Resumed with {len(chatbot_instance.metadata)} chunks already indexed")
    except Exception as e:
        print(f"[STARTUP] Could not auto-load existing index ({e}); upload files to start fresh")
        chatbot_instance = None


def load_html():
    """Load the HTML file."""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
        <h1>Error: index.html not found</h1>
        <p>Please ensure index.html is in the same directory as app.py</p>
        </body>
        </html>
        """


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_all_data():
    """Clear all uploaded and processed files."""
    global chatbot_instance, current_session_id
    
    # Clear uploads
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Clear processed
    if os.path.exists(app.config['PROCESSED_FOLDER']):
        shutil.rmtree(app.config['PROCESSED_FOLDER'])
        os.makedirs(app.config['PROCESSED_FOLDER'])
    
    chatbot_instance = None
    current_session_id = None

    # The document index is about to be rebuilt (or wiped) — any cached
    # retrieval/embedding results now point at data that no longer exists.
    cache.invalidate_document_caches()


@app.route('/')
def index():
    """Serve the main HTML page."""
    html_content = load_html()
    return render_template_string(html_content)


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and processing."""
    global chatbot_instance, processing_lock, current_session_id
    
    try:
        print("\n" + "="*70)
        print("📤 UPLOAD REQUEST RECEIVED")
        print("="*70)
        
        # Check if already processing
        if processing_lock:
            print("⚠️  System busy")
            return jsonify({
                'success': False,
                'error': 'System is busy processing files. Please wait.'
            }), 429
        
        # Check files
        if 'files' not in request.files:
            print("❌ No files in request")
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        print(f"📦 Received {len(files)} files")
        
        if not files or files[0].filename == '':
            print("❌ No files selected")
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        # Start processing
        processing_lock = True
        
        # Clear previous data
        print("🧹 Clearing previous data...")
        clear_all_data()
        
        # Create new session
        current_session_id = str(uuid.uuid4())
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], current_session_id)
        os.makedirs(session_dir, exist_ok=True)
        print(f"📁 Session: {current_session_id}")
        
        # Save uploaded files
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(session_dir, filename)
                file.save(filepath)
                uploaded_files.append(filename)
                print(f"   ✅ Saved: {filename}")
        
        if not uploaded_files:
            processing_lock = False
            print("❌ No valid files")
            return jsonify({
                'success': False,
                'error': 'No valid files uploaded'
            }), 400
        
        print(f"\n🚀 Processing {len(uploaded_files)} files...")
        
        # Process files
        pipeline = ParallelMasterPipeline(
            clip_model_path=app.config['CLIP_MODEL_PATH'],
            output_dir=app.config['PROCESSED_FOLDER'],
            max_workers=CONFIG['workers']
        )
        
        results = pipeline.run_complete_pipeline(session_dir)
        
        # Initialize chatbot with FULL capabilities
        print("\n🤖 Initializing enhanced chatbot...")
        chatbot_instance = CrossModalRAGChatbot(
            index_path=results['index_path']
        )
        
        processing_lock = False
        
        print("\n✅ UPLOAD & PROCESSING COMPLETE")
        print("="*70 + "\n")
        
        return jsonify({
            'success': True,
            'message': f'Processed {results["total_chunks"]} chunks successfully',
            'files': uploaded_files,
            'total_chunks': results['total_chunks'],
            'duration': f"{results['duration_seconds']:.1f}s"
        }), 200
    
    except Exception as e:
        processing_lock = False
        print(f"\n❌ UPLOAD FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/query', methods=['POST'])
def query():
    """
    Handle text queries with ENHANCED accuracy.
    
    KEY IMPROVEMENTS:
    1. Uses hybrid search with reranking (k=15 for better coverage)
    2. Leverages full context formatting from ragchatbot.py
    3. Proper error handling and logging
    4. Uses the comprehensive prompt that works in CLI
    5. Returns debug info for transparency
    """
    global chatbot_instance
    
    try:
        print("\n" + "="*70)
        print("💬 QUERY REQUEST (ENHANCED)")
        print("="*70)
        
        if chatbot_instance is None:
            print("❌ Chatbot not initialized")
            return jsonify({
                'success': False,
                'error': 'Please upload and process files first'
            }), 400
        
        # Get JSON data
        data = request.get_json(silent=True)
        if not data:
            print("❌ No JSON data")
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        query_text = data.get('query', '').strip()
        debug_mode = data.get('debug', False)  # Optional debug flag
        
        if not query_text:
            print("❌ Empty query")
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        print(f"🔍 Query: {query_text}")
        print(f"🐛 Debug mode: {debug_mode}")
        start_time = time.time()
        debug_info = None

        # Session management (Upgrade 1, layer 3)
        session_id = data.get('session_id') or cache.create_session()

        # Query embedding, cached (Upgrade 1, layer 1). Uses the pipeline's
        # own CLIP generator so the embedding space matches the FAISS index.
        query_embedding = cache.get_embedding(query_text)
        if query_embedding is None:
            query_embedding = chatbot_instance.generator.embed(query_text)
            cache.set_embedding(query_text, query_embedding)

        # Short, vague follow-ups ("What about culpable homicide?") retrieve
        # poorly on their own — fold in the previous user turn so retrieval
        # sees the same context the generation prompt already gets via
        # build_prompt_with_history. Only affects search, not the prompt's
        # "current question", the eval query, or what's saved to history.
        history = cache.get_history(session_id)
        previous_user_turns = [t['content'] for t in history if t['role'] == 'user']
        retrieval_query = f"{previous_user_turns[-1]} {query_text}" if previous_user_turns else query_text

        # Retrieval, cached (Upgrade 1, layer 2). Caches full result dicts
        # (score + metadata) so a cache hit skips hybrid search + reranking
        # entirely while downstream generation logic still gets everything
        # it normally would (including image paths for LLaVA).
        # CRITICAL FIX: Use the SAME method that works in CLI
        # - Hybrid search (Semantic + BM25 + Cross-encoder reranking)
        # - Proper context formatting by modality
        cache_hit = True
        search_results = cache.get_retrieval(retrieval_query)
        if search_results is None:
            cache_hit = False
            print("🔍 Performing hybrid search with reranking...")
            search_results = chatbot_instance.search_by_text(retrieval_query, k=15)
            cache.set_retrieval(retrieval_query, search_results)

        if not search_results:
            print("❌ No relevant information found")
            return jsonify({
                'success': True,
                'answer': "❌ No relevant information found.",
                'session_id': session_id,
                'cache_hit': cache_hit
            }), 200

        # Retrieval confidence gate: score the query directly against each
        # retrieved chunk with the reranker (its native query-vs-passage use
        # case, not the sentence-vs-passage repurposing check_faithfulness
        # does). If nothing clears the floor, the corpus has no relevant
        # information for this query — answer that instead of letting the
        # LLM fall back on its own pretrained knowledge for an out-of-domain
        # question. Skip the LLM call entirely; it would only be discarded.
        confidence = check_retrieval_confidence(
            retrieval_query,
            [r['metadata'].get('content', '') for r in search_results],
            chatbot_instance.search_engine.reranker
        )
        if not confidence['confident']:
            print(f"❌ Low retrieval confidence (best={confidence['best_score']:.4f}) — declining to answer")
            cache.add_turn(session_id, 'user', query_text)
            cache.add_turn(session_id, 'assistant', "I don't have relevant information about that in the indexed documents.")
            return jsonify({
                'success': True,
                'answer': "I don't have relevant information about that in the indexed documents.",
                'session_id': session_id,
                'cache_hit': cache_hit,
                'retrieval_confidence': confidence['best_score']
            }), 200

        # Build a history-aware prompt (Upgrade 1, layer 3) and generate
        # the answer through the existing multimodal generation path
        # (unchanged prompt engineering / LLaVA image handling otherwise).
        context = chatbot_instance.format_context_by_modality(search_results)
        context_text = "\n\n".join(
            r['metadata'].get('content', '')[:500] for r in search_results[:5]
        )
        prompt = cache.build_prompt_with_history(query_text, context_text, session_id)
        answer = chatbot_instance.generate_answer(
            query_text, search_results, query_type="text", custom_prompt=prompt
        )

        # Hallucination / quality evaluation (Upgrade 2). Best-effort: a
        # scoring failure must not take down an otherwise-good answer.
        try:
            chunk_texts = [r['metadata'].get('content', '') for r in search_results]
            eval_scores = evaluate_response(
                query_text, answer, chunk_texts,
                chatbot_instance.generator, chatbot_instance.search_engine.reranker
            )
            eval_panel_html = render_eval_panel(eval_scores)
        except Exception as eval_error:
            logger.warning(f"Evaluation step failed, continuing without scores: {eval_error}")
            eval_scores = None
            eval_panel_html = ""

        # Persist this turn to session memory (Upgrade 1, layer 3)
        cache.add_turn(session_id, 'user', query_text)
        cache.add_turn(session_id, 'assistant', answer)

        # Get search results for debug info
        if debug_mode:
            debug_info = {
                'num_results': len(search_results),
                'context_breakdown': {
                    'text_chunks': len(context['text']),
                    'tables': len(context['tables']),
                    'images': len(context['images']),
                    'image_files': len(context['image_paths']),
                    'audio': len(context['audio'])
                },
                'top_sources': list(set([
                    r['metadata'].get('source', 'Unknown')
                    for r in search_results[:5]
                ])),
                'top_scores': [
                    f"{r['score']:.4f}" for r in search_results[:5]
                ]
            }
            print(f"🐛 Debug info: {debug_info}")

        latency_ms = round((time.time() - start_time) * 1000, 1)
        print(f"✅ Answer generated ({len(answer)} chars) in {latency_ms}ms | cache_hit={cache_hit}")
        print("="*70 + "\n")

        response = {
            'success': True,
            'answer': answer,
            'session_id': session_id,
            'cache_hit': cache_hit,
            'latency_ms': latency_ms,
            'eval_scores': eval_scores,
            'eval_panel_html': eval_panel_html,
            'sources': [
                {
                    'source': r['metadata'].get('source', 'Unknown'),
                    'preview': r['metadata'].get('content_preview', r['metadata'].get('content', ''))[:200],
                    'score': r['score']
                }
                for r in search_results[:3]
            ]
        }

        if debug_mode:
            response['debug'] = debug_info

        return jsonify(response), 200
    
    except Exception as e:
        print(f"\n❌ QUERY FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        
        # Provide helpful error message
        error_msg = str(e)
        if "ollama" in error_msg.lower():
            error_msg = "❌ Ollama is not running. Please start it with: ollama serve"
        elif "connection" in error_msg.lower():
            error_msg = "❌ Cannot connect to Ollama. Ensure it's running on localhost:11434"
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/clear', methods=['POST'])
def clear():
    """Clear all data."""
    global chatbot_instance, processing_lock
    
    try:
        print("\n" + "="*70)
        print("🧹 CLEAR REQUEST")
        print("="*70)
        
        if processing_lock:
            print("⚠️  Cannot clear while processing")
            return jsonify({
                'success': False,
                'error': 'Cannot clear while processing'
            }), 429
        
        clear_all_data()
        
        print("✅ All data cleared")
        print("="*70 + "\n")
        
        return jsonify({
            'success': True,
            'message': 'All data cleared successfully'
        }), 200
    
    except Exception as e:
        print(f"\n❌ CLEAR FAILED: {e}\n")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/status', methods=['GET'])
def status():
    """Get system status with detailed info."""
    try:
        status_info = {
            'ready': chatbot_instance is not None,
            'processing': processing_lock,
            'session_id': current_session_id
        }
        
        # Add index statistics if chatbot is ready
        if chatbot_instance is not None:
            from collections import Counter
            metadata = chatbot_instance.metadata
            
            types = Counter(m['type'] for m in metadata)
            modalities = Counter(m.get('modality', 'text') for m in metadata)
            
            status_info['statistics'] = {
                'total_chunks': len(metadata),
                'by_type': dict(types),
                'by_modality': dict(modalities)
            }
        
        return jsonify(status_info), 200
    
    except Exception as e:
        return jsonify({
            'ready': False,
            'error': str(e)
        }), 500


@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """Return current Redis cache key counts and memory usage."""
    try:
        return jsonify(cache.get_cache_stats()), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/session/clear', methods=['POST'])
def clear_session():
    """Drop a single session's conversation history."""
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'status': 'error', 'error': 'session_id is required'}), 400
        cache.r.delete(f'session:{session_id}')
        return jsonify({'status': 'cleared'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


# Error handlers that return JSON
@app.errorhandler(404)
def not_found(e):
    print(f"❌ 404 Error: {request.path}")
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'path': request.path
    }), 404


@app.errorhandler(500)
def internal_error(e):
    print(f"❌ 500 Error: {e}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large (max 500MB)'
    }), 413


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 ENHANCED MULTIMODAL RAG CHATBOT SERVER")
    print("="*70)
    print("\n✨ KEY IMPROVEMENTS:")
    print("  ✅ Uses same search method as CLI (hybrid + reranking)")
    print("  ✅ Comprehensive prompt engineering")
    print("  ✅ Better context formatting by modality")
    print("  ✅ Enhanced error handling")
    print("  ✅ Detailed logging for debugging")
    print("\nFeatures:")
    print("  ✅ PDF, DOCX, Images, Audio support")
    print("  ✅ Parallel processing")
    print("  ✅ Cross-modal retrieval")
    print("  ✅ Natural language answers with citations")
    print("\nEndpoints:")
    print("  GET  /          - Main UI")
    print("  POST /upload    - Upload & process files")
    print("  POST /query     - Ask questions (ENHANCED)")
    print("  POST /clear     - Clear all data")
    print("  GET  /status    - System status")
    print("\n🌐 Access at: http://localhost:5001 (direct) — or via the Node API gateway on :5000")
    print("="*70 + "\n")
    
    # Check if index.html exists
    if not os.path.exists('index.html'):
        print("⚠️  WARNING: index.html not found in current directory!")
        print("   Please ensure index.html is in the same folder as app.py\n")
    
    # Check if Ollama is running
    try:
        import ollama
        ollama.list()
        print("✅ Ollama is running")
    except:
        print("⚠️  WARNING: Ollama might not be running!")
        print("   Start it with: ollama serve")
        print(f"   Pull the model with: ollama pull {CONFIG['llm']}\n")
    
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), threaded=True)