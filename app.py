"""
Web Application untuk Wedding Recommendation System
Menggunakan Flask dengan visualisasi clustering
"""

from flask import Flask, render_template, request, jsonify
import json
import io
import base64
from datetime import datetime

# Import modul lokal
from local_transformer_intent import LocalIntentPipeline
from conn import DatabaseConnection, get_db_connection
from recommendation import RecommendationEngine
from package_planner import WeddingPackagePlanner

# Import untuk visualisasi
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

app = Flask(__name__)

# Global variables untuk sistem
ai_pipeline = None
db = None
recommendation_engine = None
package_planner = None

def initialize_system():
    """Inisialisasi sistem"""
    global ai_pipeline, db, recommendation_engine, package_planner
    
    if ai_pipeline is None:
        print("🚀 Initializing Wedding Recommendation System...")
        ai_pipeline = LocalIntentPipeline("models/local_transformer_intent")
        print("   ✓ AI Model loaded")
        
        db = get_db_connection()
        if not db:
            raise Exception("Gagal terhubung ke database")
        print("   ✓ Database connected")
        
        recommendation_engine = RecommendationEngine(n_clusters=3)
        print("   ✓ Recommendation Engine ready")
        
        package_planner = WeddingPackagePlanner(db)
        print("   ✓ Package Planner ready")
        print("✅ System ready!\n")

def create_cluster_visualization(items, slots, clustering_result):
    """
    Membuat visualisasi clustering
    
    Returns:
        Base64 encoded image string
    """
    if not items or len(items) < 2:
        return None
    
    try:
        # Prepare features
        features = recommendation_engine.prepare_features(items, slots)
        
        if len(features) < 2:
            return None
        
        # Reduce dimensionality untuk visualisasi 2D
        if features.shape[1] > 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
        else:
            features_2d = features
        
        # Get cluster labels
        cluster_labels = [item.get('cluster', 0) for item in items]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot each cluster with different color
        unique_clusters = set(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster_id, color in zip(unique_clusters, colors):
            mask = np.array(cluster_labels) == cluster_id
            plt.scatter(
                features_2d[mask, 0], 
                features_2d[mask, 1],
                c=[color], 
                label=f'Cluster {cluster_id}',
                s=100,
                alpha=0.6,
                edgecolors='black'
            )
        
        # Highlight recommendations
        recommendations_ids = [rec.get('id') for rec in clustering_result.get('recommendations', [])[:5]]
        rec_indices = [i for i, item in enumerate(items) if item.get('id') in recommendations_ids]
        
        if rec_indices:
            plt.scatter(
                features_2d[rec_indices, 0],
                features_2d[rec_indices, 1],
                marker='*',
                s=500,
                c='gold',
                edgecolors='red',
                linewidths=2,
                label='Top Recommendations',
                zorder=5
            )
        
        plt.xlabel('Feature Dimension 1', fontsize=12)
        plt.ylabel('Feature Dimension 2', fontsize=12)
        plt.title('K-Means Clustering Visualization', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None

@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_query():
    """API endpoint untuk memproses query"""
    try:
        data = request.json
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return jsonify({'error': 'Query tidak boleh kosong'}), 400
        
        # Step 1: AI Intent Classification & Slot Extraction
        ai_result = ai_pipeline.predict(query_text)
        
        # Step 2: Database Query
        slots = ai_result['slots']
        intent = ai_result['intent_pred']
        
        # Check if query meminta paket pernikahan lengkap
        is_package_request = package_planner.detect_package_request(query_text, intent)
        
        # Jika query paket pernikahan, gunakan package planner
        if is_package_request:
            print(f"🎊 Detected package request: {query_text}")
            package_result = package_planner.create_package_recommendations(slots, include_optional=True)
            
            # Format response untuk package
            response = {
                'success': True,
                'query': query_text,
                'is_package': True,
                'ai_result': {
                    'intent': ai_result['intent_pred'],
                    'confidence': ai_result['probs'][ai_result['intent_pred']],
                    'all_intents': ai_result['probs'],
                    'slots': ai_result['slots']
                },
                'package': {
                    'categories_found': package_result['categories_found'],
                    'categories_missing': package_result['categories_missing'],
                    'total_categories': package_result['total_categories'],
                    'price_info': package_result['price_info'],
                    'budget_status': package_result['budget_status'],
                    'user_budget': package_result['user_budget'],
                    'package_tiers': package_result['package_tiers'],
                    'message': package_result['message']
                },
                'categories_detail': {}
            }
            
            # Tambahkan detail items per kategori
            for category, items in package_result['categories'].items():
                response['categories_detail'][category] = {
                    'items': items[:5],  # Top 5 per kategori untuk preview
                    'total_items': len(items)
                }
            
            response['timestamp'] = datetime.now().isoformat()
            
            return jsonify(response)
        
        # Else: Query normal (single category atau vendor search)
        # Map intent ke kategori database
        # Kategori yang tersedia:
        # - Venue: Tempat pernikahan
        # - Catering: Makanan dan minuman
        # - Decoration: Dekorasi
        # - MUA: Make Up Artist
        # - Docummentation: Foto & Video
        # - Attire: Busana pengantin
        # - Entertainment: Hiburan (musik, band)
        # - Master of Ceremony: MC/Pembawa acara
        # - Traditional Ceremony: Upacara adat
        
        intent_category_map = {
            "cari_rekomendasi_paket": None,  # Semua kategori
            "estimasi_budget": None,  # Semua kategori
            "cari_venue": "Venue",
            "cari_dekor": "Decoration",
            "cari_vendor": None,  # Bisa berbagai kategori
            "cari_catering": "Catering",
            "tanya_kemungkinan": None  # Semua kategori
        }
        
        # Deteksi kategori tambahan dari slots atau query
        category = intent_category_map.get(intent, None)
        
        # Cek keyword di query untuk kategori spesifik
        query_lower = query_text.lower()
        if not category:
            if any(word in query_lower for word in ['mua', 'makeup', 'make up', 'riasan', 'rias']):
                category = "MUA"
            elif any(word in query_lower for word in ['foto', 'video', 'dokumentasi', 'fotografi', 'videografi', 'photographer']):
                category = "Docummentation"
            elif any(word in query_lower for word in ['baju', 'busana', 'kebaya', 'gaun', 'attire', 'pakaian', 'sewa baju']):
                category = "Attire"
            elif any(word in query_lower for word in ['mc', 'pembawa acara', 'master of ceremony', 'host']):
                category = "Master of Ceremony"
            elif any(word in query_lower for word in ['band', 'musik', 'entertainment', 'hiburan', 'penyanyi', 'dj']):
                category = "Entertainment"
            elif any(word in query_lower for word in ['upacara', 'siraman', 'midodareni', 'ceremony', 'adat', 'ritual', 'tradisi']):
                category = "Traditional Ceremony"
        
        # Query database dengan flexible mode
        items = db.get_items_by_filter(
            tema=slots.get('tema'),
            lokasi=slots.get('lokasi'),
            budget_min=slots.get('budget_min'),
            budget_max=slots.get('budget_max'),
            category=category,
            flexible=True  # Enable flexible matching
        )
        
        # Step 3: K-Means Clustering & Recommendation
        if len(items) == 0:
            clustering_result = {
                "total_items": 0,
                "clusters": [],
                "recommendations": [],
                "message": "Tidak ada item yang ditemukan dengan kriteria tersebut"
            }
            visualization = None
        else:
            clustering_result = recommendation_engine.cluster_items(items, slots)
            
            # Create visualization
            visualization = create_cluster_visualization(items, slots, clustering_result)
        
        # Format response
        response = {
            'success': True,
            'query': query_text,
            'ai_result': {
                'intent': ai_result['intent_pred'],
                'confidence': ai_result['probs'][ai_result['intent_pred']],
                'all_intents': ai_result['probs'],
                'slots': ai_result['slots']
            },
            'database': {
                'total_items': len(items),
                'category_filter': category
            },
            'clustering': {
                'n_clusters': clustering_result.get('n_clusters', 0),
                'clusters': clustering_result.get('clusters', []),
                'best_cluster_id': clustering_result.get('best_cluster_id'),
                'message': clustering_result.get('message', '')
            },
            'recommendations': clustering_result.get('recommendations', [])[:10],
            'visualization': visualization,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """API endpoint untuk mendapatkan statistik database"""
    try:
        # Total items
        total_result = db.execute_query("SELECT COUNT(*) as total FROM items WHERE status='ongoing'")
        total_items = total_result[0]['total'] if total_result else 0
        
        # Items per kategori
        category_result = db.execute_query("""
            SELECT c.name, COUNT(i.id) as total
            FROM categories c
            LEFT JOIN items i ON c.id = i.category_id AND i.status = 'ongoing'
            GROUP BY c.id, c.name
            ORDER BY total DESC
        """)
        
        # Price range
        price_result = db.execute_query("""
            SELECT 
                MIN(price) as min_price,
                MAX(price) as max_price,
                AVG(price) as avg_price
            FROM items
            WHERE status = 'ongoing'
        """)
        
        return jsonify({
            'success': True,
            'total_items': total_items,
            'categories': category_result,
            'price_range': price_result[0] if price_result else {}
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    initialize_system()
    print("\n" + "="*80)
    print("🎊 WEDDING RECOMMENDATION SYSTEM - WEB VERSION 🎊")
    print("="*80)
    print("\n🌐 Server berjalan di: http://localhost:5000")
    print("📊 Akses statistik di: http://localhost:5000/api/stats")
    print("\n⌨️  Tekan Ctrl+C untuk menghentikan server")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
