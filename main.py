"""
Main Application
Integrasi lengkap antara AI Intent Classification, Database Query, dan K-Means Clustering
"""

import json
import sys
from typing import Dict, Any

# Import modul lokal
from local_transformer_intent import LocalIntentPipeline
from conn import DatabaseConnection, get_db_connection
from recommendation import RecommendationEngine


class WeddingRecommendationSystem:
    """Class utama untuk sistem rekomendasi pernikahan"""
    
    def __init__(self, model_dir: str = "models/local_transformer_intent", n_clusters: int = 3):
        """
        Inisialisasi sistem rekomendasi
        
        Args:
            model_dir: Path ke direktori model AI
            n_clusters: Jumlah cluster untuk K-Means
        """
        print("🚀 Menginisialisasi Wedding Recommendation System...")
        
        # Load AI model
        print("   • Loading AI Intent Model...")
        self.ai_pipeline = LocalIntentPipeline(model_dir)
        print("   ✓ AI Model loaded")
        
        # Setup database connection
        print("   • Connecting to Database...")
        self.db = get_db_connection()
        if not self.db:
            raise Exception("Gagal terhubung ke database")
        print("   ✓ Database connected")
        
        # Setup recommendation engine
        print("   • Initializing Recommendation Engine...")
        self.recommendation_engine = RecommendationEngine(n_clusters=n_clusters)
        print("   ✓ Recommendation Engine ready")
        
        print("\n✅ Sistem siap digunakan!\n")
    
    def map_intent_to_category(self, intent: str, query_text: str = "") -> str:
        """
        Mapping intent ke kategori database dengan deteksi keyword
        
        Args:
            intent: Intent yang diprediksi AI
            query_text: Teks query asli untuk deteksi keyword
            
        Returns:
            Nama kategori untuk query database
        """
        # Kategori yang tersedia di database:
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
        
        category = intent_category_map.get(intent, None)
        
        # Deteksi kategori dari keyword di query
        if query_text and not category:
            query_lower = query_text.lower()
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
        
        return category
    
    def process_query(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Memproses query dari user secara lengkap
        
        Args:
            query: Pertanyaan dari user
            verbose: Apakah menampilkan output detail
            
        Returns:
            Dict berisi hasil lengkap dari AI, database query, dan clustering
        """
        # Step 1: AI Intent Classification & Slot Extraction
        if verbose:
            print("\n" + "="*80)
            print("STEP 1: AI INTENT CLASSIFICATION & SLOT EXTRACTION")
            print("="*80)
        
        ai_result = self.ai_pipeline.predict(query)
        
        if verbose:
            print("\n📝 Input:")
            print(f"   {query}")
            print(f"\n🤖 AI Prediction:")
            print(f"   Intent: {ai_result['intent_pred']}")
            print(f"   Confidence: {ai_result['probs'][ai_result['intent_pred']]:.2%}")
            print(f"\n🎯 Extracted Slots:")
            for slot_name, slot_value in ai_result['slots'].items():
                if slot_value is not None:
                    if isinstance(slot_value, int) and slot_value >= 1000000:
                        print(f"   • {slot_name}: Rp {slot_value:,}")
                    else:
                        print(f"   • {slot_name}: {slot_value}")
        
        # Step 2: Database Query
        if verbose:
            print("\n" + "="*80)
            print("STEP 2: DATABASE QUERY")
            print("="*80)
        
        slots = ai_result['slots']
        intent = ai_result['intent_pred']
        
        # Map intent ke kategori dengan deteksi keyword
        category = self.map_intent_to_category(intent, query)
        
        # Query database
        items = self.db.get_items_by_filter(
            tema=slots.get('tema'),
            lokasi=slots.get('lokasi'),
            budget_min=slots.get('budget_min'),
            budget_max=slots.get('budget_max'),
            category=category
        )
        
        if verbose:
            print(f"\n🔍 Query Parameters:")
            print(f"   • Intent: {intent}")
            print(f"   • Category Filter: {category if category else 'All Categories'}")
            print(f"   • Tema: {slots.get('tema', 'N/A')}")
            print(f"   • Lokasi: {slots.get('lokasi', 'N/A')}")
            if slots.get('budget_min') or slots.get('budget_max'):
                budget_str = ""
                if slots.get('budget_min'):
                    budget_str += f"Rp {slots.get('budget_min'):,}"
                if slots.get('budget_max'):
                    if budget_str:
                        budget_str += f" - Rp {slots.get('budget_max'):,}"
                    else:
                        budget_str = f"max Rp {slots.get('budget_max'):,}"
                print(f"   • Budget: {budget_str}")
            
            print(f"\n📦 Query Result: {len(items)} items found")
        
        # Step 3: K-Means Clustering & Recommendation
        if verbose:
            print("\n" + "="*80)
            print("STEP 3: K-MEANS CLUSTERING & RECOMMENDATION")
            print("="*80)
        
        if len(items) == 0:
            if verbose:
                print("\n⚠️  Tidak ada item yang ditemukan dengan kriteria tersebut.")
                print("   Saran: Coba perluas budget atau ubah filter pencarian.")
            
            result = {
                "ai_result": ai_result,
                "items_found": 0,
                "clustering_result": {
                    "total_items": 0,
                    "clusters": [],
                    "recommendations": [],
                    "message": "Tidak ada item yang ditemukan"
                }
            }
        else:
            # Lakukan clustering
            clustering_result = self.recommendation_engine.cluster_items(items, slots)
            
            if verbose:
                print(self.recommendation_engine.format_recommendation_output(clustering_result, slots))
            
            result = {
                "ai_result": ai_result,
                "items_found": len(items),
                "clustering_result": clustering_result
            }
        
        return result
    
    def run_interactive(self):
        """Menjalankan mode interaktif (CLI)"""
        print("="*80)
        print("🎊 WEDDING RECOMMENDATION SYSTEM 🎊")
        print("="*80)
        print("\nSistem ini menggunakan:")
        print("  1. AI untuk memahami intent dan mengekstrak informasi")
        print("  2. Database MySQL untuk mencari item yang sesuai")
        print("  3. K-Means Clustering untuk merekomendasikan yang terbaik")
        print("\nKetik 'exit' atau 'quit' untuk keluar")
        print("="*80)
        
        while True:
            try:
                query = input('\n> ').strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Terima kasih telah menggunakan Wedding Recommendation System!")
                break
            
            if not query:
                continue
                
            if query.lower() in {'exit', 'quit', 'keluar'}:
                print("\n👋 Terima kasih telah menggunakan Wedding Recommendation System!")
                break
            
            try:
                result = self.process_query(query, verbose=True)
                
                # Tampilkan ringkasan
                print("\n" + "="*80)
                print("📊 RINGKASAN")
                print("="*80)
                print(f"✓ Intent detected: {result['ai_result']['intent_pred']}")
                print(f"✓ Items found: {result['items_found']}")
                print(f"✓ Recommendations: {len(result['clustering_result']['recommendations'])}")
                print("="*80)
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    def close(self):
        """Menutup koneksi database"""
        if self.db:
            self.db.disconnect()


def main():
    """Fungsi main untuk menjalankan aplikasi"""
    try:
        # Inisialisasi sistem
        system = WeddingRecommendationSystem(
            model_dir="models/local_transformer_intent",
            n_clusters=3
        )
        
        # Jalankan mode interaktif
        system.run_interactive()
        
        # Cleanup
        system.close()
        
    except KeyboardInterrupt:
        print("\n\n👋 Program dihentikan oleh user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
