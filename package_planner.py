"""
Wedding Package Planner
Modul untuk merancang paket pernikahan lengkap dengan multi-kategori
"""

from typing import Dict, List, Any
from conn import DatabaseConnection
from recommendation import RecommendationEngine
import numpy as np


class WeddingPackagePlanner:
    """Class untuk merancang paket pernikahan lengkap"""
    
    # Kategori yang diperlukan untuk paket pernikahan lengkap
    ESSENTIAL_CATEGORIES = [
        "Venue",           # Tempat pernikahan
        "Catering",        # Makanan dan minuman
        "Decoration",      # Dekorasi
        "MUA",            # Make Up Artist
        "Docummentation",  # Foto & Video
    ]
    
    # Kategori opsional tambahan
    OPTIONAL_CATEGORIES = [
        "Attire",                  # Busana pengantin
        "Entertainment",           # Hiburan
        "Master of Ceremony",      # MC
        "Traditional Ceremony"     # Upacara adat
    ]
    
    def __init__(self, db_connection: DatabaseConnection):
        """
        Inisialisasi package planner
        
        Args:
            db_connection: Koneksi database
        """
        self.db = db_connection
        self.recommendation_engine = RecommendationEngine(n_clusters=3)
    
    def detect_package_request(self, query_text: str, intent: str) -> bool:
        """
        Deteksi apakah query meminta paket pernikahan lengkap
        
        Args:
            query_text: Query text dari user
            intent: Intent yang diklasifikasi
            
        Returns:
            True jika query meminta paket lengkap
        """
        package_keywords = [
            'paket pernikahan',
            'paket lengkap',
            'rekomendasi pernikahan',
            'acara pernikahan',
            'pernikahan adat',
            'rancang pernikahan',
            'design pernikahan',
            'planning pernikahan',
            'konsep pernikahan'
        ]
        
        package_intents = [
            'cari_rekomendasi_paket',
            'estimasi_budget'
        ]
        
        query_lower = query_text.lower()
        
        # Check keywords
        if any(keyword in query_lower for keyword in package_keywords):
            return True
        
        # Check intent
        if intent in package_intents:
            return True
        
        return False
    
    def get_category_items(self, category: str, slots: Dict, limit: int = 5) -> List[Dict]:
        """
        Dapatkan item untuk kategori tertentu
        
        Args:
            category: Nama kategori
            slots: Slot yang diekstrak dari AI
            limit: Jumlah maksimal item yang diambil
            
        Returns:
            List of items untuk kategori tersebut
        """
        # Query database untuk kategori ini
        items = self.db.get_items_by_filter(
            tema=slots.get('tema'),
            lokasi=slots.get('lokasi'),
            budget_min=None,  # Tidak filter budget per kategori
            budget_max=None,
            category=category,
            flexible=True
        )
        
        if not items:
            return []
        
        # Hitung relevance score
        for item in items:
            item['relevance_score'] = self.recommendation_engine.calculate_relevance_score(item, slots)
        
        # Sort by relevance dan harga
        items_sorted = sorted(items, key=lambda x: (x['relevance_score'], -x.get('price', 0)), reverse=True)
        
        return items_sorted[:limit]
    
    def calculate_package_price_ranges(self, package_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Hitung range harga untuk paket
        
        Args:
            package_data: Dict berisi items per kategori
            
        Returns:
            Dict berisi informasi harga paket
        """
        # Hitung total harga minimum (ambil item termurah per kategori)
        min_total = 0
        max_total = 0
        avg_total = 0
        
        price_breakdown = {}
        
        for category, items in package_data.items():
            if not items:
                continue
            
            prices = [item.get('price', 0) for item in items]
            
            category_min = min(prices)
            category_max = max(prices)
            category_avg = sum(prices) / len(prices)
            
            price_breakdown[category] = {
                'min': category_min,
                'max': category_max,
                'avg': int(category_avg),
                'count': len(items)
            }
            
            min_total += category_min
            max_total += category_max
            avg_total += category_avg
        
        return {
            'min_package_price': int(min_total),
            'max_package_price': int(max_total),
            'avg_package_price': int(avg_total),
            'breakdown': price_breakdown
        }
    
    def create_package_recommendations(self, slots: Dict, include_optional: bool = True) -> Dict[str, Any]:
        """
        Buat rekomendasi paket pernikahan lengkap
        
        Args:
            slots: Slot yang diekstrak dari AI (tema, lokasi, budget, dll)
            include_optional: Include kategori opsional atau tidak
            
        Returns:
            Dict berisi paket rekomendasi per kategori
        """
        package_data = {}
        categories_to_process = self.ESSENTIAL_CATEGORIES.copy()
        
        # Tambahkan kategori opsional jika diminta
        if include_optional:
            # Deteksi kategori opsional yang mungkin diperlukan
            tema = slots.get('tema', '').lower() if slots.get('tema') else ''
            
            # Jika tema adat, tambahkan Traditional Ceremony
            if any(keyword in tema for keyword in ['adat', 'tradisional', 'jawa', 'sunda', 'betawi', 'bali']):
                if "Traditional Ceremony" not in categories_to_process:
                    categories_to_process.append("Traditional Ceremony")
            
            # Selalu tambahkan kategori opsional lainnya
            for opt_cat in self.OPTIONAL_CATEGORIES:
                if opt_cat not in categories_to_process:
                    categories_to_process.append(opt_cat)
        
        print(f"\n🎨 Merancang paket pernikahan dengan {len(categories_to_process)} kategori...")
        
        # Ambil items untuk setiap kategori
        for category in categories_to_process:
            print(f"   🔍 Mencari {category}...")
            items = self.get_category_items(category, slots, limit=3)  # Top 3 per kategori
            
            if items:
                package_data[category] = items
                print(f"      ✓ Ditemukan {len(items)} pilihan")
            else:
                print(f"      ⚠️  Tidak ada data untuk kategori ini")
        
        # Hitung range harga paket
        price_info = self.calculate_package_price_ranges(package_data)
        
        # Cek apakah sesuai budget user
        user_budget_min = slots.get('budget_min') or 0
        user_budget_max = slots.get('budget_max')
        
        budget_status = "unknown"
        
        # Only check budget if user specified max budget
        if user_budget_max is not None:
            if price_info['min_package_price'] <= user_budget_max:
                if price_info['max_package_price'] <= user_budget_max:
                    budget_status = "fully_affordable"  # Semua pilihan terjangkau
                else:
                    budget_status = "partially_affordable"  # Beberapa pilihan terjangkau
            else:
                budget_status = "over_budget"  # Di atas budget
        elif user_budget_min > 0:
            # Jika hanya ada min budget
            if price_info['min_package_price'] >= user_budget_min:
                budget_status = "within_range"
            else:
                budget_status = "below_minimum"
        
        # Buat package tiers (budget, standard, premium)
        package_tiers = self._create_package_tiers(package_data)
        
        return {
            'categories': package_data,
            'categories_found': list(package_data.keys()),
            'categories_missing': [cat for cat in self.ESSENTIAL_CATEGORIES if cat not in package_data],
            'price_info': price_info,
            'budget_status': budget_status,
            'user_budget': {
                'min': user_budget_min,
                'max': user_budget_max
            },
            'package_tiers': package_tiers,
            'total_categories': len(categories_to_process),
            'message': self._generate_package_message(package_data, price_info, budget_status)
        }
    
    def _create_package_tiers(self, package_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Buat 3 tier paket: Budget, Standard, Premium
        
        Args:
            package_data: Data items per kategori
            
        Returns:
            Dict berisi 3 tier paket
        """
        tiers = {
            'budget': {'items': {}, 'total_price': 0},
            'standard': {'items': {}, 'total_price': 0},
            'premium': {'items': {}, 'total_price': 0}
        }
        
        for category, items in package_data.items():
            if not items:
                continue
            
            # Sort by price
            items_sorted = sorted(items, key=lambda x: x.get('price', 0))
            
            # Budget tier: pilih termurah
            tiers['budget']['items'][category] = items_sorted[0]
            tiers['budget']['total_price'] += items_sorted[0].get('price', 0)
            
            # Standard tier: pilih tengah atau yang paling relevan
            mid_idx = len(items_sorted) // 2
            tiers['standard']['items'][category] = items_sorted[mid_idx] if len(items_sorted) > 1 else items_sorted[0]
            tiers['standard']['total_price'] += tiers['standard']['items'][category].get('price', 0)
            
            # Premium tier: pilih termahal
            tiers['premium']['items'][category] = items_sorted[-1]
            tiers['premium']['total_price'] += items_sorted[-1].get('price', 0)
        
        return tiers
    
    def _generate_package_message(self, package_data: Dict, price_info: Dict, budget_status: str) -> str:
        """Generate pesan deskriptif tentang paket"""
        total_categories = len(package_data)
        
        if total_categories == 0:
            return "⚠️  Tidak dapat membuat paket pernikahan karena tidak ada data yang sesuai"
        
        msg = f"✅ Berhasil merancang paket pernikahan dengan {total_categories} kategori. "
        
        if budget_status == "fully_affordable":
            msg += "Semua pilihan sesuai dengan budget Anda! "
        elif budget_status == "partially_affordable":
            msg += "Beberapa pilihan sesuai budget, Anda bisa mix and match! "
        elif budget_status == "over_budget":
            msg += "Paket melebihi budget, pertimbangkan untuk menyesuaikan pilihan. "
        elif budget_status == "within_range":
            msg += "Paket sesuai dengan minimum budget yang diinginkan. "
        elif budget_status == "below_minimum":
            msg += "Ada pilihan di bawah budget minimum yang Anda tentukan. "
        else:
            msg += "Pilih paket yang sesuai dengan budget Anda. "
        
        msg += f"Range harga paket: Rp {price_info['min_package_price']:,} - Rp {price_info['max_package_price']:,}"
        
        return msg
    
    def format_package_output(self, result: Dict, slots: Dict) -> str:
        """
        Format hasil paket untuk ditampilkan
        
        Args:
            result: Hasil dari create_package_recommendations
            slots: Slot dari AI
            
        Returns:
            String terformat
        """
        output = []
        output.append("\n" + "="*80)
        output.append("🎊 RANCANGAN PAKET PERNIKAHAN LENGKAP 🎊")
        output.append("="*80)
        
        # Info request
        output.append("\n📋 DETAIL REQUEST:")
        if slots.get('tema'):
            output.append(f"   • Tema: {slots['tema']}")
        if slots.get('lokasi'):
            output.append(f"   • Lokasi: {slots['lokasi']}")
        if slots.get('budget_min') or slots.get('budget_max'):
            budget_str = ""
            if slots.get('budget_min'):
                budget_str += f"Rp {slots['budget_min']:,}"
            if slots.get('budget_max'):
                if budget_str:
                    budget_str += f" - Rp {slots['budget_max']:,}"
                else:
                    budget_str += f"max Rp {slots['budget_max']:,}"
            output.append(f"   • Budget: {budget_str}")
        if slots.get('jumlah_tamu'):
            output.append(f"   • Jumlah Tamu: {slots['jumlah_tamu']} orang")
        
        # Summary
        output.append(f"\n📊 RINGKASAN PAKET:")
        output.append(f"   • Kategori ditemukan: {len(result['categories_found'])}/{result['total_categories']}")
        if result['categories_missing']:
            output.append(f"   • Kategori tidak tersedia: {', '.join(result['categories_missing'])}")
        
        price_info = result['price_info']
        output.append(f"\n💰 ESTIMASI BIAYA PAKET:")
        output.append(f"   • Paket Minimum: Rp {price_info['min_package_price']:,}")
        output.append(f"   • Paket Maksimum: Rp {price_info['max_package_price']:,}")
        output.append(f"   • Paket Rata-rata: Rp {price_info['avg_package_price']:,}")
        
        # Budget status
        budget_status = result['budget_status']
        if budget_status == "fully_affordable":
            output.append(f"   ✅ Status: Semua pilihan terjangkau dengan budget Anda!")
        elif budget_status == "partially_affordable":
            output.append(f"   ⚠️  Status: Beberapa pilihan sesuai budget")
        elif budget_status == "over_budget":
            output.append(f"   ❌ Status: Paket melebihi budget yang ditentukan")
        elif budget_status == "within_range":
            output.append(f"   ✅ Status: Paket sesuai dengan minimum budget")
        elif budget_status == "below_minimum":
            output.append(f"   ⚠️  Status: Ada pilihan di bawah budget minimum")
        else:
            output.append(f"   ℹ️  Status: Budget tidak ditentukan, silakan pilih sesuai kemampuan")
        
        # Breakdown per kategori
        output.append(f"\n📦 BREAKDOWN HARGA PER KATEGORI:")
        for category, info in price_info['breakdown'].items():
            output.append(f"\n   {category}:")
            output.append(f"      • Pilihan tersedia: {info['count']}")
            output.append(f"      • Range harga: Rp {info['min']:,} - Rp {info['max']:,}")
            output.append(f"      • Harga rata-rata: Rp {info['avg']:,}")
        
        # Package Tiers
        output.append(f"\n🎯 PAKET REKOMENDASI (3 TIER):")
        
        tiers = result['package_tiers']
        tier_names = {
            'budget': '💚 PAKET BUDGET',
            'standard': '💙 PAKET STANDARD',
            'premium': '💛 PAKET PREMIUM'
        }
        
        for tier_key, tier_name in tier_names.items():
            tier_data = tiers[tier_key]
            output.append(f"\n{tier_name} - Total: Rp {tier_data['total_price']:,}")
            output.append("-"*80)
            
            for category, item in tier_data['items'].items():
                output.append(f"\n   [{category}] {item.get('name', 'N/A')}")
                output.append(f"      • Vendor: {item.get('vendor', 'N/A')}")
                output.append(f"      • Harga: Rp {item.get('price', 0):,}")
                if item.get('relevance_score'):
                    output.append(f"      • Relevance: {item['relevance_score']:.2f}")
        
        # Detail semua pilihan per kategori
        output.append(f"\n\n📋 DETAIL SEMUA PILIHAN PER KATEGORI:")
        output.append("="*80)
        
        for category, items in result['categories'].items():
            output.append(f"\n{'='*80}")
            output.append(f"📂 {category.upper()} ({len(items)} pilihan)")
            output.append('='*80)
            
            for idx, item in enumerate(items, 1):
                output.append(f"\n{idx}. {item.get('name', 'N/A')}")
                output.append(f"   • Vendor: {item.get('vendor', 'N/A')}")
                output.append(f"   • Harga: Rp {item.get('price', 0):,}")
                output.append(f"   • Relevance Score: {item.get('relevance_score', 0):.2f}")
                if item.get('description'):
                    desc = item['description'][:150] + "..." if len(item['description']) > 150 else item['description']
                    output.append(f"   • Deskripsi: {desc}")
        
        output.append("\n" + "="*80)
        output.append(result['message'])
        output.append("="*80 + "\n")
        
        return "\n".join(output)


if __name__ == "__main__":
    # Test package planner
    from conn import get_db_connection
    
    print("=== Testing Wedding Package Planner ===")
    
    db = get_db_connection()
    if not db:
        print("❌ Gagal connect ke database")
        exit(1)
    
    planner = WeddingPackagePlanner(db)
    
    # Test query
    test_slots = {
        'tema': 'sunda',
        'lokasi': 'bandung',
        'budget_min': 100000000,
        'budget_max': 200000000,
        'jumlah_tamu': 500
    }
    
    print("\n🎊 Testing paket pernikahan adat sunda di bandung...")
    result = planner.create_package_recommendations(test_slots, include_optional=True)
    
    print(planner.format_package_output(result, test_slots))
