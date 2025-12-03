"""Quick test for package planner without budget"""
from package_planner import WeddingPackagePlanner
from conn import get_db_connection

db = get_db_connection()
planner = WeddingPackagePlanner(db)

# Test without budget
slots = {
    'tema': 'sunda',
    'lokasi': 'bandung',
    'budget_min': None,
    'budget_max': None,
    'jumlah_tamu': None
}

print("\n🧪 Testing package planner WITHOUT budget...")
result = planner.create_package_recommendations(slots, include_optional=True)
print(f"\n✅ SUCCESS!")
print(f"Message: {result['message']}")
print(f"Budget Status: {result['budget_status']}")
print(f"Categories Found: {len(result['categories_found'])}")
print(f"Price Range: Rp {result['price_info']['min_package_price']:,} - Rp {result['price_info']['max_package_price']:,}")
