"""Quick test for package planner with budget"""
from package_planner import WeddingPackagePlanner
from conn import get_db_connection

db = get_db_connection()
planner = WeddingPackagePlanner(db)

# Test WITH budget
slots = {
    'tema': 'sunda',
    'lokasi': 'bandung',
    'budget_min': 100000000,
    'budget_max': 200000000,
    'jumlah_tamu': 500
}

print("\n🧪 Testing package planner WITH budget (100-200 juta)...")
result = planner.create_package_recommendations(slots, include_optional=True)
print(f"\n✅ SUCCESS!")
print(f"Message: {result['message']}")
print(f"Budget Status: {result['budget_status']}")
print(f"Categories Found: {len(result['categories_found'])}")
print(f"Price Range: Rp {result['price_info']['min_package_price']:,} - Rp {result['price_info']['max_package_price']:,}")
print(f"User Budget: Rp {result['user_budget']['min']:,} - Rp {result['user_budget']['max']:,}")
