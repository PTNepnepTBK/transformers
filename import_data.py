"""
Script untuk import data tambahan ke database dan cek isi database
"""

from conn import get_db_connection

def check_database():
    """Cek isi database saat ini"""
    db = get_db_connection()
    
    if not db:
        print("❌ Gagal terhubung ke database")
        return
    
    print("\n" + "="*80)
    print("CEK ISI DATABASE")
    print("="*80)
    
    # Total items
    result = db.execute_query("SELECT COUNT(*) as total FROM items")
    if result:
        print(f"\n📊 Total items: {result[0]['total']}")
    
    # Sample data sunda + bandung
    print("\n🔍 Sample items dengan 'sunda' DAN 'bandung':")
    query = """
        SELECT name, vendor, price
        FROM items
        WHERE (name LIKE %s OR description LIKE %s OR vendor LIKE %s)
        AND (vendor LIKE %s OR description LIKE %s)
        AND status = 'ongoing'
        LIMIT 10
    """
    results = db.execute_query(query, ('%sunda%', '%sunda%', '%sunda%', '%bandung%', '%bandung%'))
    if results:
        for row in results:
            print(f"   • {row['name']} - Rp {row['price']:,}")
    else:
        print("   ⚠️  Tidak ada data ditemukan!")
    
    # Range harga
    print("\n💰 Range harga items:")
    result = db.execute_query("""
        SELECT 
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(price) as avg_price
        FROM items
    """)
    if result and result[0]['min_price']:
        row = result[0]
        print(f"   • Min: Rp {row['min_price']:,}")
        print(f"   • Max: Rp {row['max_price']:,}")
        print(f"   • Avg: Rp {int(row['avg_price']):,}")
    
    db.disconnect()


def import_additional_data():
    """Import data tambahan dari file SQL"""
    db = get_db_connection()
    
    if not db:
        print("❌ Gagal terhubung ke database")
        return False
    
    try:
        print("\n" + "="*80)
        print("IMPORT DATA TAMBAHAN")
        print("="*80)
        
        # Read SQL file
        with open('insert_additional_data.sql', 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Split by INSERT statements
        import re
        insert_statements = re.findall(r'INSERT INTO.*?;', sql_content, re.DOTALL | re.IGNORECASE)
        
        success_count = 0
        error_count = 0
        
        print(f"\n📦 Ditemukan {len(insert_statements)} INSERT statements...")
        
        for i, statement in enumerate(insert_statements, 1):
            try:
                cursor = db.connection.cursor()
                cursor.execute(statement)
                db.connection.commit()
                cursor.close()
                success_count += 1
                
                if i % 20 == 0:
                    print(f"   ✓ Progress: {i}/{len(insert_statements)} statements")
                    
            except Exception as e:
                error_count += 1
                if "Duplicate entry" not in str(e):
                    print(f"   ⚠️  Error pada statement {i}: {str(e)[:100]}")
        
        print(f"\n✅ Import selesai!")
        print(f"   • Sukses: {success_count} statements")
        print(f"   • Error: {error_count} statements")
        
        db.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        db.disconnect()
        return False


if __name__ == "__main__":
    print("="*80)
    print("DATABASE MANAGEMENT TOOL")
    print("="*80)
    print("\n1. Cek isi database")
    print("2. Import data tambahan")
    print("3. Keduanya (cek -> import -> cek lagi)")
    
    choice = input("\nPilih (1/2/3): ").strip()
    
    if choice == "1":
        check_database()
    elif choice == "2":
        import_additional_data()
        check_database()
    elif choice == "3":
        print("\n=== CEK DATABASE SEBELUM IMPORT ===")
        check_database()
        import_additional_data()
        print("\n=== CEK DATABASE SETELAH IMPORT ===")
        check_database()
    else:
        print("Pilihan tidak valid")
    
    print("\n" + "="*80)
    print("CONTOH PERTANYAAN YANG BISA DICOBA DI main.py:")
    print("="*80)
    print("\n1. rekomendasi pernikahan adat sunda di bandung")
    print("2. cari venue pernikahan di bandung budget 40 juta")
    print("3. paket catering sunda untuk 500 orang di bandung")
    print("4. dekorasi pernikahan adat sunda bandung budget 25-35 juta")
    print("5. MUA pengantin sunda di bandung")
    print("6. dokumentasi foto video pernikahan sunda bandung")
    print("7. sewa kebaya sunda untuk pengantin di bandung")
    print("8. band akustik sunda untuk pernikahan di bandung")
    print("9. MC pernikahan adat sunda di bandung budget 3-4 juta")
    print("10. upacara adat sunda lengkap di bandung")
    print("\n" + "="*80)
