"""
Database Connection Module
Modul untuk koneksi ke database MySQL/MariaDB phpMyAdmin
"""

import mysql.connector
from mysql.connector import Error
import os

class DatabaseConnection:
    """Class untuk mengelola koneksi database"""
    
    def __init__(self, host='localhost', database='sepasangwp', user='root', password=''):
        """
        Inisialisasi koneksi database
        
        Args:
            host: Host database (default: localhost)
            database: Nama database (default: sepasangwp)
            user: Username database (default: root)
            password: Password database (default: '')
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
    
    def connect(self):
        """Membuat koneksi ke database"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            if self.connection.is_connected():
                db_info = self.connection.get_server_info()
                print(f"✓ Berhasil terhubung ke MySQL Server versi {db_info}")
                cursor = self.connection.cursor()
                cursor.execute("select database();")
                record = cursor.fetchone()
                print(f"✓ Terhubung ke database: {record[0]}")
                cursor.close()
                return True
        except Error as e:
            print(f"✗ Error saat koneksi ke MySQL: {e}")
            return False
    
    def disconnect(self):
        """Menutup koneksi database"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("✓ Koneksi MySQL ditutup")
    
    def execute_query(self, query, params=None):
        """
        Eksekusi query SELECT dan mengembalikan hasil
        
        Args:
            query: SQL query string
            params: Parameter untuk prepared statement (optional)
            
        Returns:
            List of tuples berisi hasil query
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            print(f"✗ Error saat eksekusi query: {e}")
            return []
    
    def get_items_by_filter(self, tema=None, lokasi=None, budget_min=None, budget_max=None, category=None, flexible=True):
        """
        Mengambil items dari database berdasarkan filter dengan mode flexible
        
        Args:
            tema: Tema pernikahan (optional)
            lokasi: Lokasi (optional)
            budget_min: Budget minimum (optional)
            budget_max: Budget maximum (optional)
            category: Nama kategori (optional)
            flexible: Jika True, relax filter jika tidak ada hasil (default: True)
            
        Returns:
            List of dict berisi data items
        """
        query = """
            SELECT 
                i.id, i.name, i.vendor, i.description, i.price, i.image_url,
                c.name as category_name
            FROM items i
            JOIN categories c ON i.category_id = c.id
            WHERE i.status = 'ongoing'
        """
        
        conditions = []
        params = []
        
        # Filter berdasarkan budget (dengan toleransi 20% jika flexible)
        if budget_min is not None:
            if flexible:
                # Toleransi 20% ke bawah
                adjusted_min = budget_min * 0.8
                conditions.append("i.price >= %s")
                params.append(adjusted_min)
            else:
                conditions.append("i.price >= %s")
                params.append(budget_min)
        
        if budget_max is not None:
            if flexible:
                # Toleransi 20% ke atas
                adjusted_max = budget_max * 1.2
                conditions.append("i.price <= %s")
                params.append(adjusted_max)
            else:
                conditions.append("i.price <= %s")
                params.append(budget_max)
        
        # Filter berdasarkan kategori
        if category:
            conditions.append("c.name LIKE %s")
            params.append(f"%{category}%")
        
        # Filter berdasarkan tema (search di description atau name)
        if tema:
            conditions.append("(i.name LIKE %s OR i.description LIKE %s OR i.vendor LIKE %s)")
            params.extend([f"%{tema}%", f"%{tema}%", f"%{tema}%"])
        
        # Filter berdasarkan lokasi (search di vendor atau description)
        if lokasi:
            conditions.append("(i.vendor LIKE %s OR i.description LIKE %s)")
            params.extend([f"%{lokasi}%", f"%{lokasi}%"])
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += " ORDER BY i.price ASC"
        
        results = self.execute_query(query, params if params else None)
        
        # Jika tidak ada hasil dan flexible mode, coba relax filter
        if not results and flexible and (tema or lokasi or budget_min or budget_max):
            print("   🔄 Tidak ada hasil dengan filter ketat, mencoba filter yang lebih fleksibel...")
            
            # Coba tanpa budget constraint
            if budget_min or budget_max:
                return self.get_items_by_filter(tema, lokasi, None, None, category, flexible=False)
            
            # Coba tanpa lokasi
            elif lokasi:
                return self.get_items_by_filter(tema, None, budget_min, budget_max, category, flexible=False)
            
            # Coba tanpa tema
            elif tema:
                return self.get_items_by_filter(None, lokasi, budget_min, budget_max, category, flexible=False)
        
        return results
    
    def get_all_categories(self):
        """Mengambil semua kategori dari database"""
        query = "SELECT id, name, description FROM categories ORDER BY name"
        return self.execute_query(query)
    
    def get_venue_by_capacity(self, min_capacity=0):
        """
        Mengambil venue berdasarkan kapasitas minimum
        
        Args:
            min_capacity: Kapasitas minimum yang dibutuhkan
            
        Returns:
            List of dict berisi data venue
        """
        query = """
            SELECT 
                i.id, i.name, i.vendor, i.description, i.price, i.image_url
            FROM items i
            JOIN categories c ON i.category_id = c.id
            WHERE c.name = 'Venue' AND i.status = 'ongoing'
        """
        
        # Mencari di description yang menyebut kapasitas
        if min_capacity > 0:
            query += f" AND (i.description LIKE '%{min_capacity}%' OR i.description LIKE '%tamu%')"
        
        query += " ORDER BY i.price ASC"
        
        return self.execute_query(query)


# Fungsi helper untuk koneksi cepat
def get_db_connection():
    """
    Fungsi helper untuk mendapatkan koneksi database
    
    Returns:
        DatabaseConnection object
    """
    db = DatabaseConnection()
    if db.connect():
        return db
    return None


if __name__ == "__main__":
    # Test koneksi
    print("=== Testing Database Connection ===")
    db = get_db_connection()
    
    if db:
        print("\n=== Testing Get All Categories ===")
        categories = db.get_all_categories()
        for cat in categories[:5]:  # Tampilkan 5 kategori pertama
            print(f"- {cat['name']}: {cat['description']}")
        
        print("\n=== Testing Get Items by Filter ===")
        items = db.get_items_by_filter(budget_min=1000000, budget_max=5000000, category="MUA")
        print(f"Ditemukan {len(items)} item MUA dengan budget 1-5 juta:")
        for item in items[:3]:  # Tampilkan 3 item pertama
            print(f"- {item['name']} ({item['vendor']}): Rp {item['price']:,}")
        
        db.disconnect()
