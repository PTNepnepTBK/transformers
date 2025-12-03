"""
Database Expander - Generate Wedding Items Data
Generate data untuk database dengan tema Jawa, Sunda, dan Betawi
Mencakup semua 9 kategori: Venue, Catering, Decoration, MUA, Docummentation, Attire, Entertainment, MC, Traditional Ceremony
"""

import random
import json
from datetime import datetime


class WeddingDataGenerator:
    """Generator untuk data pernikahan dengan berbagai tema"""
    
    # Lokasi untuk setiap tema
    LOCATIONS = {
        'sunda': ['Bandung', 'Bogor', 'Sukabumi', 'Cirebon', 'Tasikmalaya', 'Garut'],
        'jawa': ['Yogyakarta', 'Solo', 'Semarang', 'Surabaya', 'Malang', 'Magelang'],
        'betawi': ['Jakarta Pusat', 'Jakarta Selatan', 'Jakarta Barat', 'Jakarta Timur', 'Jakarta Utara', 'Tangerang']
    }
    
    # Category IDs (sesuai database)
    CATEGORY_IDS = {
        'Venue': '3c476203-a266-42c4-b2f5-3cdd7efcf510',
        'Catering': '33126b96-7f71-4141-b9d8-999ef4fd0cb8',
        'Decoration': 'b01d482d-7301-44d4-80c1-155f1866a604',
        'MUA': '1b2f2a98-af5e-44f2-9d7e-7c26b8b62f6a',
        'Docummentation': 'b4c13e15-afcf-4e9e-a0c7-5d82c5499742',
        'Attire': 'ae51cb01-edd5-4ed0-8889-ce49d7a5dfdf',
        'Entertainment': 'bef9f40f-9c48-4c2a-9849-9e0cf0e8c5d9',
        'Master of Ceremony': 'b123275f-6423-4a49-99a1-0a4e0c0e39c0',
        'Traditional Ceremony': 'c5b14377-e5e8-4c2a-95d6-3e6f3f3e5e5e'
    }
    
    def __init__(self):
        """Inisialisasi generator"""
        self.generated_items = []
    
    def generate_uuid(self):
        """Generate simple UUID"""
        import uuid
        return str(uuid.uuid4())
    
    # ==================== VENUE ====================
    def generate_venues(self, tema: str, count: int = 15) -> list:
        """Generate data venue"""
        venues = []
        locations = self.LOCATIONS[tema]
        
        venue_types = {
            'sunda': [
                ('Ballroom Hotel', 'ballroom mewah dengan nuansa sunda', (60000000, 120000000)),
                ('Gedung Pernikahan', 'gedung luas untuk resepsi adat sunda', (40000000, 80000000)),
                ('Villa Outdoor', 'villa dengan pemandangan gunung tema sunda', (35000000, 70000000)),
                ('Garden Venue', 'taman outdoor untuk pernikahan sunda', (30000000, 60000000)),
                ('Pendopo Adat', 'pendopo tradisional sunda', (25000000, 50000000)),
            ],
            'jawa': [
                ('Ballroom Hotel', 'ballroom elegan dengan sentuhan jawa', (65000000, 130000000)),
                ('Gedung Pernikahan', 'gedung resepsi adat jawa', (45000000, 85000000)),
                ('Pendopo Joglo', 'pendopo joglo tradisional jawa', (35000000, 70000000)),
                ('Outdoor Garden', 'taman outdoor tema jawa', (30000000, 65000000)),
                ('Griya Jawa', 'rumah jawa untuk intimate wedding', (28000000, 55000000)),
            ],
            'betawi': [
                ('Ballroom Hotel', 'ballroom modern dengan aksen betawi', (70000000, 140000000)),
                ('Gedung Betawi', 'gedung resepsi adat betawi', (50000000, 90000000)),
                ('Rumah Betawi', 'rumah betawi klasik untuk pernikahan', (40000000, 75000000)),
                ('Garden Venue', 'taman outdoor tema betawi', (35000000, 70000000)),
                ('Balai Betawi', 'balai pertemuan tradisional betawi', (30000000, 60000000)),
            ]
        }
        
        for i in range(count):
            vtype, desc, price_range = random.choice(venue_types[tema])
            location = random.choice(locations)
            capacity = random.choice([300, 400, 500, 600, 700, 800, 1000])
            price = random.randint(price_range[0], price_range[1])
            
            venue = {
                'id': self.generate_uuid(),
                'category_id': self.CATEGORY_IDS['Venue'],
                'name': f'{vtype} {tema.capitalize()} {location} #{i+1}',
                'description': f'{desc}, kapasitas {capacity} tamu, lokasi strategis di {location}',
                'vendor': f'{location} Venue & Hall',
                'price': price,
                'status': 'ongoing'
            }
            venues.append(venue)
        
        return venues
    
    # ==================== CATERING ====================
    def generate_catering(self, tema: str, count: int = 15) -> list:
        """Generate data catering"""
        catering_items = []
        locations = self.LOCATIONS[tema]
        
        menu_types = {
            'sunda': [
                ('Paket Prasmanan Sunda', 'nasi timbel, ayam goreng, pepes ikan, sayur asem, sambal', (80000, 150000)),
                ('Paket Nasi Kotak Sunda', 'nasi putih, ayam bakar, tahu goreng, lalapan, sambal', (50000, 90000)),
                ('Paket Gubukan Sunda', 'nasi liwet, ayam pop, pepes tahu, karedok', (70000, 130000)),
                ('Paket Buffet Sunda Premium', 'menu lengkap sunda modern dengan live cooking', (100000, 180000)),
                ('Paket Nasi Liwet Sunda', 'nasi liwet komplit dengan lauk pauk tradisional', (65000, 120000)),
            ],
            'jawa': [
                ('Paket Prasmanan Jawa', 'nasi putih, ayam goreng kremes, gudeg, sambal goreng', (85000, 160000)),
                ('Paket Nasi Kotak Jawa', 'nasi putih, ayam bakar bumbu jawa, tempe goreng', (55000, 95000)),
                ('Paket Gubukan Jawa', 'nasi kuning, ayam goreng, perkedel, sambal krecek', (75000, 140000)),
                ('Paket Buffet Jawa Premium', 'menu lengkap jawa klasik dengan dessert', (110000, 190000)),
                ('Paket Tumpeng Jawa', 'tumpeng lengkap dengan lauk pauk tradisional', (70000, 130000)),
            ],
            'betawi': [
                ('Paket Prasmanan Betawi', 'nasi uduk, ayam goreng, kerak telor, asinan', (90000, 170000)),
                ('Paket Nasi Kotak Betawi', 'nasi uduk, ayam goreng, semur daging, kerupuk', (60000, 100000)),
                ('Paket Gubukan Betawi', 'nasi uduk, soto betawi, kerak telor, es selendang mayang', (80000, 150000)),
                ('Paket Buffet Betawi Premium', 'menu lengkap betawi klasik modern', (120000, 200000)),
                ('Paket Nasi Uduk Betawi', 'nasi uduk komplit dengan lauk lengkap', (75000, 140000)),
            ]
        }
        
        for i in range(count):
            menu_type, menu_desc, price_per_pax = random.choice(menu_types[tema])
            location = random.choice(locations)
            pax = random.choice([300, 400, 500, 600, 800])
            price_total = random.randint(price_per_pax[0] * pax, price_per_pax[1] * pax)
            
            catering = {
                'id': self.generate_uuid(),
                'category_id': self.CATEGORY_IDS['Catering'],
                'name': f'{menu_type} untuk {pax} Pax #{i+1}',
                'description': f'{menu_desc}, melayani area {location} dan sekitarnya, harga per pax: Rp {price_per_pax[0]:,} - {price_per_pax[1]:,}',
                'vendor': f'{location} Catering & Kuliner',
                'price': price_total,
                'status': 'ongoing'
            }
            catering_items.append(catering)
        
        return catering_items
    
    # ==================== DECORATION ====================
    def generate_decoration(self, tema: str, count: int = 15) -> list:
        """Generate data dekorasi"""
        decorations = []
        locations = self.LOCATIONS[tema]
        
        decor_types = {
            'sunda': [
                ('Dekorasi Pelaminan Sunda Siger', 'pelaminan adat sunda dengan siger dan kain batik', (15000000, 30000000)),
                ('Dekorasi Minimalis Sunda', 'dekorasi modern dengan sentuhan sunda', (12000000, 25000000)),
                ('Dekorasi Full Ballroom Sunda', 'dekorasi lengkap ballroom tema sunda', (25000000, 50000000)),
                ('Dekorasi Garden Outdoor Sunda', 'dekorasi taman outdoor nuansa sunda', (18000000, 35000000)),
                ('Dekorasi Pelaminan Modern Sunda', 'pelaminan kontemporer dengan aksen sunda', (20000000, 40000000)),
            ],
            'jawa': [
                ('Dekorasi Pelaminan Jawa Klasik', 'pelaminan adat jawa dengan blangkon dan kebaya', (16000000, 32000000)),
                ('Dekorasi Minimalis Jawa', 'dekorasi modern dengan sentuhan jawa', (13000000, 27000000)),
                ('Dekorasi Full Ballroom Jawa', 'dekorasi lengkap ballroom tema jawa', (28000000, 55000000)),
                ('Dekorasi Pendopo Joglo Jawa', 'dekorasi pendopo tradisional jawa', (20000000, 38000000)),
                ('Dekorasi Pelaminan Modern Jawa', 'pelaminan kontemporer dengan aksen jawa', (22000000, 42000000)),
            ],
            'betawi': [
                ('Dekorasi Pelaminan Betawi Klasik', 'pelaminan adat betawi dengan aksen merah gold', (18000000, 35000000)),
                ('Dekorasi Minimalis Betawi', 'dekorasi modern dengan sentuhan betawi', (14000000, 28000000)),
                ('Dekorasi Full Ballroom Betawi', 'dekorasi lengkap ballroom tema betawi', (30000000, 60000000)),
                ('Dekorasi Garden Betawi', 'dekorasi taman outdoor nuansa betawi', (20000000, 40000000)),
                ('Dekorasi Pelaminan Modern Betawi', 'pelaminan kontemporer dengan aksen betawi', (24000000, 45000000)),
            ]
        }
        
        for i in range(count):
            dtype, desc, price_range = random.choice(decor_types[tema])
            location = random.choice(locations)
            price = random.randint(price_range[0], price_range[1])
            
            decoration = {
                'id': self.generate_uuid(),
                'category_id': self.CATEGORY_IDS['Decoration'],
                'name': f'{dtype} {location} #{i+1}',
                'description': f'{desc}, melayani wilayah {location}, include instalasi dan pembongkaran',
                'vendor': f'{location} Decoration & Design',
                'price': price,
                'status': 'ongoing'
            }
            decorations.append(decoration)
        
        return decorations
    
    # ==================== MUA ====================
    def generate_mua(self, tema: str, count: int = 15) -> list:
        """Generate data MUA"""
        mua_items = []
        locations = self.LOCATIONS[tema]
        
        mua_packages = {
            'sunda': [
                ('Paket MUA Pengantin Sunda Siger', 'makeup pengantin sunda dengan siger lengkap, sanggul, hijab styling', (4000000, 8000000)),
                ('Paket MUA Akad dan Resepsi Sunda', '2x makeup untuk akad dan resepsi adat sunda', (6000000, 12000000)),
                ('Paket MUA Bridesmaid Sunda', 'makeup untuk 5 bridesmaid tema sunda', (3000000, 6000000)),
                ('Paket MUA Premium Sunda', 'makeup pengantin sunda premium dengan retouch', (8000000, 15000000)),
                ('Paket MUA Simple Sunda', 'makeup pengantin sunda modern minimalis', (3500000, 7000000)),
            ],
            'jawa': [
                ('Paket MUA Pengantin Jawa Paes', 'makeup pengantin jawa dengan paes lengkap', (4500000, 9000000)),
                ('Paket MUA Akad dan Resepsi Jawa', '2x makeup untuk akad dan resepsi adat jawa', (6500000, 13000000)),
                ('Paket MUA Bridesmaid Jawa', 'makeup untuk 5 bridesmaid tema jawa', (3200000, 6500000)),
                ('Paket MUA Premium Jawa', 'makeup pengantin jawa premium dengan paes ageng', (9000000, 16000000)),
                ('Paket MUA Simple Jawa', 'makeup pengantin jawa modern minimalis', (4000000, 7500000)),
            ],
            'betawi': [
                ('Paket MUA Pengantin Betawi None Jakarta', 'makeup pengantin betawi dengan aksesoris lengkap', (5000000, 10000000)),
                ('Paket MUA Akad dan Resepsi Betawi', '2x makeup untuk akad dan resepsi adat betawi', (7000000, 14000000)),
                ('Paket MUA Bridesmaid Betawi', 'makeup untuk 5 bridesmaid tema betawi', (3500000, 7000000)),
                ('Paket MUA Premium Betawi', 'makeup pengantin betawi premium dengan retouch', (10000000, 18000000)),
                ('Paket MUA Simple Betawi', 'makeup pengantin betawi modern minimalis', (4500000, 8000000)),
            ]
        }
        
        for i in range(count):
            pkg, desc, price_range = random.choice(mua_packages[tema])
            location = random.choice(locations)
            price = random.randint(price_range[0], price_range[1])
            
            mua = {
                'id': self.generate_uuid(),
                'category_id': self.CATEGORY_IDS['MUA'],
                'name': f'{pkg} {location} #{i+1}',
                'description': f'{desc}, melayani area {location}, include makeup trial dan touch up',
                'vendor': f'{location} MUA & Beauty',
                'price': price,
                'status': 'ongoing'
            }
            mua_items.append(mua)
        
        return mua_items
    
    # ==================== DOCUMMENTATION ====================
    def generate_documentation(self, tema: str, count: int = 15) -> list:
        """Generate data dokumentasi"""
        docs = []
        locations = self.LOCATIONS[tema]
        
        doc_packages = {
            'sunda': [
                ('Paket Foto + Video Pernikahan Sunda', '1 fotografer + 1 videografer untuk akad dan resepsi, raw + edited', (8000000, 15000000)),
                ('Paket Premium Cinematic Sunda', '2 fotografer + 2 videografer + drone, video cinematic', (12000000, 25000000)),
                ('Paket Foto Pernikahan Sunda', '2 fotografer untuk dokumentasi lengkap', (5000000, 10000000)),
                ('Paket Video Cinematic Sunda', '2 videografer + drone untuk video cinematic', (7000000, 14000000)),
                ('Paket Pre-Wedding Outdoor Sunda', 'foto pre-wedding outdoor tema sunda', (4000000, 8000000)),
            ],
            'jawa': [
                ('Paket Foto + Video Pernikahan Jawa', '1 fotografer + 1 videografer untuk dokumentasi adat jawa', (8500000, 16000000)),
                ('Paket Premium Cinematic Jawa', '2 fotografer + 2 videografer + drone, dokumentasi lengkap', (13000000, 27000000)),
                ('Paket Foto Pernikahan Jawa', '2 fotografer untuk akad dan resepsi jawa', (5500000, 11000000)),
                ('Paket Video Cinematic Jawa', '2 videografer + drone untuk video wedding', (7500000, 15000000)),
                ('Paket Pre-Wedding Candi Jawa', 'foto pre-wedding di candi dan lokasi jawa', (4500000, 9000000)),
            ],
            'betawi': [
                ('Paket Foto + Video Pernikahan Betawi', '1 fotografer + 1 videografer untuk dokumentasi betawi', (9000000, 17000000)),
                ('Paket Premium Cinematic Betawi', '2 fotografer + 2 videografer + drone, full coverage', (14000000, 28000000)),
                ('Paket Foto Pernikahan Betawi', '2 fotografer untuk akad dan resepsi betawi', (6000000, 12000000)),
                ('Paket Video Cinematic Betawi', '2 videografer + drone untuk video wedding', (8000000, 16000000)),
                ('Paket Pre-Wedding Jakarta Betawi', 'foto pre-wedding di spot Jakarta', (5000000, 10000000)),
            ]
        }
        
        for i in range(count):
            pkg, desc, price_range = random.choice(doc_packages[tema])
            location = random.choice(locations)
            price = random.randint(price_range[0], price_range[1])
            
            doc = {
                'id': self.generate_uuid(),
                'category_id': self.CATEGORY_IDS['Docummentation'],
                'name': f'{pkg} {location} #{i+1}',
                'description': f'{desc}, melayani area {location}, include semua file RAW dan edited',
                'vendor': f'{location} Photo & Video',
                'price': price,
                'status': 'ongoing'
            }
            docs.append(doc)
        
        return docs
    
    # ==================== ATTIRE ====================
    def generate_attire(self, tema: str, count: int = 10) -> list:
        """Generate data busana"""
        attires = []
        locations = self.LOCATIONS[tema]
        
        attire_packages = {
            'sunda': [
                ('Paket Sewa Kebaya Sunda Pengantin', 'kebaya sunda lengkap dengan siger dan aksesoris', (5000000, 10000000)),
                ('Paket Sewa Baju Akad Sunda', 'baju akad nikah sunda couple', (3000000, 6000000)),
                ('Paket Sewa Kebaya Keluarga Sunda', 'kebaya untuk keluarga 10 orang', (8000000, 15000000)),
                ('Paket Custom Kebaya Sunda Premium', 'kebaya custom design tema sunda', (12000000, 25000000)),
            ],
            'jawa': [
                ('Paket Sewa Kebaya Jawa Pengantin', 'kebaya jawa lengkap dengan paes dan aksesoris', (5500000, 11000000)),
                ('Paket Sewa Beskap Jawa Couple', 'beskap dan kebaya jawa untuk pengantin', (4000000, 8000000)),
                ('Paket Sewa Kebaya Keluarga Jawa', 'kebaya jawa untuk keluarga 10 orang', (9000000, 17000000)),
                ('Paket Custom Kebaya Jawa Premium', 'kebaya jawa custom design', (13000000, 27000000)),
            ],
            'betawi': [
                ('Paket Sewa Kebaya Betawi Pengantin', 'kebaya encim betawi lengkap dengan aksesoris', (6000000, 12000000)),
                ('Paket Sewa Baju None Jakarta', 'baju none jakarta untuk pengantin betawi', (4500000, 9000000)),
                ('Paket Sewa Kebaya Keluarga Betawi', 'kebaya betawi untuk keluarga 10 orang', (10000000, 18000000)),
                ('Paket Custom Kebaya Betawi Premium', 'kebaya betawi custom design mewah', (15000000, 30000000)),
            ]
        }
        
        for i in range(count):
            pkg, desc, price_range = random.choice(attire_packages[tema])
            location = random.choice(locations)
            price = random.randint(price_range[0], price_range[1])
            
            attire = {
                'id': self.generate_uuid(),
                'category_id': self.CATEGORY_IDS['Attire'],
                'name': f'{pkg} {location} #{i+1}',
                'description': f'{desc}, tersedia di {location}, include fitting dan alterasi',
                'vendor': f'{location} Busana & Attire',
                'price': price,
                'status': 'ongoing'
            }
            attires.append(attire)
        
        return attires
    
    # ==================== ENTERTAINMENT ====================
    def generate_entertainment(self, tema: str, count: int = 10) -> list:
        """Generate data entertainment"""
        entertainments = []
        locations = self.LOCATIONS[tema]
        
        entertainment_packages = {
            'sunda': [
                ('Band Akustik Sunda', 'band dengan lagu-lagu sunda dan pop sunda', (5000000, 10000000)),
                ('Gamelan Sunda', 'gamelan degung untuk acara adat sunda', (8000000, 15000000)),
                ('Angklung Sunda', 'performance angklung sunda', (4000000, 8000000)),
                ('DJ + Sound System Sunda', 'DJ dengan playlist sunda modern', (6000000, 12000000)),
            ],
            'jawa': [
                ('Band Akustik Jawa', 'band dengan lagu-lagu jawa dan campursari', (5500000, 11000000)),
                ('Gamelan Jawa', 'gamelan jawa untuk acara adat', (9000000, 17000000)),
                ('Orkes Keroncong Jawa', 'orkes keroncong untuk pernikahan jawa', (7000000, 14000000)),
                ('DJ + Sound System Jawa', 'DJ dengan playlist jawa modern', (6500000, 13000000)),
            ],
            'betawi': [
                ('Band Akustik Betawi', 'band dengan lagu-lagu betawi dan gambang kromong', (6000000, 12000000)),
                ('Marawis Betawi', 'group marawis untuk acara pernikahan betawi', (5000000, 10000000)),
                ('Gambang Kromong Betawi', 'performance gambang kromong tradisional', (10000000, 20000000)),
                ('DJ + Sound System Betawi', 'DJ dengan playlist betawi modern', (7000000, 14000000)),
            ]
        }
        
        for i in range(count):
            pkg, desc, price_range = random.choice(entertainment_packages[tema])
            location = random.choice(locations)
            price = random.randint(price_range[0], price_range[1])
            
            ent = {
                'id': self.generate_uuid(),
                'category_id': self.CATEGORY_IDS['Entertainment'],
                'name': f'{pkg} {location} #{i+1}',
                'description': f'{desc}, melayani area {location}, include sound system dan lighting',
                'vendor': f'{location} Entertainment',
                'price': price,
                'status': 'ongoing'
            }
            entertainments.append(ent)
        
        return entertainments
    
    # ==================== MASTER OF CEREMONY ====================
    def generate_mc(self, tema: str, count: int = 10) -> list:
        """Generate data MC"""
        mcs = []
        locations = self.LOCATIONS[tema]
        
        mc_packages = {
            'sunda': [
                ('MC Pernikahan Sunda Single', 'MC profesional untuk acara adat sunda', (2000000, 4000000)),
                ('MC Pernikahan Sunda Couple', 'MC couple untuk akad dan resepsi sunda', (3500000, 7000000)),
                ('MC Bilingual Sunda-Inggris', 'MC bilingual untuk pernikahan sunda', (4000000, 8000000)),
            ],
            'jawa': [
                ('MC Pernikahan Jawa Single', 'MC profesional untuk acara adat jawa', (2200000, 4500000)),
                ('MC Pernikahan Jawa Couple', 'MC couple untuk akad dan resepsi jawa', (4000000, 8000000)),
                ('MC Bilingual Jawa-Inggris', 'MC bilingual untuk pernikahan jawa', (4500000, 9000000)),
            ],
            'betawi': [
                ('MC Pernikahan Betawi Single', 'MC profesional untuk acara adat betawi', (2500000, 5000000)),
                ('MC Pernikahan Betawi Couple', 'MC couple untuk akad dan resepsi betawi', (4500000, 9000000)),
                ('MC Bilingual Betawi-Inggris', 'MC bilingual untuk pernikahan betawi', (5000000, 10000000)),
            ]
        }
        
        for i in range(count):
            pkg, desc, price_range = random.choice(mc_packages[tema])
            location = random.choice(locations)
            price = random.randint(price_range[0], price_range[1])
            
            mc = {
                'id': self.generate_uuid(),
                'category_id': self.CATEGORY_IDS['Master of Ceremony'],
                'name': f'{pkg} {location} #{i+1}',
                'description': f'{desc}, berpengalaman di area {location}, include rundown dan koordinasi',
                'vendor': f'{location} MC Professional',
                'price': price,
                'status': 'ongoing'
            }
            mcs.append(mc)
        
        return mcs
    
    # ==================== TRADITIONAL CEREMONY ====================
    def generate_traditional_ceremony(self, tema: str, count: int = 10) -> list:
        """Generate data upacara adat"""
        ceremonies = []
        locations = self.LOCATIONS[tema]
        
        ceremony_packages = {
            'sunda': [
                ('Paket Upacara Ngeuyeuk Seureuh Sunda', 'upacara ngeuyeuk seureuh lengkap dengan perlengkapan', (5000000, 10000000)),
                ('Paket Siraman Sunda', 'siraman adat sunda dengan sesepuh', (4000000, 8000000)),
                ('Paket Upacara Adat Sunda Lengkap', 'paket upacara sunda dari siraman hingga seserahan', (12000000, 25000000)),
            ],
            'jawa': [
                ('Paket Upacara Siraman Jawa', 'siraman adat jawa dengan sesepuh', (4500000, 9000000)),
                ('Paket Midodareni Jawa', 'upacara midodareni lengkap', (5000000, 10000000)),
                ('Paket Upacara Adat Jawa Lengkap', 'paket upacara jawa dari siraman hingga panggih', (15000000, 30000000)),
            ],
            'betawi': [
                ('Paket Upacara Siraman Betawi', 'siraman adat betawi', (5000000, 10000000)),
                ('Paket Buka Palang Pintu Betawi', 'upacara buka palang pintu dengan silat', (6000000, 12000000)),
                ('Paket Upacara Adat Betawi Lengkap', 'paket upacara betawi lengkap', (18000000, 35000000)),
            ]
        }
        
        for i in range(count):
            pkg, desc, price_range = random.choice(ceremony_packages[tema])
            location = random.choice(locations)
            price = random.randint(price_range[0], price_range[1])
            
            ceremony = {
                'id': self.generate_uuid(),
                'category_id': self.CATEGORY_IDS['Traditional Ceremony'],
                'name': f'{pkg} {location} #{i+1}',
                'description': f'{desc}, melayani area {location}, include sesepuh dan perlengkapan lengkap',
                'vendor': f'{location} Traditional Ceremony',
                'price': price,
                'status': 'ongoing'
            }
            ceremonies.append(ceremony)
        
        return ceremonies
    
    # ==================== MAIN GENERATOR ====================
    def generate_all_data(self, tema: str, items_per_category: dict = None) -> list:
        """
        Generate semua data untuk tema tertentu
        
        Args:
            tema: 'sunda', 'jawa', atau 'betawi'
            items_per_category: Dict untuk override jumlah item per kategori
        
        Returns:
            List semua items
        """
        if items_per_category is None:
            items_per_category = {
                'Venue': 15,
                'Catering': 15,
                'Decoration': 15,
                'MUA': 15,
                'Docummentation': 15,
                'Attire': 10,
                'Entertainment': 10,
                'Master of Ceremony': 10,
                'Traditional Ceremony': 10
            }
        
        all_items = []
        
        print(f"\n🎨 Generating data untuk tema: {tema.upper()}")
        print("="*80)
        
        # Generate Venue
        print(f"📍 Generating {items_per_category['Venue']} Venue...")
        all_items.extend(self.generate_venues(tema, items_per_category['Venue']))
        
        # Generate Catering
        print(f"🍽️  Generating {items_per_category['Catering']} Catering...")
        all_items.extend(self.generate_catering(tema, items_per_category['Catering']))
        
        # Generate Decoration
        print(f"🎨 Generating {items_per_category['Decoration']} Decoration...")
        all_items.extend(self.generate_decoration(tema, items_per_category['Decoration']))
        
        # Generate MUA
        print(f"💄 Generating {items_per_category['MUA']} MUA...")
        all_items.extend(self.generate_mua(tema, items_per_category['MUA']))
        
        # Generate Docummentation
        print(f"📸 Generating {items_per_category['Docummentation']} Docummentation...")
        all_items.extend(self.generate_documentation(tema, items_per_category['Docummentation']))
        
        # Generate Attire
        print(f"👗 Generating {items_per_category['Attire']} Attire...")
        all_items.extend(self.generate_attire(tema, items_per_category['Attire']))
        
        # Generate Entertainment
        print(f"🎵 Generating {items_per_category['Entertainment']} Entertainment...")
        all_items.extend(self.generate_entertainment(tema, items_per_category['Entertainment']))
        
        # Generate MC
        print(f"🎤 Generating {items_per_category['Master of Ceremony']} Master of Ceremony...")
        all_items.extend(self.generate_mc(tema, items_per_category['Master of Ceremony']))
        
        # Generate Traditional Ceremony
        print(f"🕌 Generating {items_per_category['Traditional Ceremony']} Traditional Ceremony...")
        all_items.extend(self.generate_traditional_ceremony(tema, items_per_category['Traditional Ceremony']))
        
        print(f"\n✅ Total {len(all_items)} items generated untuk tema {tema}")
        
        return all_items
    
    def save_to_sql(self, items: list, filename: str):
        """
        Save items ke SQL file
        
        Args:
            items: List of items
            filename: Nama file output
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("-- Auto-generated Wedding Items Data\n")
            f.write(f"-- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"-- Total items: {len(items)}\n\n")
            
            for item in items:
                sql = f"""INSERT INTO items (id, category_id, name, description, vendor, price, status) VALUES (
    '{item['id']}',
    '{item['category_id']}',
    '{item['name']}',
    '{item['description']}',
    '{item['vendor']}',
    {item['price']},
    '{item['status']}'
);\n\n"""
                f.write(sql)
        
        print(f"\n💾 Data saved to: {filename}")
    
    def save_to_json(self, items: list, filename: str):
        """
        Save items ke JSON file
        
        Args:
            items: List of items
            filename: Nama file output
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Data saved to: {filename}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎊 WEDDING DATABASE EXPANDER 🎊")
    print("="*80)
    
    generator = WeddingDataGenerator()
    
    # Generate data untuk 3 tema
    all_data = []
    
    for tema in ['sunda', 'jawa', 'betawi']:
        tema_data = generator.generate_all_data(tema)
        all_data.extend(tema_data)
        
        # Save per tema
        generator.save_to_sql(tema_data, f'generated_data_{tema}.sql')
        generator.save_to_json(tema_data, f'generated_data_{tema}.json')
    
    # Save all combined
    generator.save_to_sql(all_data, 'generated_data_all.sql')
    generator.save_to_json(all_data, 'generated_data_all.json')
    
    print("\n" + "="*80)
    print(f"✅ SELESAI! Total {len(all_data)} items generated")
    print("="*80)
    print("\nFile yang di-generate:")
    print("   • generated_data_sunda.sql / .json")
    print("   • generated_data_jawa.sql / .json")
    print("   • generated_data_betawi.sql / .json")
    print("   • generated_data_all.sql / .json")
    print("\nCara import ke database:")
    print("   python import_data.py generated_data_all.sql")
    print("="*80 + "\n")
