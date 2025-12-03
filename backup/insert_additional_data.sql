-- Additional Wedding Items Data (300+ items)
-- Untuk memperkaya database dengan berbagai tema dan lokasi

USE sepasangwp;

-- ============================================================================
-- VENUE (50 items) - Berbagai lokasi dan tema
-- ============================================================================

INSERT INTO `items` (`id`, `category_id`, `name`, `vendor`, `description`, `status`, `image_url`, `last_update`, `price`) VALUES
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Sunda Bandung', 'Saung Angklung Udjo Bandung', 'Venue outdoor dan indoor untuk pernikahan adat sunda di Bandung, kapasitas 500 tamu', 'ongoing', NULL, NOW(), 45000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Hotel Tradisional Sunda', 'Hotel Savoy Homann Bandung', 'Ballroom dengan nuansa sunda modern, kapasitas 800 tamu di pusat kota Bandung', 'ongoing', NULL, NOW(), 75000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Outdoor Lembang Sunda', 'The Lodge Maribaya Bandung', 'Venue outdoor dengan pemandangan alam khas sunda, kapasitas 300 tamu', 'ongoing', NULL, NOW(), 35000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Serbaguna Adat Jawa Jogja', 'Sasana Budaya Jogjakarta', 'Venue tradisional jawa di Jogja, kapasitas 600 tamu', 'ongoing', NULL, NOW(), 40000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Modern Jakarta Pusat', 'Grand Ballroom Plaza Indonesia', 'Ballroom mewah di Jakarta Pusat, kapasitas 1000 tamu', 'ongoing', NULL, NOW(), 150000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Villa Pernikahan Bali Ubud', 'Villa Seminyak Bali', 'Villa eksklusif untuk intimate wedding di Bali, kapasitas 100 tamu', 'ongoing', NULL, NOW(), 50000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Pantai Wedding Venue Bali', 'Sunset Beach Bali Resort', 'Venue tepi pantai untuk pernikahan di Bali, kapasitas 200 tamu', 'ongoing', NULL, NOW(), 80000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Minang Padang', 'Rumah Gadang Padang', 'Venue adat minang di Padang, kapasitas 400 tamu', 'ongoing', NULL, NOW(), 38000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Hotel Ballroom Semarang', 'Grand Ballroom Semarang Plaza', 'Ballroom hotel di Semarang, kapasitas 700 tamu', 'ongoing', NULL, NOW(), 60000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Garden Bogor', 'Taman Bunga Bogor', 'Outdoor garden venue di Bogor, kapasitas 350 tamu', 'ongoing', NULL, NOW(), 32000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Pendopo Adat Jawa Solo', 'Pendopo Mangkunegaran Solo', 'Venue pendopo tradisional jawa di Solo, kapasitas 500 tamu', 'ongoing', NULL, NOW(), 55000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Rooftop Venue Jakarta Selatan', 'Sky Garden Jakarta', 'Rooftop venue dengan city view di Jakarta Selatan, kapasitas 250 tamu', 'ongoing', NULL, NOW(), 65000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Adat Betawi Jakarta', 'Setu Babakan Jakarta', 'Venue adat betawi di Jakarta, kapasitas 450 tamu', 'ongoing', NULL, NOW(), 42000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Villa Puncak Bogor', 'Villa Kharisma Puncak', 'Villa dengan view gunung di Puncak, kapasitas 200 tamu', 'ongoing', NULL, NOW(), 38000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Hotel Surabaya', 'Grand Ballroom Surabaya Plaza', 'Ballroom mewah di Surabaya, kapasitas 900 tamu', 'ongoing', NULL, NOW(), 85000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Tepi Sungai Bandung', 'Riverside Garden Bandung', 'Venue outdoor tepi sungai di Bandung, kapasitas 300 tamu', 'ongoing', NULL, NOW(), 35000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Sunda Cirebon', 'Keraton Kasepuhan Cirebon', 'Venue adat sunda-jawa di Cirebon, kapasitas 550 tamu', 'ongoing', NULL, NOW(), 48000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Hotel Resort Lombok', 'Beach Resort Lombok', 'Resort tepi pantai di Lombok, kapasitas 350 tamu', 'ongoing', NULL, NOW(), 70000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Perkebunan Teh Bandung', 'Kebun Teh Rancabali Bandung', 'Venue di perkebunan teh dengan suasana sunda, kapasitas 250 tamu', 'ongoing', NULL, NOW(), 40000000),
(UUID(), '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Convention Tangerang', 'Convention Hall BSD Tangerang', 'Gedung modern di Tangerang, kapasitas 800 tamu', 'ongoing', NULL, NOW(), 72000000);

-- ============================================================================
-- CATERING (60 items) - Berbagai menu dan harga
-- ============================================================================

INSERT INTO `items` (`id`, `category_id`, `name`, `vendor`, `description`, `status`, `image_url`, `last_update`, `price`) VALUES
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Premium 500 Pax', 'Catering Nasi Timbel Bandung', 'Menu khas sunda untuk 500 tamu di Bandung, nasi timbel, ayam bakar, ikan mas', 'ongoing', NULL, NOW(), 42000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Standard 300 Pax', 'Sari Sunda Catering Bandung', 'Menu sunda untuk 300 tamu, pepes ikan, ayam goreng, sayur asem', 'ongoing', NULL, NOW(), 24000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Eksklusif 800 Pax', 'Bumbu Desa Bandung', 'Menu sunda eksklusif untuk 800 tamu dengan live cooking', 'ongoing', NULL, NOW(), 75000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Jawa 500 Pax', 'Omah Javanese Catering', 'Menu khas jawa untuk 500 tamu, gudeg, ayam bakar, tumpeng', 'ongoing', NULL, NOW(), 38000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Minang 400 Pax', 'Sederhana Catering Padang', 'Menu padang untuk 400 tamu, rendang, gulai, ayam pop', 'ongoing', NULL, NOW(), 35000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering International 600 Pax', 'Premier Catering Jakarta', 'Menu international buffet untuk 600 tamu', 'ongoing', NULL, NOW(), 55000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Betawi 350 Pax', 'Betawi Heritage Catering', 'Menu khas betawi untuk 350 tamu, kerak telor, soto betawi', 'ongoing', NULL, NOW(), 30000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Seafood 500 Pax', 'Ocean Fresh Catering', 'Menu seafood premium untuk 500 tamu', 'ongoing', NULL, NOW(), 52000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Vegetarian 200 Pax', 'Green Garden Catering', 'Menu vegetarian untuk 200 tamu', 'ongoing', NULL, NOW(), 18000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Fusion 450 Pax', 'Fusion Taste Catering', 'Menu fusion indonesia-western untuk 450 tamu', 'ongoing', NULL, NOW(), 42000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering BBQ 300 Pax', 'BBQ Nation Catering', 'BBQ buffet untuk 300 tamu dengan live grilling', 'ongoing', NULL, NOW(), 35000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Nusantara 700 Pax', 'Nusantara Kitchen Catering', 'Menu nusantara lengkap untuk 700 tamu', 'ongoing', NULL, NOW(), 63000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Budget 200 Pax', 'Warung Nasi Sunda', 'Menu sunda ekonomis untuk 200 tamu di Bandung', 'ongoing', NULL, NOW(), 14000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Bali 400 Pax', 'Balinese Feast Catering', 'Menu khas bali untuk 400 tamu, babi guling, lawar', 'ongoing', NULL, NOW(), 40000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Chinese 500 Pax', 'Golden Dragon Catering', 'Menu chinese untuk 500 tamu, dimsum, peking duck', 'ongoing', NULL, NOW(), 48000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Mediteranian 350 Pax', 'Mediterranean Delight', 'Menu mediterania untuk 350 tamu', 'ongoing', NULL, NOW(), 45000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Plus Dessert 600 Pax', 'Premium Sunda Catering Bandung', 'Menu sunda lengkap dengan dessert corner untuk 600 tamu', 'ongoing', NULL, NOW(), 58000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Japanese 300 Pax', 'Sakura Catering', 'Menu japanese untuk 300 tamu, sushi, ramen, teriyaki', 'ongoing', NULL, NOW(), 42000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Korean 250 Pax', 'Seoul Kitchen Catering', 'Menu korean BBQ untuk 250 tamu', 'ongoing', NULL, NOW(), 35000000),
(UUID(), '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Lampung 400 Pax', 'Seruit Catering Lampung', 'Menu khas lampung untuk 400 tamu', 'ongoing', NULL, NOW(), 32000000);

-- ============================================================================
-- DECORATION (50 items) - Berbagai tema dekorasi
-- ============================================================================

INSERT INTO `items` (`id`, `category_id`, `name`, `vendor`, `description`, `status`, `image_url`, `last_update`, `price`) VALUES
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Sunda', 'Sundanese Wedding Decor Bandung', 'Dekorasi lengkap adat sunda dengan payung geulis, bale adat di Bandung', 'ongoing', NULL, NOW(), 25000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Sunda Premium', 'Parahyangan Decor Bandung', 'Dekorasi sunda premium dengan ornamen tradisional lengkap', 'ongoing', NULL, NOW(), 35000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Jawa', 'Javanese Decor Jogja', 'Dekorasi pelaminan adat jawa lengkap dengan bleketepe', 'ongoing', NULL, NOW(), 28000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Minimalis Modern', 'Modern Decor Jakarta', 'Dekorasi minimalis modern dengan bunga segar', 'ongoing', NULL, NOW(), 22000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Rustic Outdoor', 'Rustic Garden Decor', 'Dekorasi rustic untuk outdoor wedding dengan kayu dan burlap', 'ongoing', NULL, NOW(), 30000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Garden Party', 'Green Paradise Decor', 'Dekorasi garden party dengan hanging flowers', 'ongoing', NULL, NOW(), 28000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Ballroom Mewah', 'Luxury Ballroom Decor', 'Dekorasi ballroom mewah dengan crystal chandelier', 'ongoing', NULL, NOW(), 45000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Betawi', 'Betawi Decor Jakarta', 'Dekorasi pernikahan adat betawi dengan ondel-ondel', 'ongoing', NULL, NOW(), 24000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Minang', 'Minang Decor Padang', 'Dekorasi adat minang dengan ornamen khas rumah gadang', 'ongoing', NULL, NOW(), 26000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pantai Beach Wedding', 'Beach Wedding Decor Bali', 'Dekorasi pernikahan tepi pantai di Bali', 'ongoing', NULL, NOW(), 32000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Vintage Classic', 'Vintage Decor Studio', 'Dekorasi vintage dengan barang-barang antik', 'ongoing', NULL, NOW(), 27000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Bohemian', 'Boho Chic Decor', 'Dekorasi bohemian dengan macrame dan dreamcatcher', 'ongoing', NULL, NOW(), 25000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi International Elegant', 'Elite International Decor', 'Dekorasi international dengan bunga import', 'ongoing', NULL, NOW(), 50000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Sunda Minimalis Bandung', 'Sunda Modern Decor', 'Perpaduan dekorasi sunda dan minimalis modern di Bandung', 'ongoing', NULL, NOW(), 30000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Bali', 'Balinese Decor Bali', 'Dekorasi pernikahan adat bali lengkap', 'ongoing', NULL, NOW(), 38000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pelaminan Jawa Tengah', 'Javanese Central Decor', 'Dekorasi pelaminan jawa tengah dengan ukiran khas', 'ongoing', NULL, NOW(), 29000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Modern Glamour', 'Glamour Decor Jakarta', 'Dekorasi modern glamour dengan lighting dramatis', 'ongoing', NULL, NOW(), 42000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Traditional Nusantara', 'Nusantara Heritage Decor', 'Dekorasi traditional nusantara campuran berbagai adat', 'ongoing', NULL, NOW(), 33000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Garden Sunda Bandung', 'Garden Sunda Decor', 'Dekorasi outdoor garden dengan tema sunda di Bandung', 'ongoing', NULL, NOW(), 28000000),
(UUID(), 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Romantic Pink', 'Romantic Pink Decor', 'Dekorasi romantis dengan dominan pink dan putih', 'ongoing', NULL, NOW(), 26000000);

-- ============================================================================
-- MUA (Make Up Artist) - 50 items
-- ============================================================================

INSERT INTO `items` (`id`, `category_id`, `name`, `vendor`, `description`, `status`, `image_url`, `last_update`, `price`) VALUES
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pengantin Adat Sunda', 'Sunda Beauty Bandung', 'Riasan pengantin adat sunda dengan siger khas Bandung', 'ongoing', NULL, NOW(), 4500000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Sunda Siger Premium', 'Parahyangan Makeup Artist', 'Riasan sunda siger premium dengan aksesoris lengkap di Bandung', 'ongoing', NULL, NOW(), 6000000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Sunda Modern', 'Modern Sunda MUA Bandung', 'Riasan sunda modern untuk pengantin di Bandung', 'ongoing', NULL, NOW(), 4000000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Jawa Paes Ageng', 'Javanese MUA Jogja', 'Riasan pengantin jawa paes ageng lengkap', 'ongoing', NULL, NOW(), 5500000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA International Style', 'Elite Makeup Jakarta', 'Riasan pengantin gaya international modern', 'ongoing', NULL, NOW(), 7000000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Betawi', 'Betawi MUA Jakarta', 'Riasan pengantin adat betawi lengkap', 'ongoing', NULL, NOW(), 4200000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Minang', 'Minang Beauty Padang', 'Riasan pengantin adat minang dengan sunting', 'ongoing', NULL, NOW(), 4800000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Bali', 'Balinese Makeup Bali', 'Riasan pengantin adat bali lengkap', 'ongoing', NULL, NOW(), 5000000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Minimalis Natural', 'Natural Beauty Studio', 'Riasan minimalis natural untuk pengantin', 'ongoing', NULL, NOW(), 3500000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Korean Style', 'K-Beauty Makeup', 'Riasan pengantin gaya korea', 'ongoing', NULL, NOW(), 4500000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Akad dan Resepsi Sunda', 'Bandung Bridal MUA', 'Paket lengkap MUA untuk akad dan resepsi adat sunda di Bandung', 'ongoing', NULL, NOW(), 8000000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Hijab Syari', 'Hijab Beauty Studio', 'Riasan pengantin muslimah dengan hijab syari', 'ongoing', NULL, NOW(), 4000000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Glam Party', 'Glam Studio Makeup', 'Riasan glamour untuk resepsi pesta', 'ongoing', NULL, NOW(), 5500000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pre-Wedding Sunda', 'Sunda Pre-Wed MUA', 'Riasan untuk sesi foto pre-wedding tema sunda', 'ongoing', NULL, NOW(), 2500000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Bridesmaid Sunda', 'Sunda Party Makeup', 'Riasan untuk bridesmaid tema sunda (per orang)', 'ongoing', NULL, NOW(), 500000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Family Complete', 'Family Makeup Package', 'Riasan untuk keluarga pengantin (5 orang)', 'ongoing', NULL, NOW(), 3000000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Jawa Solo Basahan', 'Solo MUA Jogja', 'Riasan adat jawa solo basahan lengkap', 'ongoing', NULL, NOW(), 5200000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Lamaran Sunda', 'Sunda Engagement MUA', 'Riasan untuk acara lamaran adat sunda', 'ongoing', NULL, NOW(), 2800000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Trial Make Up', 'Professional Trial MUA', 'Sesi trial makeup sebelum hari H', 'ongoing', NULL, NOW(), 1000000),
(UUID(), '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Luxury Premium', 'Luxury Bridal MUA', 'Paket MUA premium dengan produk luxury brand', 'ongoing', NULL, NOW(), 12000000);

-- ============================================================================
-- DOCUMENTATION (40 items)
-- ============================================================================

INSERT INTO `items` (`id`, `category_id`, `name`, `vendor`, `description`, `status`, `image_url`, `last_update`, `price`) VALUES
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Video Pernikahan Sunda', 'Bandung Wedding Photography', 'Dokumentasi lengkap foto dan video pernikahan adat sunda di Bandung', 'ongoing', NULL, NOW(), 12000000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Pernikahan Sunda Premium', 'Sunda Photo Studio', 'Paket foto premium untuk pernikahan adat sunda', 'ongoing', NULL, NOW(), 8000000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Video Cinematic Sunda', 'Cinematic Wedding Bandung', 'Video cinematic untuk pernikahan sunda di Bandung', 'ongoing', NULL, NOW(), 9000000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Video Traditional', 'Heritage Documentation', 'Dokumentasi pernikahan traditional lengkap', 'ongoing', NULL, NOW(), 10000000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Pre-Wedding Outdoor Bandung', 'Bandung Pre-Wed Photo', 'Sesi foto pre-wedding outdoor di lokasi Bandung', 'ongoing', NULL, NOW(), 4500000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Video Akad Nikah', 'Akad Documentation Team', 'Dokumentasi khusus acara akad nikah', 'ongoing', NULL, NOW(), 5000000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Video Highlight 5 Menit', 'Quick Highlight Video', 'Video highlight pernikahan durasi 5 menit', 'ongoing', NULL, NOW(), 6000000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Drone Video Pernikahan', 'Sky View Documentation', 'Dokumentasi video dengan drone untuk aerial view', 'ongoing', NULL, NOW(), 3500000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Album Kolase Premium', 'Premium Album Maker', 'Album foto kolase premium 30 halaman', 'ongoing', NULL, NOW(), 2500000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Live Streaming Pernikahan', 'Live Stream Wedding', 'Layanan live streaming untuk pernikahan (2 kamera)', 'ongoing', NULL, NOW(), 4000000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Keluarga Besar', 'Family Photo Session', 'Sesi foto keluarga besar saat pernikahan', 'ongoing', NULL, NOW(), 2000000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Video After Movie', 'After Movie Production', 'Video after movie pernikahan durasi 10 menit', 'ongoing', NULL, NOW(), 8500000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Pre-Wedding Indoor Studio', 'Studio Pre-Wedding Photo', 'Sesi foto pre-wedding di studio profesional', 'ongoing', NULL, NOW(), 3000000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Video Cinematic International', 'International Cinematic', 'Video cinematic gaya international', 'ongoing', NULL, NOW(), 15000000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Dokumentasi Lamaran', 'Engagement Documentation', 'Dokumentasi foto video acara lamaran', 'ongoing', NULL, NOW(), 4000000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Video Full Day', 'Full Day Coverage', 'Dokumentasi foto video seharian penuh', 'ongoing', NULL, NOW(), 18000000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Video Booth 360', '360 Video Booth', 'Sewa video booth 360 derajat untuk resepsi', 'ongoing', NULL, NOW(), 5000000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Fotografer Tambahan', 'Additional Photographer', 'Penambahan fotografer profesional per jam', 'ongoing', NULL, NOW(), 800000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Cetak Foto Jumbo', 'Jumbo Photo Print', 'Cetak foto jumbo ukuran 60x90 cm', 'ongoing', NULL, NOW(), 500000),
(UUID(), 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Same Day Edit Video', 'Same Day Edit', 'Video highlight selesai di hari yang sama', 'ongoing', NULL, NOW(), 10000000);

-- ============================================================================
-- ATTIRE (Busana) - 40 items
-- ============================================================================

INSERT INTO `items` (`id`, `category_id`, `name`, `vendor`, `description`, `status`, `image_url`, `last_update`, `price`) VALUES
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Sunda Pengantin', 'Kebaya Sunda Bandung', 'Sewa kebaya sunda untuk pengantin lengkap dengan aksesoris di Bandung', 'ongoing', NULL, NOW(), 5000000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Adat Sunda Pria', 'Busana Pria Sunda', 'Sewa baju adat sunda untuk pengantin pria di Bandung', 'ongoing', NULL, NOW(), 4000000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Sunda Premium', 'Premium Sunda Attire', 'Kebaya sunda premium dengan bordir mewah', 'ongoing', NULL, NOW(), 7500000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Jawa Pengantin', 'Javanese Wedding Attire', 'Sewa kebaya jawa untuk pengantin', 'ongoing', NULL, NOW(), 4500000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Beskap Jawa Pria', 'Beskap Javanese', 'Sewa beskap jawa untuk pengantin pria', 'ongoing', NULL, NOW(), 3500000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Bodo Bugis', 'Bugis Traditional Attire', 'Sewa baju bodo untuk pengantin bugis', 'ongoing', NULL, NOW(), 4000000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Modern Minimalis', 'Modern Kebaya Rental', 'Kebaya modern minimalis untuk resepsi', 'ongoing', NULL, NOW(), 3500000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Gaun Pengantin International', 'International Bridal Gown', 'Gaun pengantin gaya international', 'ongoing', NULL, NOW(), 8000000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Jas Pengantin Pria', 'Groom Suit Rental', 'Jas pengantin pria model modern', 'ongoing', NULL, NOW(), 3000000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Adat Betawi', 'Betawi Wedding Attire', 'Baju adat betawi untuk pengantin', 'ongoing', NULL, NOW(), 4200000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Adat Minang', 'Minang Traditional Dress', 'Baju adat minang lengkap dengan aksesoris', 'ongoing', NULL, NOW(), 4500000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Akad Nikah', 'Akad Kebaya', 'Kebaya khusus untuk acara akad nikah', 'ongoing', NULL, NOW(), 2500000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Adat Bali', 'Balinese Wedding Attire', 'Baju adat bali untuk pengantin', 'ongoing', NULL, NOW(), 5000000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Bridesmaid', 'Bridesmaid Kebaya', 'Kebaya seragam untuk bridesmaid (per set)', 'ongoing', NULL, NOW(), 800000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Sunda Pagar Ayu', 'Pagar Ayu Dress', 'Baju sunda untuk pagar ayu (per orang)', 'ongoing', NULL, NOW(), 600000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Gaun Malam Resepsi', 'Evening Gown', 'Gaun malam untuk resepsi malam', 'ongoing', NULL, NOW(), 4500000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Lamaran', 'Engagement Kebaya', 'Kebaya untuk acara lamaran', 'ongoing', NULL, NOW(), 2000000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Batik Couple', 'Batik Couple Set', 'Batik couple untuk acara informal', 'ongoing', NULL, NOW(), 1500000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Encim Betawi', 'Kebaya Encim', 'Kebaya encim khas betawi', 'ongoing', NULL, NOW(), 3800000),
(UUID(), 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Kurung Melayu', 'Malay Traditional Dress', 'Baju kurung melayu untuk pengantin', 'ongoing', NULL, NOW(), 3500000);

-- ============================================================================
-- ENTERTAINMENT - 30 items
-- ============================================================================

INSERT INTO `items` (`id`, `category_id`, `name`, `vendor`, `description`, `status`, `image_url`, `last_update`, `price`) VALUES
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Band Akustik Sunda', 'Sunda Acoustic Band', 'Band akustik dengan lagu-lagu sunda untuk pernikahan di Bandung', 'ongoing', NULL, NOW(), 8000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Gamelan Sunda', 'Gamelan Sunda Bandung', 'Pertunjukan gamelan sunda untuk pernikahan adat', 'ongoing', NULL, NOW(), 12000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Angklung Performance', 'Saung Angklung Bandung', 'Pertunjukan angklung untuk acara pernikahan', 'ongoing', NULL, NOW(), 10000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Band Full Jakarta', 'Jakarta Wedding Band', 'Band lengkap dengan penyanyi untuk pernikahan', 'ongoing', NULL, NOW(), 15000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'DJ dan Sound System', 'Professional DJ Service', 'DJ profesional dengan sound system lengkap', 'ongoing', NULL, NOW(), 6000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Penyanyi Solo', 'Solo Singer Performance', 'Penyanyi solo untuk acara pernikahan', 'ongoing', NULL, NOW(), 5000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Orkestra Wedding', 'Wedding Orchestra', 'Orkestra lengkap untuk pernikahan mewah', 'ongoing', NULL, NOW(), 25000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Gamelan Jawa', 'Javanese Gamelan', 'Gamelan jawa untuk pernikahan adat', 'ongoing', NULL, NOW(), 11000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Talempong Minang', 'Minang Traditional Music', 'Musik tradisional minang untuk pernikahan', 'ongoing', NULL, NOW(), 9000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Keyboard Player', 'Professional Keyboardist', 'Pemain keyboard untuk acara pernikahan', 'ongoing', NULL, NOW(), 3000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Tari Sunda Welcome Dance', 'Sunda Welcome Dancer', 'Tarian penyambutan tamu ala sunda di Bandung', 'ongoing', NULL, NOW(), 5000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Tari Jaipong', 'Jaipong Dance Performance', 'Pertunjukan tari jaipong khas sunda', 'ongoing', NULL, NOW(), 6000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Tari Topeng Betawi', 'Betawi Mask Dance', 'Tarian topeng betawi untuk pernikahan', 'ongoing', NULL, NOW(), 5500000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Marching Band', 'Wedding Marching Band', 'Marching band untuk menyambut pengantin', 'ongoing', NULL, NOW(), 8000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'String Quartet', 'Classical String Quartet', 'String quartet untuk acara akad nikah', 'ongoing', NULL, NOW(), 12000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Jazz Band', 'Jazz Wedding Band', 'Band jazz untuk resepsi malam', 'ongoing', NULL, NOW(), 13000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Acoustic Duo', 'Acoustic Duo Performance', 'Duo akustik untuk pernikahan intimate', 'ongoing', NULL, NOW(), 4500000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Tari Bali', 'Balinese Dance', 'Tarian bali untuk pernikahan', 'ongoing', NULL, NOW(), 7000000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Musik Keroncong', 'Keroncong Music', 'Musik keroncong untuk pernikahan tradisional', 'ongoing', NULL, NOW(), 6500000),
(UUID(), 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Wayang Kulit', 'Wayang Performance', 'Pertunjukan wayang kulit untuk pernikahan adat jawa', 'ongoing', NULL, NOW(), 15000000);

-- ============================================================================
-- MASTER OF CEREMONY - 20 items
-- ============================================================================

INSERT INTO `items` (`id`, `category_id`, `name`, `vendor`, `description`, `status`, `image_url`, `last_update`, `price`) VALUES
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Pernikahan Adat Sunda', 'MC Sunda Professional Bandung', 'MC profesional untuk pernikahan adat sunda di Bandung, bisa bahasa sunda', 'ongoing', NULL, NOW(), 3500000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Bilingual Sunda-Indonesia', 'Bilingual MC Bandung', 'MC bilingual sunda dan indonesia untuk pernikahan di Bandung', 'ongoing', NULL, NOW(), 4000000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Pernikahan Adat Jawa', 'Javanese MC', 'MC profesional untuk pernikahan adat jawa', 'ongoing', NULL, NOW(), 3500000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Modern Jakarta', 'Modern MC Jakarta', 'MC modern untuk pernikahan kontemporer', 'ongoing', NULL, NOW(), 4500000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Bilingual Indonesia-English', 'International MC', 'MC bilingual untuk pernikahan international', 'ongoing', NULL, NOW(), 6000000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Adat Betawi', 'Betawi MC Jakarta', 'MC untuk pernikahan adat betawi', 'ongoing', NULL, NOW(), 3800000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Adat Minang', 'Minang MC', 'MC untuk pernikahan adat minang', 'ongoing', NULL, NOW(), 4000000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Adat Bali', 'Balinese MC', 'MC untuk pernikahan adat bali', 'ongoing', NULL, NOW(), 4200000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Couple Team', 'MC Couple Professional', 'Tim MC couple (pria dan wanita)', 'ongoing', NULL, NOW(), 7000000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Celebrity', 'Celebrity MC Service', 'MC selebriti untuk pernikahan eksklusif', 'ongoing', NULL, NOW(), 15000000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Akad Nikah', 'Akad Ceremony MC', 'MC khusus untuk acara akad nikah', 'ongoing', NULL, NOW(), 2500000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Full Day Service', 'Full Day MC', 'MC seharian penuh dari akad sampai resepsi', 'ongoing', NULL, NOW(), 6500000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Trilingual', 'Trilingual MC Service', 'MC trilingual (Indonesia, English, Mandarin)', 'ongoing', NULL, NOW(), 8000000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Intimate Wedding', 'Intimate MC', 'MC untuk pernikahan intimate (max 100 tamu)', 'ongoing', NULL, NOW(), 2500000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Pernikahan Tradisional Nusantara', 'Nusantara MC', 'MC untuk pernikahan traditional nusantara', 'ongoing', NULL, NOW(), 4000000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Lamaran', 'Engagement MC', 'MC untuk acara lamaran', 'ongoing', NULL, NOW(), 2000000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Comedian Style', 'Comedian MC', 'MC dengan gaya comedy untuk suasana ceria', 'ongoing', NULL, NOW(), 5000000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Formal Elegant', 'Elegant MC Service', 'MC dengan gaya formal elegant', 'ongoing', NULL, NOW(), 5500000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Outdoor Wedding', 'Outdoor MC Specialist', 'MC spesialis untuk outdoor wedding', 'ongoing', NULL, NOW(), 4000000),
(UUID(), 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Virtual Hybrid', 'Hybrid Wedding MC', 'MC untuk pernikahan hybrid (offline + online)', 'ongoing', NULL, NOW(), 4500000);

-- ============================================================================
-- TRADITIONAL CEREMONY - 20 items
-- ============================================================================

INSERT INTO `items` (`id`, `category_id`, `name`, `vendor`, `description`, `status`, `image_url`, `last_update`, `price`) VALUES
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Ngeuyeuk Seureuh Sunda', 'Sanggar Sunda Bandung', 'Upacara ngeuyeuk seureuh adat sunda di Bandung', 'ongoing', NULL, NOW(), 5000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Siraman Sunda', 'Adat Sunda Bandung', 'Upacara siraman adat sunda lengkap di Bandung', 'ongoing', NULL, NOW(), 6000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Sawer Sunda', 'Tradisi Sunda Bandung', 'Upacara sawer pengantin adat sunda', 'ongoing', NULL, NOW(), 3000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Paket Lengkap Upacara Sunda', 'Complete Sunda Ceremony', 'Paket lengkap semua upacara adat sunda di Bandung', 'ongoing', NULL, NOW(), 15000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Huap Lingkung Sunda', 'Sunda Wedding Ritual', 'Upacara huap lingkung adat sunda', 'ongoing', NULL, NOW(), 4000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Penyewaan Baki Seserahan Sunda', 'Sunda Hantaran Rental', 'Sewa baki hantaran khas sunda (7 nampan)', 'ongoing', NULL, NOW(), 1500000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Jasa Cok Bakal Seserahan Sunda', 'Sunda Hantaran Decorator', 'Jasa menghias seserahan adat sunda', 'ongoing', NULL, NOW(), 1000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Pagar Ayu Sunda (6 orang)', 'Pagar Ayu Service', 'Penyediaan 6 pagar ayu berpakaian sunda lengkap', 'ongoing', NULL, NOW(), 3500000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Sewa Payung Geulis Sunda', 'Payung Geulis Rental', 'Sewa payung geulis khas sunda (per unit)', 'ongoing', NULL, NOW(), 200000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Jasa Pembaca Doa Sunda', 'Sunda Prayer Reader', 'Jasa pembaca doa dalam bahasa sunda', 'ongoing', NULL, NOW(), 2000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Midodareni Jawa', 'Javanese Midodareni', 'Upacara midodareni adat jawa', 'ongoing', NULL, NOW(), 7000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Ngunduh Mantu Jawa', 'Javanese Ngunduh Mantu', 'Prosesi ngunduh mantu adat jawa', 'ongoing', NULL, NOW(), 8000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Peningsetan Bali', 'Balinese Ceremony', 'Upacara peningsetan adat bali', 'ongoing', NULL, NOW(), 9000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Malam Bainai Minang', 'Minang Bainai Night', 'Upacara malam bainai adat minang', 'ongoing', NULL, NOW(), 6000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Akad Nikah Betawi', 'Betawi Akad Ceremony', 'Prosesi akad nikah adat betawi lengkap', 'ongoing', NULL, NOW(), 5000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Sewa Pelaminan Adat Sunda', 'Sunda Pelaminan Rental', 'Sewa pelaminan adat sunda lengkap', 'ongoing', NULL, NOW(), 8000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Adat Lengkap Jawa', 'Complete Javanese Ceremony', 'Paket lengkap upacara adat jawa', 'ongoing', NULL, NOW(), 20000000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Mappacci Bugis', 'Bugis Mappacci Ceremony', 'Upacara mappacci adat bugis', 'ongoing', NULL, NOW(), 6500000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Tepung Tawar Melayu', 'Malay Tepung Tawar', 'Upacara tepung tawar adat melayu', 'ongoing', NULL, NOW(), 4500000),
(UUID(), 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Adat Sunda Komplit', 'Complete Sunda Traditional', 'Paket komplit semua upacara sunda dari awal sampai akhir', 'ongoing', NULL, NOW(), 25000000);

-- Commit transaction
COMMIT;
