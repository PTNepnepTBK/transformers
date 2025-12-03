-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Dec 03, 2025 at 05:52 AM
-- Server version: 10.4.28-MariaDB
-- PHP Version: 8.2.4

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `sepasangwp`
--

-- --------------------------------------------------------

--
-- Table structure for table `categories`
--

CREATE TABLE `categories` (
  `id` char(36) NOT NULL DEFAULT uuid(),
  `name` varchar(255) NOT NULL,
  `description` text DEFAULT NULL,
  `created_at` datetime DEFAULT current_timestamp(),
  `updated_at` datetime DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `image_url` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `categories`
--

INSERT INTO `categories` (`id`, `name`, `description`, `created_at`, `updated_at`, `image_url`) VALUES
('1b2f2a98-7c81-47fb-b336-c939733ada78', 'MUA', 'Make Up Artist dan hair do', '2025-11-10 10:56:25', '2025-11-10 10:56:25', NULL),
('33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Catering', 'Solusi untuk makanan yang akan di pilih', '2025-11-15 17:39:11', '2025-11-23 10:38:05', 'bahlil_a94948d9-7b02-4720-8f33-24616c757eb1.webp'),
('3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue', 'Venue yang tersedia dari berbagai daerah', '2025-11-10 11:00:24', '2025-11-23 10:36:46', NULL),
('ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Attire', 'busana khas ciri indonesia', '2025-11-23 10:31:12', '2025-11-23 10:31:12', 'Gemini_Generated_Image_kubdqnkubdqnkubd_82cbf9c5-5247-48cd-9c35-6fc8804dc0ab.webp'),
('b01d482d-7301-44d4-80c1-155f1866a604', 'Decoration', 'dekorasi dari indonesia', '2025-11-10 10:56:10', '2025-11-23 10:33:48', NULL),
('b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'Master of Ceremony', 'Pembawa Acara', '2025-11-10 10:59:40', '2025-11-23 10:43:19', NULL),
('b4c13e15-6725-49c2-946e-98f57895cad3', 'Docummentation', 'Dokumentasi dari fotografi dan videografi', '2025-11-10 10:49:13', '2025-11-23 10:35:26', NULL),
('bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Entertainment', 'Solusi hiburan seperti musik dan keyboard', '2025-11-10 11:00:13', '2025-11-23 10:39:25', NULL),
('c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Traditional Ceremony', 'Upacara tradisional', '2025-11-10 11:00:03', '2025-11-23 10:41:25', NULL),
('d1541569-2ec0-4ef7-9e17-a2cc3f964097', 'etc', 'dll', '2025-11-10 10:59:27', '2025-11-23 10:44:09', NULL);

-- --------------------------------------------------------

--
-- Table structure for table `category_logs`
--

CREATE TABLE `category_logs` (
  `id` bigint(20) NOT NULL,
  `category_id` char(36) DEFAULT NULL,
  `created_at` datetime DEFAULT current_timestamp(),
  `total_visit` smallint(6) DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `items`
--

CREATE TABLE `items` (
  `id` char(36) NOT NULL DEFAULT uuid(),
  `category_id` char(36) DEFAULT NULL,
  `name` varchar(255) NOT NULL,
  `vendor` varchar(255) DEFAULT NULL,
  `description` text DEFAULT NULL,
  `status` varchar(50) DEFAULT 'ongoing' CHECK (`status` in ('airing','ongoing','archived')),
  `image_url` varchar(255) DEFAULT NULL,
  `last_update` datetime DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `price` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `items`
--

INSERT INTO `items` (`id`, `category_id`, `name`, `vendor`, `description`, `status`, `image_url`, `last_update`, `price`) VALUES
('0cf070e8-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto & Video Pernikahan Silver (1 Day)', 'Aura Media Production', 'Liputan foto & video 1 hari (Akad/Pemberkatan & Resepsi), 1 fotografer, 1 videografer.', 'ongoing', 'docu_silver_package.jpg', '2025-11-29 17:49:32', 8000000),
('0cf07e6a-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Pernikahan Saja (Full Day)', 'Kamera Senja Studio', 'Liputan foto seharian, 2 fotografer, free cetak kanvas 40x60.', 'ongoing', NULL, '2025-11-29 17:49:32', 5500000),
('0cf07f05-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Jasa Video Cinematic (Highlight 3 Menit)', 'Vidio Kreatif', 'Video highlight sinematik berdurasi 3 menit, editing cepat (maks. 2 minggu).', 'ongoing', 'docu_video_cinematic.jpg', '2025-11-29 17:49:32', 4000000),
('0cf07f56-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Pre-Wedding Outdoor Eksklusif', 'Aura Media Production', 'Sesi foto pre-wedding outdoor 1 hari (maks. 2 lokasi), free 10 foto edit premium.', 'ongoing', 'docu_prewed_outdoor.jpg', '2025-11-29 17:49:32', 3500000),
('0cf07fab-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Sewa Drone Video & Foto (per jam)', 'Kamera Senja Studio', 'Jasa pilot drone untuk pengambilan sudut pandang dari udara, durasi minimal 2 jam.', 'ongoing', NULL, '2025-11-29 17:49:32', 1500000),
('0cf08078-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Video Dokumentasi Acara Korporat', 'Vidio Kreatif', 'Liputan video formal untuk seminar atau launching produk, durasi full event.', 'ongoing', NULL, '2025-11-29 17:49:32', 6000000),
('0cf080c2-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Live Streaming Acara (2 Kamera)', 'Aura Media Production', 'Layanan live streaming 2 kamera via YouTube/Zoom, termasuk operator dan mixing.', 'ongoing', NULL, '2025-11-29 17:49:32', 4500000),
('0cf0810a-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Cetak Album Kolase Hard Cover (20 Halaman)', 'Kamera Senja Studio', 'Jasa cetak album foto kolase ukuran 20x30 cm, bahan glossy premium.', 'ongoing', NULL, '2025-11-29 17:49:32', 1200000),
('0cf08145-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Video Dokumentasi Lamaran/Tunangan', 'Vidio Kreatif', 'Liputan video untuk acara lamaran, durasi highlight 1 menit, video full 30 menit.', 'ongoing', NULL, '2025-11-29 17:49:32', 2800000),
('0cf08184-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Keluarga (Studio Session)', 'Aura Media Production', 'Sesi foto keluarga di studio (maks. 6 orang), free 5 cetak foto besar.', 'ongoing', NULL, '2025-11-29 17:49:32', 1800000),
('0cf081bf-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Fotografer Tambahan (per jam/orang)', 'Kamera Senja Studio', 'Penambahan fotografer profesional untuk coverage yang lebih detail.', 'ongoing', NULL, '2025-11-29 17:49:32', 500000),
('0cf081fb-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Video Dokumentasi After Movie (5-7 Menit)', 'Vidio Kreatif', 'Video berdurasi 5-7 menit, editing sinematik dengan narasi atau musik berlisensi.', 'ongoing', 'docu_after_movie.jpg', '2025-11-29 17:49:32', 7500000),
('0cf08239-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Jasa Video Booth 360 Derajat', 'Aura Media Production', 'Penyewaan platform video booth 360 derajat untuk resepsi/pesta.', 'ongoing', NULL, '2025-11-29 17:49:32', 5000000),
('0cf08278-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Dokumentasi Produk (E-commerce)', 'Kamera Senja Studio', 'Jasa foto produk untuk kebutuhan e-commerce (10 produk, 5 angle per produk).', 'ongoing', NULL, '2025-11-29 17:49:32', 2500000),
('0cf082b7-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Video Sesi Wawancara (Testimonial)', 'Vidio Kreatif', 'Jasa perekaman dan editing video testimonial/wawancara profesional.', 'ongoing', NULL, '2025-11-29 17:49:32', 3000000),
('0cf082f3-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Video Klip Musik/Brand (Full Concept)', 'Aura Media Production', 'Pembuatan video klip atau brand video lengkap dengan konsep dan storyboard.', 'airing', NULL, '2025-11-29 17:49:32', 15000000),
('0cf08332-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Sewa Kamera DSLR/Mirrorless (per hari)', 'Kamera Senja Studio', 'Penyewaan unit kamera pro (tanpa operator) untuk kebutuhan pribadi/amatir.', 'ongoing', NULL, '2025-11-29 17:49:32', 400000),
('0cf083a5-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Jasa Edit Video Saja (Mentahan Disediakan)', 'Vidio Kreatif', 'Jasa editing video dari mentahan yang sudah disediakan klien, maksimal durasi 10 menit.', 'ongoing', NULL, '2025-11-29 17:49:32', 1500000),
('0cf084b1-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto & Video Platinum (2 Hari)', 'Aura Media Production', 'Liputan foto & video selama 2 hari (Pre-wed & Wedding), 3 kru, free album premium.', 'ongoing', 'docu_platinum_package.jpg', '2025-11-29 17:49:32', 18000000),
('0cf084f3-cd11-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Jasa Retouching Foto Tambahan (per foto)', 'Kamera Senja Studio', 'Layanan retouching foto ekstra (penghalusan kulit, koreksi warna kompleks).', 'ongoing', NULL, '2025-11-29 17:49:32', 50000),
('1e35c944-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Siraman & Midodareni Adat Jawa', 'Sanggar Adat Kirana', 'Paket lengkap untuk upacara siraman dan midodareni, termasuk air 7 sumber dan dekorasi bunga.', 'ongoing', 'ceremony_siraman.jpg', '2025-11-29 17:50:01', 8000000),
('1e35cb80-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Seserahan & Tukar Cincin Adat Sunda', 'Pusat Budaya Parahyangan', 'Penyediaan nampan hantaran, MC adat, dan penataan prosesi tukar cincin.', 'ongoing', NULL, '2025-11-29 17:50:01', 4500000),
('1e35cbf2-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Adat Palang Pintu Betawi', 'Kelompok Palang Pintu Jaya', 'Pertunjukan palang pintu Betawi lengkap (silat dan pantun), 4 pasang pemain.', 'ongoing', 'ceremony_palang_pintu.jpg', '2025-11-29 17:50:01', 5500000),
('1e35cc3d-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Paket Upacara Adat Minang (Malam Bainai)', 'Raja Adat Minang', 'Perlengkapan untuk Malam Bainai, termasuk dekorasi dan pendamping adat.', 'ongoing', NULL, '2025-11-29 17:50:01', 6000000),
('1e35cc83-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Penyewaan Baki Seserahan (per nampan)', 'Sanggar Adat Kirana', 'Sewa nampan hantaran eksklusif (kaca, rotan, atau kayu ukir), minimal sewa 7 nampan.', 'ongoing', NULL, '2025-11-29 17:50:01', 150000),
('1e35ccc8-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Panggih Adat Jawa', 'Pusat Budaya Parahyangan', 'Prosesi Panggih lengkap, termasuk pelemparan sirih, injak telur, dan dulangan.', 'ongoing', 'ceremony_panggih.jpg', '2025-11-29 17:50:01', 3500000),
('1e35cd08-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Pemberkatan Adat Bali', 'Raja Adat Bali', 'Perlengkapan dan pendamping upacara adat Bali untuk pernikahan.', 'ongoing', 'ceremony_bali.jpg', '2025-11-29 17:50:01', 7000000),
('1e35cd59-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Jasa Cok bakal (Hiasan Hantaran)', 'Kelompok Palang Pintu Jaya', 'Jasa menghias dan menata barang-barang seserahan ke dalam nampan.', 'ongoing', NULL, '2025-11-29 17:50:01', 100000),
('1e35cd99-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Potong Gigi (Metatah) Bali', 'Raja Adat Bali', 'Paket upacara Metatah, termasuk perlengkapan dan pendeta (Pemangku).', 'airing', NULL, '2025-11-29 17:50:01', 9000000),
('1e35cdd9-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Sewa Baju Adat Pagar Ayu (Gadis Jawa)', 'Sanggar Adat Kirana', 'Sewa baju adat gadis Jawa lengkap untuk pagar ayu, setelan kebaya dan kain.', 'ongoing', NULL, '2025-11-29 17:50:01', 400000),
('1e35ce17-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Sungkeman (Iringan Musik Tradisional)', 'Pusat Budaya Parahyangan', 'Panduan prosesi sungkeman, termasuk iringan musik Siter atau Kecapi Suling.', 'ongoing', NULL, '2025-11-29 17:50:01', 2500000),
('1e35ce51-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Tujuh Bulanan (Mitoni) Jawa', 'Sanggar Adat Kirana', 'Paket Mitoni lengkap, termasuk siraman kembang 7 rupa dan dawet ayu.', 'ongoing', 'ceremony_mitoni.jpg', '2025-11-29 17:50:01', 7000000),
('1e35cfc2-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Adat Batak (Martumpol)', 'Raja Adat Batak', 'Persiapan dan panduan upacara martumpol dan sinamot.', 'ongoing', NULL, '2025-11-29 17:50:01', 10000000),
('1e35d010-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Sewa Payung Hias (Payung Geulis) Tradisional', 'Kelompok Palang Pintu Jaya', 'Sewa payung hias Geulis khas Sunda untuk dekorasi atau properti.', 'ongoing', NULL, '2025-11-29 17:50:01', 100000),
('1e35d050-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Jasa Penjaga Pintu (Pagar Bagus/Ayu) Adat', 'Sanggar Adat Kirana', 'Penyediaan 6 orang berpakaian adat untuk menyambut dan mengarahkan tamu.', 'ongoing', NULL, '2025-11-29 17:50:01', 3000000),
('1e35d08e-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Tedak Siten (Turun Tanah) Jawa', 'Pusat Budaya Parahyangan', 'Prosesi Tedak Siten lengkap, termasuk kurungan ayam dan tangga tebu wulung.', 'ongoing', NULL, '2025-11-29 17:50:01', 5500000),
('1e35d0c9-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Jasa Pembaca Doa Adat/Penghulu Non-Formal', 'Raja Adat Minang', 'Penyediaan pembaca doa atau pemandu acara ritual non-agama.', 'ongoing', NULL, '2025-11-29 17:50:01', 1500000),
('1e35d170-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Mapacci Adat Bugis', 'Raja Adat Bugis', 'Perlengkapan dan panduan upacara Mapacci sehari sebelum akad nikah.', 'ongoing', NULL, '2025-11-29 17:50:01', 4000000),
('1e35d1b4-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Sewa Perlengkapan Perang Kembang (Properti)', 'Kelompok Palang Pintu Jaya', 'Sewa properti kuda lumping dan gunungan untuk prosesi Perang Kembang.', 'airing', NULL, '2025-11-29 17:50:01', 1800000),
('1e35d22b-cd11-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Paket Upacara Seluruh Adat (Jawa Full Day)', 'Sanggar Adat Kirana', 'Paket ALL-IN Siraman, Midodareni, Panggih, dan Resepsi Adat Jawa dalam 1-2 hari.', 'ongoing', 'ceremony_adat_full.jpg', '2025-11-29 17:50:01', 15000000),
('4b0e415c-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pengantin Akad Nikah', 'Risma Beauty Art', 'Riasan flawless dan tahan lama untuk acara akad, termasuk retouch.', 'ongoing', 'mua_akad_01.jpg', '2025-11-29 17:44:06', 2500000),
('4b0e4c79-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Resepsi Adat Jawa', 'Jeng Sri Makeup', 'Riasan adat Jawa Paes Ageng/Solo Putri, lengkap dengan aksesoris.', 'ongoing', 'mua_resepsi_jawa.jpg', '2025-11-29 17:44:06', 4000000),
('4b0e4d33-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Hair Do & Sanggul Modern', 'Nina Hair Styling', 'Jasa penataan rambut dan sanggul modern untuk ibu/pendamping.', 'ongoing', NULL, '2025-11-29 17:44:06', 750000),
('4b0e4e55-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Touch Up MUA After Akad', 'Risma Beauty Art', 'Layanan touch up 3 jam setelah selesai akad.', 'ongoing', NULL, '2025-11-29 17:44:06', 500000),
('4b0e4e9a-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket Pre-Wedding Indoor', 'Diva Makeup Studio', 'Riasan natural untuk sesi foto pre-wedding di dalam studio.', 'ongoing', 'prewed_indoor_01.jpg', '2025-11-29 17:44:06', 1800000),
('4b0e4ef2-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'MUA & Hair Do Bridesmaid (per orang)', 'Risma Beauty Art', 'Riasan dan penataan rambut untuk pendamping pengantin.', 'ongoing', NULL, '2025-11-29 17:44:06', 350000),
('4b0e4f59-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'MUA Pesta/Wisuda (Layanan di Tempat)', 'Aura Makeup Team', 'Layanan riasan untuk acara pesta atau wisuda, panggilan ke lokasi.', 'ongoing', 'mua_wisuda.jpg', '2025-11-29 17:44:06', 450000),
('4b0e4f95-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA All Day (Akad & Resepsi)', 'Jeng Sri Makeup', 'Riasan lengkap untuk dua acara (pagi dan malam) dalam satu hari.', 'ongoing', 'mua_allday.jpg', '2025-11-29 17:44:06', 6000000),
('4b0e4fd1-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Trial Make Up Pengantin', 'Diva Makeup Studio', 'Sesi percobaan riasan dengan berbagai pilihan konsep.', 'ongoing', NULL, '2025-11-29 17:44:06', 800000),
('4b0e5006-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'MUA Pengantin Muslimah Berhijab', 'Risma Beauty Art', 'Riasan pengantin khusus muslimah termasuk penataan hijab premium.', 'ongoing', 'mua_muslimah.jpg', '2025-11-29 17:44:06', 3500000),
('4b0e503d-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Keluarga Inti (4 orang)', 'Nina Hair Styling', 'Riasan dan hair do untuk 4 anggota keluarga terdekat.', 'ongoing', NULL, '2025-11-29 17:44:06', 2000000),
('4b0e5077-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'MUA Adat Sunda Siger', 'Jeng Sri Makeup', 'Riasan pengantin adat Sunda Siger lengkap dengan aksesoris.', 'ongoing', 'mua_siger.jpg', '2025-11-29 17:44:06', 4200000),
('4b0e50b6-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'MUA & Hair Do Engagement (Tunangan)', 'Diva Makeup Studio', 'Riasan semi-glamour untuk acara lamaran/pertunangan.', 'ongoing', 'mua_lamaran.jpg', '2025-11-29 17:44:06', 1500000),
('4b0e50f2-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'MUA Panggilan Foto Komersial', 'Aura Makeup Team', 'Layanan MUA untuk keperluan pemotretan komersial/katalog.', 'airing', NULL, '2025-11-29 17:44:06', 1200000),
('4b0e512f-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pengantin Internasional', 'Risma Beauty Art', 'Riasan bergaya Barat/Internasional dengan teknik kontur premium.', 'ongoing', 'mua_internasional.jpg', '2025-11-29 17:44:06', 3800000),
('4b0e5169-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Jasa Hair Do Pria (Grooming)', 'Nina Hair Styling', 'Penataan rambut dan grooming ringan untuk pengantin pria.', 'ongoing', NULL, '2025-11-29 17:44:06', 300000),
('4b0e5232-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'MUA & Hijab Do Seragam', 'Aura Makeup Team', 'Riasan dan hijab seragam untuk panitia/penyambut tamu (min. 5 orang).', 'ongoing', NULL, '2025-11-29 17:44:06', 300000),
('4b0e5276-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Upgrade Bulu Mata Premium', 'Diva Makeup Studio', 'Upgrade dari bulu mata standar ke bulu mata handmade premium.', 'ongoing', NULL, '2025-11-29 17:44:06', 200000),
('4b0e52b0-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'MUA Simpel untuk Pesta Ulang Tahun', 'Risma Beauty Art', 'Riasan ringan dan segar untuk acara non-formal.', 'ongoing', NULL, '2025-11-29 17:44:06', 400000),
('4b0e52e6-cd10-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Exclusive (Full Day + 2x Retouch)', 'Jeng Sri Makeup', 'Layanan MUA eksklusif seharian penuh dengan 2 kali ganti look dan retouch.', 'ongoing', 'mua_exclusive.jpg', '2025-11-29 17:44:06', 8500000),
('601609a3-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Pernikahan Silver (300 Pax)', 'Puspita Sari Catering', 'Paket lengkap untuk 300 tamu, menu utama 2 jenis, 3 gubukan, free dessert.', 'ongoing', 'catering_silver.jpg', '2025-11-29 17:44:42', 25000000),
('601629a9-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Pernikahan Gold (500 Pax)', 'Mawar Indah Katering', 'Paket premium untuk 500 tamu, 3 menu utama, 5 gubukan spesial, dekorasi meja prasmanan.', 'ongoing', 'catering_gold.jpg', '2025-11-29 17:44:42', 45000000),
('60162a6c-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Gubukan Nasi Kebuli Kambing', 'Al Barokah Catering', 'Gubukan spesial Nasi Kebuli porsi 100 pax dengan lauk kambing oven.', 'ongoing', 'gubukan_kebuli.jpg', '2025-11-29 17:44:42', 6000000),
('60162ab7-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Catering Khitanan (200 Pax)', 'Puspita Sari Catering', 'Paket prasmanan untuk acara khitanan/syukuran, 4 pilihan lauk dan sayur.', 'ongoing', NULL, '2025-11-29 17:44:42', 15000000),
('60162b53-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Gubukan Soto Ayam Lamongan', 'Mawar Indah Katering', 'Gubukan Soto Ayam Lamongan porsi 150 mangkuk, lengkap dengan koya.', 'ongoing', 'gubukan_soto.jpg', '2025-11-29 17:44:42', 4500000),
('60162b9c-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Snack Box Rapat (Isi 3 kue)', 'Al Barokah Catering', 'Kotak snack untuk rapat atau acara kantor, pilihan kue tradisional dan modern.', 'ongoing', NULL, '2025-11-29 17:44:42', 18000),
('60162bd8-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Prasmanan Family Gathering (100 Pax)', 'Puspita Sari Catering', 'Menu prasmanan santai untuk acara keluarga/kantor, BBQ tambahan opsional.', 'ongoing', NULL, '2025-11-29 17:44:42', 9000000),
('60162c8a-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Gubukan Dimsum & Siomay', 'Mawar Indah Katering', 'Gubukan Dimsum kukus dan goreng, Siomay Bandung, porsi 120 porsi.', 'ongoing', 'gubukan_dimsum.jpg', '2025-11-29 17:44:42', 5500000),
('60162cc7-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Catering Tumpeng Mini Nasi Kuning (per box)', 'Al Barokah Catering', 'Tumpeng mini komplit dengan ayam suwir, telur balado, dan kering tempe.', 'ongoing', 'tumpeng_mini.jpg', '2025-11-29 17:44:42', 35000),
('60162d02-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Coffee Break & Teh (50 orang)', 'Puspita Sari Catering', 'Layanan coffee break termasuk kopi, teh, dan 2 jenis kudapan ringan.', 'airing', NULL, '2025-11-29 17:44:42', 3000000),
('60162d45-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Gubukan Bakso Komplit', 'Mawar Indah Katering', 'Gubukan Bakso Sapi komplit dengan mie, pangsit, dan sambal, porsi 150 mangkuk.', 'ongoing', 'gubukan_bakso.jpg', '2025-11-29 17:44:42', 4800000),
('601677c6-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Nasi Kotak Ayam Geprek (Lunch/Dinner)', 'Al Barokah Catering', 'Nasi kotak untuk makan siang/malam, lauk Ayam Geprek pedas.', 'ongoing', NULL, '2025-11-29 17:44:42', 22000),
('6016787e-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Dessert Corner Premium', 'Puspita Sari Catering', 'Pilihan dessert manis premium: pudding, mini tart, es krim, dan buah potong.', 'ongoing', 'dessert_corner.jpg', '2025-11-29 17:44:42', 7500000),
('601678cf-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Gubukan Pempek Palembang', 'Mawar Indah Katering', 'Gubukan Pempek Kapal Selam, Lenjer, dan Adaan, lengkap dengan kuah cuka.', 'ongoing', NULL, '2025-11-29 17:44:42', 3900000),
('60167916-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Catering Pernikahan Platinum (800 Pax)', 'Al Barokah Catering', 'Paket VIP 800 tamu, semua gubukan dan menu utama full service, dekorasi eksklusif.', 'ongoing', 'catering_platinum.jpg', '2025-11-29 17:44:42', 80000000),
('6016795e-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Gubukan Es Campur & Es Dawet', 'Puspita Sari Catering', 'Gubukan minuman segar tradisional Es Campur dan Es Dawet, porsi 200 cup.', 'ongoing', NULL, '2025-11-29 17:44:42', 3500000),
('6016799f-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Nasi Kotak Menu Oriental (Chicken Katsu)', 'Mawar Indah Katering', 'Nasi kotak dengan menu Chicken Katsu dan salad mayonaise.', 'ongoing', NULL, '2025-11-29 17:44:42', 28000),
('601679df-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Aqiqah (50 Box)', 'Al Barokah Catering', 'Layanan catering Aqiqah, termasuk pemotongan dan pengemasan dalam 50 box.', 'ongoing', NULL, '2025-11-29 17:44:42', 7000000),
('60167a1a-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Menu Tambahan: Salmon Panggang (per porsi)', 'Puspita Sari Catering', 'Tambahan menu premium Salmon Panggang dengan saus lemon butter.', 'airing', 'salmon_panggang.jpg', '2025-11-29 17:44:42', 95000),
('60167a58-cd10-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Gubukan Zuppa Soup', 'Mawar Indah Katering', 'Gubukan Zuppa Soup creamy dengan pastry renyah, porsi 100 mangkuk.', 'ongoing', 'gubukan_zuppa.jpg', '2025-11-29 17:44:42', 3800000),
('76483ef2-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Grand Ballroom Hotel Bintang 5', 'The Majestic Hotel', 'Kapasitas 1000 tamu, include sound system, lighting, dan standar dekorasi.', 'ongoing', 'venue_ballroom_grand.jpg', '2025-11-29 17:45:19', 150000000),
('76484382-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Outdoor Garden Pagi Hari', 'Lavender Garden', 'Area taman luas, cocok untuk akad nikah dan resepsi kecil (maks. 300 tamu).', 'ongoing', 'venue_garden_pagi.jpg', '2025-11-29 17:45:19', 35000000),
('76484427-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Serbaguna Kota', 'Pusat Convention Hall', 'Lokasi strategis di pusat kota, kapasitas 700 tamu, fasilitas parkir memadai.', 'ongoing', 'venue_gedung_serba.jpg', '2025-11-29 17:45:19', 40000000),
('7648448d-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Aula Pertemuan Kampus (Harga Mahasiswa)', 'Universitas Cendekia', 'Aula standar untuk wisuda/seminar, kapasitas 500 kursi. Harga diskon untuk internal.', 'ongoing', NULL, '2025-11-29 17:45:19', 15000000),
('764844f9-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Pinggir Pantai Eksklusif', 'Sunset Beach Resort', 'Venue tepi laut dengan pemandangan sunset, cocok untuk intimate wedding (maks. 150 tamu).', 'ongoing', 'venue_pantai_sunset.jpg', '2025-11-29 17:45:19', 75000000),
('76484574-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Private Villa dengan Kolam Renang', 'Villa Kemuning', 'Sewa villa 3 hari 2 malam untuk acara privat, kapasitas 100 standing party.', 'ongoing', 'venue_villa_pool.jpg', '2025-11-29 17:45:19', 30000000),
('76484607-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Rooftop dengan City View', 'Skylight Tower', 'Venue di lantai teratas gedung, pemandangan kota 360 derajat (maks. 250 tamu).', 'ongoing', 'venue_rooftop.jpg', '2025-11-29 17:45:19', 60000000),
('76484673-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ruang Meeting Kecil (10 orang)', 'Business Center X', 'Ruang rapat dilengkapi proyektor dan Wi-Fi, sewa per jam.', 'ongoing', NULL, '2025-11-29 17:45:19', 500000),
('764846d1-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ruang Pameran Seni Kontemporer', 'Galeri Adiwarna', 'Venue unik untuk pameran atau peluncuran produk.', 'ongoing', NULL, '2025-11-29 17:45:19', 20000000),
('7648472e-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Resepsi Sederhana di Restoran', 'Restoran Padang Jaya', 'Penyewaan lantai 2 restoran untuk acara private dinner (maks. 80 tamu).', 'ongoing', NULL, '2025-11-29 17:45:19', 5000000),
('764847d8-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Pernikahan Adat Jawa/Sunda (Pendopo)', 'Pendopo Agung Bhumi', 'Venue tradisional dengan arsitektur kayu, cocok untuk acara adat.', 'ongoing', 'venue_pendopo_adat.jpg', '2025-11-29 17:45:19', 48000000),
('7648a1ee-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Resepsi Malam Hari (dengan Lighting)', 'The Atrium Hall', 'Ruangan indoor mewah, free penggunaan lighting panggung dan FOH.', 'ongoing', NULL, '2025-11-29 17:45:19', 85000000),
('7648a346-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Olahraga (GOR) untuk Konser', 'GOR Patriot', 'Kapasitas besar, cocok untuk konser musik atau pameran skala besar.', 'airing', NULL, '2025-11-29 17:45:19', 120000000),
('7648a3d3-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Kapel Pernikahan Eksklusif', 'White Chapel', 'Gereja kecil/kapel untuk pemberkatan dan upacara intim (maks. 100 tamu).', 'ongoing', 'venue_chapel_white.jpg', '2025-11-29 17:45:19', 25000000),
('7648a43d-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Sewa Seluruh Area Kebun Binatang', 'Safari Park Malam', 'Venue sangat unik untuk gala dinner atau private party.', 'airing', NULL, '2025-11-29 17:45:19', 180000000),
('7648a4a9-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Outdoor Kolam Renang (Poolside)', 'The Oasis Club', 'Area di tepi kolam renang, cocok untuk pesta koktail dan ulang tahun.', 'ongoing', 'venue_poolside.jpg', '2025-11-29 17:45:19', 42000000),
('7648a543-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Aula Balai Desa/Kelurahan', 'Balai Warga Makmur', 'Venue sangat terjangkau untuk acara komunitas atau pernikahan sederhana.', 'ongoing', NULL, '2025-11-29 17:45:19', 5000000),
('7648a5b7-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Exhibition Hall (Harian)', 'International Expo Center', 'Sewa hall besar untuk pameran B2B atau B2C, harga sewa per hari.', 'airing', NULL, '2025-11-29 17:45:19', 95000000),
('7648a623-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Khusus Pernikahan (All-in One)', 'The Wedding Castle', 'Paket sewa venue yang sudah termasuk catering dan dekorasi dasar (min. 400 pax).', 'ongoing', 'venue_all_in.jpg', '2025-11-29 17:45:19', 90000000),
('7648a69a-cd10-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Restoran Mewah Klasik', 'Casa De Amor', 'Venue bergaya Eropa klasik, ideal untuk acara formal dan *fine dining* (maks. 120 tamu).', 'ongoing', NULL, '2025-11-29 17:45:19', 38000000),
('79b4ff45-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Sunda Bandung', 'Saung Angklung Udjo Bandung', 'Venue outdoor dan indoor untuk pernikahan adat sunda di Bandung, kapasitas 500 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 45000000),
('79b51fec-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Hotel Tradisional Sunda', 'Hotel Savoy Homann Bandung', 'Ballroom dengan nuansa sunda modern, kapasitas 800 tamu di pusat kota Bandung', 'ongoing', NULL, '2025-12-01 22:29:28', 75000000),
('79b52308-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Outdoor Lembang Sunda', 'The Lodge Maribaya Bandung', 'Venue outdoor dengan pemandangan alam khas sunda, kapasitas 300 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 35000000),
('79b5237c-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Serbaguna Adat Jawa Jogja', 'Sasana Budaya Jogjakarta', 'Venue tradisional jawa di Jogja, kapasitas 600 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 40000000),
('79b523cb-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Modern Jakarta Pusat', 'Grand Ballroom Plaza Indonesia', 'Ballroom mewah di Jakarta Pusat, kapasitas 1000 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 150000000),
('79b5241a-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Villa Pernikahan Bali Ubud', 'Villa Seminyak Bali', 'Villa eksklusif untuk intimate wedding di Bali, kapasitas 100 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 50000000),
('79b524f8-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Pantai Wedding Venue Bali', 'Sunset Beach Bali Resort', 'Venue tepi pantai untuk pernikahan di Bali, kapasitas 200 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 80000000),
('79b52559-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Minang Padang', 'Rumah Gadang Padang', 'Venue adat minang di Padang, kapasitas 400 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 38000000),
('79b52599-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Hotel Ballroom Semarang', 'Grand Ballroom Semarang Plaza', 'Ballroom hotel di Semarang, kapasitas 700 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 60000000),
('79b525dc-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Garden Bogor', 'Taman Bunga Bogor', 'Outdoor garden venue di Bogor, kapasitas 350 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 32000000),
('79b52635-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Pendopo Adat Jawa Solo', 'Pendopo Mangkunegaran Solo', 'Venue pendopo tradisional jawa di Solo, kapasitas 500 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 55000000),
('79b52675-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Rooftop Venue Jakarta Selatan', 'Sky Garden Jakarta', 'Rooftop venue dengan city view di Jakarta Selatan, kapasitas 250 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 65000000),
('79b526b7-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Adat Betawi Jakarta', 'Setu Babakan Jakarta', 'Venue adat betawi di Jakarta, kapasitas 450 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 42000000),
('79b526f4-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Villa Puncak Bogor', 'Villa Kharisma Puncak', 'Villa dengan view gunung di Puncak, kapasitas 200 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 38000000),
('79b52736-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Hotel Surabaya', 'Grand Ballroom Surabaya Plaza', 'Ballroom mewah di Surabaya, kapasitas 900 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 85000000),
('79b52775-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Tepi Sungai Bandung', 'Riverside Garden Bandung', 'Venue outdoor tepi sungai di Bandung, kapasitas 300 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 35000000),
('79b527ae-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Sunda Cirebon', 'Keraton Kasepuhan Cirebon', 'Venue adat sunda-jawa di Cirebon, kapasitas 550 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 48000000),
('79b527ec-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Hotel Resort Lombok', 'Beach Resort Lombok', 'Resort tepi pantai di Lombok, kapasitas 350 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 70000000),
('79b52829-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Perkebunan Teh Bandung', 'Kebun Teh Rancabali Bandung', 'Venue di perkebunan teh dengan suasana sunda, kapasitas 250 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 40000000),
('79b5286b-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Convention Tangerang', 'Convention Hall BSD Tangerang', 'Gedung modern di Tangerang, kapasitas 800 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 72000000),
('79b80e4b-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Premium 500 Pax', 'Catering Nasi Timbel Bandung', 'Menu khas sunda untuk 500 tamu di Bandung, nasi timbel, ayam bakar, ikan mas', 'ongoing', NULL, '2025-12-01 22:29:28', 42000000),
('79b81cf3-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Standard 300 Pax', 'Sari Sunda Catering Bandung', 'Menu sunda untuk 300 tamu, pepes ikan, ayam goreng, sayur asem', 'ongoing', NULL, '2025-12-01 22:29:28', 24000000),
('79b81d8c-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Eksklusif 800 Pax', 'Bumbu Desa Bandung', 'Menu sunda eksklusif untuk 800 tamu dengan live cooking', 'ongoing', NULL, '2025-12-01 22:29:28', 75000000),
('79b81de9-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Jawa 500 Pax', 'Omah Javanese Catering', 'Menu khas jawa untuk 500 tamu, gudeg, ayam bakar, tumpeng', 'ongoing', NULL, '2025-12-01 22:29:28', 38000000),
('79b81e32-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Minang 400 Pax', 'Sederhana Catering Padang', 'Menu padang untuk 400 tamu, rendang, gulai, ayam pop', 'ongoing', NULL, '2025-12-01 22:29:28', 35000000),
('79b81e77-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering International 600 Pax', 'Premier Catering Jakarta', 'Menu international buffet untuk 600 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 55000000),
('79b81eb8-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Betawi 350 Pax', 'Betawi Heritage Catering', 'Menu khas betawi untuk 350 tamu, kerak telor, soto betawi', 'ongoing', NULL, '2025-12-01 22:29:28', 30000000),
('79b81efb-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Seafood 500 Pax', 'Ocean Fresh Catering', 'Menu seafood premium untuk 500 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 52000000),
('79b81f3d-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Vegetarian 200 Pax', 'Green Garden Catering', 'Menu vegetarian untuk 200 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 18000000),
('79b81f7d-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Fusion 450 Pax', 'Fusion Taste Catering', 'Menu fusion indonesia-western untuk 450 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 42000000),
('79b81fb9-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering BBQ 300 Pax', 'BBQ Nation Catering', 'BBQ buffet untuk 300 tamu dengan live grilling', 'ongoing', NULL, '2025-12-01 22:29:28', 35000000),
('79b8202e-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Nusantara 700 Pax', 'Nusantara Kitchen Catering', 'Menu nusantara lengkap untuk 700 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 63000000),
('79b82076-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Budget 200 Pax', 'Warung Nasi Sunda', 'Menu sunda ekonomis untuk 200 tamu di Bandung', 'ongoing', NULL, '2025-12-01 22:29:28', 14000000),
('79b820b4-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Bali 400 Pax', 'Balinese Feast Catering', 'Menu khas bali untuk 400 tamu, babi guling, lawar', 'ongoing', NULL, '2025-12-01 22:29:28', 40000000),
('79b820f5-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Chinese 500 Pax', 'Golden Dragon Catering', 'Menu chinese untuk 500 tamu, dimsum, peking duck', 'ongoing', NULL, '2025-12-01 22:29:28', 48000000),
('79b82131-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Mediteranian 350 Pax', 'Mediterranean Delight', 'Menu mediterania untuk 350 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 45000000),
('79b8216e-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Plus Dessert 600 Pax', 'Premium Sunda Catering Bandung', 'Menu sunda lengkap dengan dessert corner untuk 600 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 58000000),
('79b824c0-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Japanese 300 Pax', 'Sakura Catering', 'Menu japanese untuk 300 tamu, sushi, ramen, teriyaki', 'ongoing', NULL, '2025-12-01 22:29:28', 42000000),
('79b82530-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Korean 250 Pax', 'Seoul Kitchen Catering', 'Menu korean BBQ untuk 250 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 35000000),
('79b82575-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Lampung 400 Pax', 'Seruit Catering Lampung', 'Menu khas lampung untuk 400 tamu', 'ongoing', NULL, '2025-12-01 22:29:28', 32000000),
('79baa591-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Sunda', 'Sundanese Wedding Decor Bandung', 'Dekorasi lengkap adat sunda dengan payung geulis, bale adat di Bandung', 'ongoing', NULL, '2025-12-01 22:29:28', 25000000),
('79bab144-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Sunda Premium', 'Parahyangan Decor Bandung', 'Dekorasi sunda premium dengan ornamen tradisional lengkap', 'ongoing', NULL, '2025-12-01 22:29:28', 35000000),
('79bab1e7-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Jawa', 'Javanese Decor Jogja', 'Dekorasi pelaminan adat jawa lengkap dengan bleketepe', 'ongoing', NULL, '2025-12-01 22:29:28', 28000000),
('79bab241-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Minimalis Modern', 'Modern Decor Jakarta', 'Dekorasi minimalis modern dengan bunga segar', 'ongoing', NULL, '2025-12-01 22:29:28', 22000000),
('79bab28e-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Rustic Outdoor', 'Rustic Garden Decor', 'Dekorasi rustic untuk outdoor wedding dengan kayu dan burlap', 'ongoing', NULL, '2025-12-01 22:29:28', 30000000),
('79bab2d7-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Garden Party', 'Green Paradise Decor', 'Dekorasi garden party dengan hanging flowers', 'ongoing', NULL, '2025-12-01 22:29:28', 28000000),
('79bab318-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Ballroom Mewah', 'Luxury Ballroom Decor', 'Dekorasi ballroom mewah dengan crystal chandelier', 'ongoing', NULL, '2025-12-01 22:29:28', 45000000),
('79bab35a-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Betawi', 'Betawi Decor Jakarta', 'Dekorasi pernikahan adat betawi dengan ondel-ondel', 'ongoing', NULL, '2025-12-01 22:29:28', 24000000),
('79bab396-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Minang', 'Minang Decor Padang', 'Dekorasi adat minang dengan ornamen khas rumah gadang', 'ongoing', NULL, '2025-12-01 22:29:28', 26000000),
('79bab3d3-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pantai Beach Wedding', 'Beach Wedding Decor Bali', 'Dekorasi pernikahan tepi pantai di Bali', 'ongoing', NULL, '2025-12-01 22:29:28', 32000000),
('79bab410-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Vintage Classic', 'Vintage Decor Studio', 'Dekorasi vintage dengan barang-barang antik', 'ongoing', NULL, '2025-12-01 22:29:28', 27000000),
('79bab450-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Bohemian', 'Boho Chic Decor', 'Dekorasi bohemian dengan macrame dan dreamcatcher', 'ongoing', NULL, '2025-12-01 22:29:28', 25000000),
('79bab48f-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi International Elegant', 'Elite International Decor', 'Dekorasi international dengan bunga import', 'ongoing', NULL, '2025-12-01 22:29:28', 50000000),
('79bab4cf-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Sunda Minimalis Bandung', 'Sunda Modern Decor', 'Perpaduan dekorasi sunda dan minimalis modern di Bandung', 'ongoing', NULL, '2025-12-01 22:29:28', 30000000),
('79bab510-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Bali', 'Balinese Decor Bali', 'Dekorasi pernikahan adat bali lengkap', 'ongoing', NULL, '2025-12-01 22:29:28', 38000000),
('79bab55f-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pelaminan Jawa Tengah', 'Javanese Central Decor', 'Dekorasi pelaminan jawa tengah dengan ukiran khas', 'ongoing', NULL, '2025-12-01 22:29:28', 29000000),
('79bab5a3-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Modern Glamour', 'Glamour Decor Jakarta', 'Dekorasi modern glamour dengan lighting dramatis', 'ongoing', NULL, '2025-12-01 22:29:28', 42000000),
('79bab5e0-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Traditional Nusantara', 'Nusantara Heritage Decor', 'Dekorasi traditional nusantara campuran berbagai adat', 'ongoing', NULL, '2025-12-01 22:29:28', 33000000),
('79bab61d-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Garden Sunda Bandung', 'Garden Sunda Decor', 'Dekorasi outdoor garden dengan tema sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:29:28', 28000000),
('79bab65d-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Romantic Pink', 'Romantic Pink Decor', 'Dekorasi romantis dengan dominan pink dan putih', 'ongoing', NULL, '2025-12-01 22:29:28', 26000000),
('79bcfdf9-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pengantin Adat Sunda', 'Sunda Beauty Bandung', 'Riasan pengantin adat sunda dengan siger khas Bandung', 'ongoing', NULL, '2025-12-01 22:29:28', 4500000),
('79bd0f51-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Sunda Siger Premium', 'Parahyangan Makeup Artist', 'Riasan sunda siger premium dengan aksesoris lengkap di Bandung', 'ongoing', NULL, '2025-12-01 22:29:28', 6000000),
('79bd1062-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Sunda Modern', 'Modern Sunda MUA Bandung', 'Riasan sunda modern untuk pengantin di Bandung', 'ongoing', NULL, '2025-12-01 22:29:28', 4000000),
('79bd10dd-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Jawa Paes Ageng', 'Javanese MUA Jogja', 'Riasan pengantin jawa paes ageng lengkap', 'ongoing', NULL, '2025-12-01 22:29:28', 5500000),
('79bd11d1-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA International Style', 'Elite Makeup Jakarta', 'Riasan pengantin gaya international modern', 'ongoing', NULL, '2025-12-01 22:29:28', 7000000),
('79bd1230-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Betawi', 'Betawi MUA Jakarta', 'Riasan pengantin adat betawi lengkap', 'ongoing', NULL, '2025-12-01 22:29:28', 4200000),
('79bd1295-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Minang', 'Minang Beauty Padang', 'Riasan pengantin adat minang dengan sunting', 'ongoing', NULL, '2025-12-01 22:29:28', 4800000),
('79bd12f3-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Bali', 'Balinese Makeup Bali', 'Riasan pengantin adat bali lengkap', 'ongoing', NULL, '2025-12-01 22:29:28', 5000000),
('79bd134c-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Minimalis Natural', 'Natural Beauty Studio', 'Riasan minimalis natural untuk pengantin', 'ongoing', NULL, '2025-12-01 22:29:28', 3500000),
('79bd13a4-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Korean Style', 'K-Beauty Makeup', 'Riasan pengantin gaya korea', 'ongoing', NULL, '2025-12-01 22:29:28', 4500000),
('79bd13ff-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Akad dan Resepsi Sunda', 'Bandung Bridal MUA', 'Paket lengkap MUA untuk akad dan resepsi adat sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:29:28', 8000000),
('79bd1473-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Hijab Syari', 'Hijab Beauty Studio', 'Riasan pengantin muslimah dengan hijab syari', 'ongoing', NULL, '2025-12-01 22:29:28', 4000000),
('79bd14c7-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Glam Party', 'Glam Studio Makeup', 'Riasan glamour untuk resepsi pesta', 'ongoing', NULL, '2025-12-01 22:29:28', 5500000),
('79bd151e-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pre-Wedding Sunda', 'Sunda Pre-Wed MUA', 'Riasan untuk sesi foto pre-wedding tema sunda', 'ongoing', NULL, '2025-12-01 22:29:28', 2500000),
('79bd1573-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Bridesmaid Sunda', 'Sunda Party Makeup', 'Riasan untuk bridesmaid tema sunda (per orang)', 'ongoing', NULL, '2025-12-01 22:29:28', 500000),
('79bd160a-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Family Complete', 'Family Makeup Package', 'Riasan untuk keluarga pengantin (5 orang)', 'ongoing', NULL, '2025-12-01 22:29:28', 3000000),
('79bd1670-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Jawa Solo Basahan', 'Solo MUA Jogja', 'Riasan adat jawa solo basahan lengkap', 'ongoing', NULL, '2025-12-01 22:29:28', 5200000),
('79bd1776-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Lamaran Sunda', 'Sunda Engagement MUA', 'Riasan untuk acara lamaran adat sunda', 'ongoing', NULL, '2025-12-01 22:29:28', 2800000),
('79bd35ba-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Trial Make Up', 'Professional Trial MUA', 'Sesi trial makeup sebelum hari H', 'ongoing', NULL, '2025-12-01 22:29:28', 1000000),
('79bd374e-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Luxury Premium', 'Luxury Bridal MUA', 'Paket MUA premium dengan produk luxury brand', 'ongoing', NULL, '2025-12-01 22:29:28', 12000000),
('86feb410-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Baju Pengantin Adat Sunda Siger', 'Sanggar Busana Kenanga', 'Sewa lengkap busana pengantin Sunda Siger, termasuk aksesoris dan kain.', 'ongoing', 'attire_sunda_siger.jpg', '2025-11-29 17:45:47', 5000000),
('86febd92-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Gaun Pengantin Internasional Putih', 'House of Bridal Elegance', 'Sewa gaun model ballgown lace premium, free veil dan petticoat.', 'ongoing', 'attire_ballgown_white.jpg', '2025-11-29 17:45:47', 7500000),
('86febe26-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Batik Sarimbit Eksklusif', 'Batik Prabu', 'Batik couple (pria dan wanita) untuk acara keluarga atau resepsi.', 'ongoing', NULL, '2025-11-29 17:45:47', 1500000),
('86febe73-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Jas Pengantin Pria Tuxedo Hitam', 'Gentleman Tailor', 'Sewa set jas Tuxedo 3-piece, termasuk dasi kupu-kupu dan vest.', 'ongoing', 'attire_tuxedo_hitam.jpg', '2025-11-29 17:45:47', 2500000),
('86fee125-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Kebaya Akad Nikah Putih Modern', 'Sanggar Busana Kenanga', 'Sewa kebaya akad kutubaru modern dengan payet premium.', 'ongoing', 'attire_kebaya_akad.jpg', '2025-11-29 17:45:47', 3000000),
('86fee1cd-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Busana Adat Minang (Baralek)', 'Busana Tradisi Minang', 'Sewa busana lengkap pengantin Minang (Suntiang dan Saluak).', 'ongoing', 'attire_minang.jpg', '2025-11-29 17:45:47', 6500000),
('86fee216-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Gaun Pesta Bridesmaid (Sewa per Pcs)', 'House of Bridal Elegance', 'Sewa gaun seragam untuk bridesmaid, tersedia 5 pilihan warna.', 'ongoing', NULL, '2025-11-29 17:45:47', 450000),
('86fee267-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Beskap Jawa Solo Basahan', 'Batik Prabu', 'Sewa Beskap Solo Basahan lengkap dengan blangkon dan keris.', 'ongoing', 'attire_beskap_jawa.jpg', '2025-11-29 17:45:47', 3800000),
('86fee2db-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Busana Adat Bali Payas Agung', 'Busana Tradisi Bali', 'Sewa busana lengkap adat Bali Payas Agung, termasuk mahkota.', 'ongoing', 'attire_bali.jpg', '2025-11-29 17:45:47', 7000000),
('86fee326-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Kebaya Modern Wisuda/Pesta', 'Sanggar Busana Kenanga', 'Sewa kebaya pendek modern untuk wisuda atau acara semi-formal.', 'ongoing', NULL, '2025-11-29 17:45:47', 750000),
('86fee37b-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Gaun Malam (Evening Dress) Merah', 'House of Bridal Elegance', 'Sewa gaun malam panjang berbahan satin silk untuk resepsi.', 'ongoing', 'attire_evening_dress.jpg', '2025-11-29 17:45:47', 4200000),
('86fee3c3-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Aksesoris Headpiece Pengantin (Sewa)', 'Gentleman Tailor', 'Sewa mahkota, tiara, atau headpiece kristal untuk pengantin.', 'ongoing', NULL, '2025-11-29 17:45:47', 800000),
('86fee3ff-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Gaun Pre-Wedding Bohemian Style', 'Busana Tradisi Bali', 'Sewa gaun ringan dan flowy untuk sesi foto pre-wedding outdoor.', 'ongoing', 'attire_prewed_boho.jpg', '2025-11-29 17:45:47', 1800000),
('86fee4b4-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Busana Adat Bugis Makassar', 'Busana Tradisi Makassar', 'Sewa Baju Bodo dan Passapu lengkap untuk pengantin Bugis.', 'ongoing', 'attire_bugis.jpg', '2025-11-29 17:45:47', 6000000);
INSERT INTO `items` (`id`, `category_id`, `name`, `vendor`, `description`, `status`, `image_url`, `last_update`, `price`) VALUES
('86ff1efe-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Kebaya Ibu Pengantin (Sewa 1 set)', 'Sanggar Busana Kenanga', 'Sewa kebaya dan kain untuk Ibu Pengantin (Besan), motif seragam.', 'ongoing', NULL, '2025-11-29 17:45:47', 2000000),
('86ff1ff7-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Gaun Pengantin Internasional (Mermaid Cut)', 'House of Bridal Elegance', 'Sewa gaun model mermaid fit-body, cocok untuk resepsi modern.', 'ongoing', 'attire_mermaid_cut.jpg', '2025-11-29 17:45:47', 5500000),
('86ff206a-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Vest dan Kemeja Pria (Acara Non-Formal)', 'Gentleman Tailor', 'Sewa vest dan kemeja polos untuk pendamping atau MC.', 'ongoing', NULL, '2025-11-29 17:45:47', 600000),
('86ff20d8-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Kain Songket Palembang Asli', 'Busana Tradisi Palembang', 'Sewa kain songket tenun tangan premium untuk akad/resepsi.', 'ongoing', 'attire_songket.jpg', '2025-11-29 17:45:47', 3500000),
('86ff2138-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Gaun Anak (Flower Girl/Pagar Ayu)', 'House of Bridal Elegance', 'Sewa gaun mini untuk anak-anak, warna dan model bisa custom.', 'ongoing', NULL, '2025-11-29 17:45:47', 300000),
('86ff21a4-cd10-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Pakaian Tradisional Betawi (Abang None)', 'Sanggar Busana Kenanga', 'Sewa pakaian adat Betawi lengkap untuk penyambutan tamu atau pre-wed.', 'ongoing', 'attire_betawi_abnon.jpg', '2025-11-29 17:45:47', 2200000),
('9acb3438-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Paket Dekorasi Pernikahan Adat Jawa', 'Melati Dekorasi', 'Dekorasi pelaminan adat Jawa, termasuk gebyok ukir, bunga segar, dan panggung.', 'ongoing', 'deco_adat_jawa.jpg', '2025-11-29 17:46:20', 20000000),
('9acb5503-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Paket Dekorasi Pernikahan Internasional Modern', 'Vivid Dreams Decoration', 'Dekorasi modern minimalis, full lighting, backdrop floral wall, dan kristal.', 'ongoing', 'deco_modern_kristal.jpg', '2025-11-29 17:46:20', 35000000),
('9acb563a-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Meja Akad Nikah Sederhana', 'Rizky Decoration', 'Dekorasi meja untuk akad, termasuk bunga kecil, taplak premium, dan kursi tiffany.', 'ongoing', NULL, '2025-11-29 17:46:20', 4000000),
('9acb56dc-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Backdrop Photobooth Rustic Vintage', 'Melati Dekorasi', 'Area photobooth dengan kayu, lampu gantung, dan ornamen vintage, sewa 6 jam.', 'ongoing', 'deco_photobooth_rustic.jpg', '2025-11-29 17:46:20', 3000000),
('9acb575d-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Panggung Musik/Hiburan', 'Vivid Dreams Decoration', 'Dekorasi khusus untuk panggung band atau orkestra, backdrop LED opsional.', 'ongoing', NULL, '2025-11-29 17:46:20', 8000000),
('9acb57ed-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Karpet Merah (Red Carpet) Premium', 'Rizky Decoration', 'Sewa karpet merah lebar 2 meter panjang 20 meter, termasuk tiang pembatas.', 'ongoing', NULL, '2025-11-29 17:46:20', 1500000),
('9acb5850-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Plafon Kain Drapery', 'Melati Dekorasi', 'Penataan kain drapery di langit-langit gedung, meliputi area resepsi utama.', 'ongoing', 'deco_plafon_drapery.jpg', '2025-11-29 17:46:20', 10000000),
('9acb58b3-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Lorong Masuk (Wedding Gate)', 'Vivid Dreams Decoration', 'Dekorasi gerbang/pintu masuk utama dengan bunga artifisial dan tirai.', 'ongoing', NULL, '2025-11-29 17:46:20', 5000000),
('9acb5913-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Lampu Taman Fairy Light (Sewa per titik)', 'Rizky Decoration', 'Sewa lampu tumblr/fairy light untuk dekorasi outdoor atau garden party.', 'ongoing', NULL, '2025-11-29 17:46:20', 50000),
('9acb5a58-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Papan Sambutan (Welcome Sign) Kayu Rustic', 'Melati Dekorasi', 'Sewa papan sambutan (standing board) berbahan kayu rustic custom nama.', 'ongoing', NULL, '2025-11-29 17:46:20', 800000),
('9acb9586-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pelaminan Minimalis (3x4m)', 'Vivid Dreams Decoration', 'Dekorasi pelaminan sederhana untuk rumah, bunga artifisial, dan lampu sorot.', 'ongoing', 'deco_minimalis_rumah.jpg', '2025-11-29 17:46:20', 12000000),
('9acb9697-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Gubukan Makanan (per unit)', 'Rizky Decoration', 'Dekorasi untuk meja gubukan, termasuk skirting, hiasan bunga, dan lampu sorot kecil.', 'ongoing', NULL, '2025-11-29 17:46:20', 750000),
('9acb9706-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Altar Pemberkatan (White Theme)', 'Melati Dekorasi', 'Dekorasi khusus altar atau kapel, full bunga segar putih, lilin, dan tirai.', 'ongoing', 'deco_altar_white.jpg', '2025-11-29 17:46:20', 9000000),
('9acb9766-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Penyewaan Kursi Tiffany (per buah)', 'Vivid Dreams Decoration', 'Sewa kursi tiffany warna emas atau putih, sudah termasuk bantal duduk.', 'ongoing', NULL, '2025-11-29 17:46:20', 30000),
('9acbaa2c-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Balon Ulang Tahun Anak', 'Rizky Decoration', 'Dekorasi tema karakter anak, termasuk gate balon dan standing balon 5 titik.', 'ongoing', NULL, '2025-11-29 17:46:20', 2500000),
('9acbaab7-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Photobooth 360 Video', 'Melati Dekorasi', 'Dekorasi backdrop khusus untuk platform video 360 derajat.', 'airing', NULL, '2025-11-29 17:46:20', 4500000),
('9acbab08-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Papan Bunga Ucapan Selamat (Sewa 1 hari)', 'Vivid Dreams Decoration', 'Sewa papan bunga ucapan selamat ukuran standar.', 'ongoing', NULL, '2025-11-29 17:46:20', 350000),
('9acbab50-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pre-Wedding Studio Minimalis', 'Rizky Decoration', 'Set dekorasi studio foto minimalis, termasuk 3 properti utama.', 'ongoing', NULL, '2025-11-29 17:46:20', 1000000),
('9acbab91-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pelaminan Outdoor Bohemien', 'Melati Dekorasi', 'Dekorasi pelaminan dengan macrame, dream catcher, dan bunga kering/pampas grass.', 'ongoing', 'deco_outdoor_boho.jpg', '2025-11-29 17:46:20', 28000000),
('9acbabea-cd10-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Lampu Sorot Panggung Par LED (Sewa per unit)', 'Vivid Dreams Decoration', 'Sewa lampu sorot panggung Par LED, termasuk kabel dan instalasi.', 'ongoing', NULL, '2025-11-29 17:46:20', 150000),
('a1b4849b-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Sunda Bandung', 'Saung Angklung Udjo Bandung', 'Venue outdoor dan indoor untuk pernikahan adat sunda di Bandung, kapasitas 500 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 45000000),
('a1b4a664-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Hotel Tradisional Sunda', 'Hotel Savoy Homann Bandung', 'Ballroom dengan nuansa sunda modern, kapasitas 800 tamu di pusat kota Bandung', 'ongoing', NULL, '2025-12-01 22:30:35', 75000000),
('a1b4ca88-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Outdoor Lembang Sunda', 'The Lodge Maribaya Bandung', 'Venue outdoor dengan pemandangan alam khas sunda, kapasitas 300 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 35000000),
('a1b4cb3a-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Serbaguna Adat Jawa Jogja', 'Sasana Budaya Jogjakarta', 'Venue tradisional jawa di Jogja, kapasitas 600 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 40000000),
('a1b4cb95-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Modern Jakarta Pusat', 'Grand Ballroom Plaza Indonesia', 'Ballroom mewah di Jakarta Pusat, kapasitas 1000 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 150000000),
('a1b4cbee-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Villa Pernikahan Bali Ubud', 'Villa Seminyak Bali', 'Villa eksklusif untuk intimate wedding di Bali, kapasitas 100 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 50000000),
('a1b4cc3e-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Pantai Wedding Venue Bali', 'Sunset Beach Bali Resort', 'Venue tepi pantai untuk pernikahan di Bali, kapasitas 200 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 80000000),
('a1b4cc86-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Minang Padang', 'Rumah Gadang Padang', 'Venue adat minang di Padang, kapasitas 400 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 38000000),
('a1b4ccca-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Hotel Ballroom Semarang', 'Grand Ballroom Semarang Plaza', 'Ballroom hotel di Semarang, kapasitas 700 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 60000000),
('a1b4cd0f-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Garden Bogor', 'Taman Bunga Bogor', 'Outdoor garden venue di Bogor, kapasitas 350 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 32000000),
('a1b4cd57-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Pendopo Adat Jawa Solo', 'Pendopo Mangkunegaran Solo', 'Venue pendopo tradisional jawa di Solo, kapasitas 500 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 55000000),
('a1b4cd99-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Rooftop Venue Jakarta Selatan', 'Sky Garden Jakarta', 'Rooftop venue dengan city view di Jakarta Selatan, kapasitas 250 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 65000000),
('a1b4cde0-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Adat Betawi Jakarta', 'Setu Babakan Jakarta', 'Venue adat betawi di Jakarta, kapasitas 450 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 42000000),
('a1b4ce24-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Villa Puncak Bogor', 'Villa Kharisma Puncak', 'Villa dengan view gunung di Puncak, kapasitas 200 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 38000000),
('a1b4ce69-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Hotel Surabaya', 'Grand Ballroom Surabaya Plaza', 'Ballroom mewah di Surabaya, kapasitas 900 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 85000000),
('a1b4ceae-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Tepi Sungai Bandung', 'Riverside Garden Bandung', 'Venue outdoor tepi sungai di Bandung, kapasitas 300 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 35000000),
('a1b4cf02-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Sunda Cirebon', 'Keraton Kasepuhan Cirebon', 'Venue adat sunda-jawa di Cirebon, kapasitas 550 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 48000000),
('a1b4cf48-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Hotel Resort Lombok', 'Beach Resort Lombok', 'Resort tepi pantai di Lombok, kapasitas 350 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 70000000),
('a1b4cf87-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Perkebunan Teh Bandung', 'Kebun Teh Rancabali Bandung', 'Venue di perkebunan teh dengan suasana sunda, kapasitas 250 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 40000000),
('a1b4cfdb-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Convention Tangerang', 'Convention Hall BSD Tangerang', 'Gedung modern di Tangerang, kapasitas 800 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 72000000),
('a1b798d2-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Premium 500 Pax', 'Catering Nasi Timbel Bandung', 'Menu khas sunda untuk 500 tamu di Bandung, nasi timbel, ayam bakar, ikan mas', 'ongoing', NULL, '2025-12-01 22:30:35', 42000000),
('a1b7a930-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Standard 300 Pax', 'Sari Sunda Catering Bandung', 'Menu sunda untuk 300 tamu, pepes ikan, ayam goreng, sayur asem', 'ongoing', NULL, '2025-12-01 22:30:35', 24000000),
('a1b7a9d1-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Eksklusif 800 Pax', 'Bumbu Desa Bandung', 'Menu sunda eksklusif untuk 800 tamu dengan live cooking', 'ongoing', NULL, '2025-12-01 22:30:35', 75000000),
('a1b7aa31-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Jawa 500 Pax', 'Omah Javanese Catering', 'Menu khas jawa untuk 500 tamu, gudeg, ayam bakar, tumpeng', 'ongoing', NULL, '2025-12-01 22:30:35', 38000000),
('a1b7aa7b-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Minang 400 Pax', 'Sederhana Catering Padang', 'Menu padang untuk 400 tamu, rendang, gulai, ayam pop', 'ongoing', NULL, '2025-12-01 22:30:35', 35000000),
('a1b7aac3-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering International 600 Pax', 'Premier Catering Jakarta', 'Menu international buffet untuk 600 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 55000000),
('a1b7ab06-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Betawi 350 Pax', 'Betawi Heritage Catering', 'Menu khas betawi untuk 350 tamu, kerak telor, soto betawi', 'ongoing', NULL, '2025-12-01 22:30:35', 30000000),
('a1b7ab51-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Seafood 500 Pax', 'Ocean Fresh Catering', 'Menu seafood premium untuk 500 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 52000000),
('a1b7ab97-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Vegetarian 200 Pax', 'Green Garden Catering', 'Menu vegetarian untuk 200 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 18000000),
('a1b7abd8-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Fusion 450 Pax', 'Fusion Taste Catering', 'Menu fusion indonesia-western untuk 450 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 42000000),
('a1b7ac19-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering BBQ 300 Pax', 'BBQ Nation Catering', 'BBQ buffet untuk 300 tamu dengan live grilling', 'ongoing', NULL, '2025-12-01 22:30:35', 35000000),
('a1b7ac58-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Nusantara 700 Pax', 'Nusantara Kitchen Catering', 'Menu nusantara lengkap untuk 700 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 63000000),
('a1b7aca4-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Budget 200 Pax', 'Warung Nasi Sunda', 'Menu sunda ekonomis untuk 200 tamu di Bandung', 'ongoing', NULL, '2025-12-01 22:30:35', 14000000),
('a1b7ace8-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Bali 400 Pax', 'Balinese Feast Catering', 'Menu khas bali untuk 400 tamu, babi guling, lawar', 'ongoing', NULL, '2025-12-01 22:30:35', 40000000),
('a1b7ad28-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Chinese 500 Pax', 'Golden Dragon Catering', 'Menu chinese untuk 500 tamu, dimsum, peking duck', 'ongoing', NULL, '2025-12-01 22:30:35', 48000000),
('a1b7ad6a-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Mediteranian 350 Pax', 'Mediterranean Delight', 'Menu mediterania untuk 350 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 45000000),
('a1b7adaa-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Plus Dessert 600 Pax', 'Premium Sunda Catering Bandung', 'Menu sunda lengkap dengan dessert corner untuk 600 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 58000000),
('a1b7adeb-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Japanese 300 Pax', 'Sakura Catering', 'Menu japanese untuk 300 tamu, sushi, ramen, teriyaki', 'ongoing', NULL, '2025-12-01 22:30:35', 42000000),
('a1b7aecc-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Korean 250 Pax', 'Seoul Kitchen Catering', 'Menu korean BBQ untuk 250 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 35000000),
('a1b7af56-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Lampung 400 Pax', 'Seruit Catering Lampung', 'Menu khas lampung untuk 400 tamu', 'ongoing', NULL, '2025-12-01 22:30:35', 32000000),
('a1ba1d5e-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Sunda', 'Sundanese Wedding Decor Bandung', 'Dekorasi lengkap adat sunda dengan payung geulis, bale adat di Bandung', 'ongoing', NULL, '2025-12-01 22:30:35', 25000000),
('a1ba311d-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Sunda Premium', 'Parahyangan Decor Bandung', 'Dekorasi sunda premium dengan ornamen tradisional lengkap', 'ongoing', NULL, '2025-12-01 22:30:35', 35000000),
('a1ba3217-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Jawa', 'Javanese Decor Jogja', 'Dekorasi pelaminan adat jawa lengkap dengan bleketepe', 'ongoing', NULL, '2025-12-01 22:30:35', 28000000),
('a1ba3275-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Minimalis Modern', 'Modern Decor Jakarta', 'Dekorasi minimalis modern dengan bunga segar', 'ongoing', NULL, '2025-12-01 22:30:35', 22000000),
('a1ba32c1-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Rustic Outdoor', 'Rustic Garden Decor', 'Dekorasi rustic untuk outdoor wedding dengan kayu dan burlap', 'ongoing', NULL, '2025-12-01 22:30:35', 30000000),
('a1ba330e-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Garden Party', 'Green Paradise Decor', 'Dekorasi garden party dengan hanging flowers', 'ongoing', NULL, '2025-12-01 22:30:35', 28000000),
('a1ba3350-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Ballroom Mewah', 'Luxury Ballroom Decor', 'Dekorasi ballroom mewah dengan crystal chandelier', 'ongoing', NULL, '2025-12-01 22:30:35', 45000000),
('a1ba3390-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Betawi', 'Betawi Decor Jakarta', 'Dekorasi pernikahan adat betawi dengan ondel-ondel', 'ongoing', NULL, '2025-12-01 22:30:35', 24000000),
('a1ba33d1-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Minang', 'Minang Decor Padang', 'Dekorasi adat minang dengan ornamen khas rumah gadang', 'ongoing', NULL, '2025-12-01 22:30:35', 26000000),
('a1ba340f-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pantai Beach Wedding', 'Beach Wedding Decor Bali', 'Dekorasi pernikahan tepi pantai di Bali', 'ongoing', NULL, '2025-12-01 22:30:35', 32000000),
('a1ba3450-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Vintage Classic', 'Vintage Decor Studio', 'Dekorasi vintage dengan barang-barang antik', 'ongoing', NULL, '2025-12-01 22:30:35', 27000000),
('a1ba348f-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Bohemian', 'Boho Chic Decor', 'Dekorasi bohemian dengan macrame dan dreamcatcher', 'ongoing', NULL, '2025-12-01 22:30:35', 25000000),
('a1ba34cc-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi International Elegant', 'Elite International Decor', 'Dekorasi international dengan bunga import', 'ongoing', NULL, '2025-12-01 22:30:35', 50000000),
('a1ba350d-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Sunda Minimalis Bandung', 'Sunda Modern Decor', 'Perpaduan dekorasi sunda dan minimalis modern di Bandung', 'ongoing', NULL, '2025-12-01 22:30:35', 30000000),
('a1ba354e-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Bali', 'Balinese Decor Bali', 'Dekorasi pernikahan adat bali lengkap', 'ongoing', NULL, '2025-12-01 22:30:35', 38000000),
('a1ba359d-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pelaminan Jawa Tengah', 'Javanese Central Decor', 'Dekorasi pelaminan jawa tengah dengan ukiran khas', 'ongoing', NULL, '2025-12-01 22:30:35', 29000000),
('a1ba35e0-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Modern Glamour', 'Glamour Decor Jakarta', 'Dekorasi modern glamour dengan lighting dramatis', 'ongoing', NULL, '2025-12-01 22:30:35', 42000000),
('a1ba3622-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Traditional Nusantara', 'Nusantara Heritage Decor', 'Dekorasi traditional nusantara campuran berbagai adat', 'ongoing', NULL, '2025-12-01 22:30:35', 33000000),
('a1ba3664-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Garden Sunda Bandung', 'Garden Sunda Decor', 'Dekorasi outdoor garden dengan tema sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:30:35', 28000000),
('a1ba36a9-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Romantic Pink', 'Romantic Pink Decor', 'Dekorasi romantis dengan dominan pink dan putih', 'ongoing', NULL, '2025-12-01 22:30:35', 26000000),
('a1bc7593-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pengantin Adat Sunda', 'Sunda Beauty Bandung', 'Riasan pengantin adat sunda dengan siger khas Bandung', 'ongoing', NULL, '2025-12-01 22:30:35', 4500000),
('a1bc85e1-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Sunda Siger Premium', 'Parahyangan Makeup Artist', 'Riasan sunda siger premium dengan aksesoris lengkap di Bandung', 'ongoing', NULL, '2025-12-01 22:30:35', 6000000),
('a1bc9c56-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Sunda Modern', 'Modern Sunda MUA Bandung', 'Riasan sunda modern untuk pengantin di Bandung', 'ongoing', NULL, '2025-12-01 22:30:35', 4000000),
('a1bc9ce4-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Jawa Paes Ageng', 'Javanese MUA Jogja', 'Riasan pengantin jawa paes ageng lengkap', 'ongoing', NULL, '2025-12-01 22:30:35', 5500000),
('a1bc9d36-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA International Style', 'Elite Makeup Jakarta', 'Riasan pengantin gaya international modern', 'ongoing', NULL, '2025-12-01 22:30:35', 7000000),
('a1bc9d84-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Betawi', 'Betawi MUA Jakarta', 'Riasan pengantin adat betawi lengkap', 'ongoing', NULL, '2025-12-01 22:30:35', 4200000),
('a1bc9dd2-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Minang', 'Minang Beauty Padang', 'Riasan pengantin adat minang dengan sunting', 'ongoing', NULL, '2025-12-01 22:30:35', 4800000),
('a1bc9e1a-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Bali', 'Balinese Makeup Bali', 'Riasan pengantin adat bali lengkap', 'ongoing', NULL, '2025-12-01 22:30:35', 5000000),
('a1bc9e63-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Minimalis Natural', 'Natural Beauty Studio', 'Riasan minimalis natural untuk pengantin', 'ongoing', NULL, '2025-12-01 22:30:35', 3500000),
('a1bc9ea7-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Korean Style', 'K-Beauty Makeup', 'Riasan pengantin gaya korea', 'ongoing', NULL, '2025-12-01 22:30:35', 4500000),
('a1bc9ee6-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Akad dan Resepsi Sunda', 'Bandung Bridal MUA', 'Paket lengkap MUA untuk akad dan resepsi adat sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:30:35', 8000000),
('a1bc9f39-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Hijab Syari', 'Hijab Beauty Studio', 'Riasan pengantin muslimah dengan hijab syari', 'ongoing', NULL, '2025-12-01 22:30:35', 4000000),
('a1bc9f7d-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Glam Party', 'Glam Studio Makeup', 'Riasan glamour untuk resepsi pesta', 'ongoing', NULL, '2025-12-01 22:30:35', 5500000),
('a1bc9fbf-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pre-Wedding Sunda', 'Sunda Pre-Wed MUA', 'Riasan untuk sesi foto pre-wedding tema sunda', 'ongoing', NULL, '2025-12-01 22:30:35', 2500000),
('a1bca003-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Bridesmaid Sunda', 'Sunda Party Makeup', 'Riasan untuk bridesmaid tema sunda (per orang)', 'ongoing', NULL, '2025-12-01 22:30:35', 500000),
('a1bca044-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Family Complete', 'Family Makeup Package', 'Riasan untuk keluarga pengantin (5 orang)', 'ongoing', NULL, '2025-12-01 22:30:35', 3000000),
('a1bca089-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Jawa Solo Basahan', 'Solo MUA Jogja', 'Riasan adat jawa solo basahan lengkap', 'ongoing', NULL, '2025-12-01 22:30:35', 5200000),
('a1bca0c9-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Lamaran Sunda', 'Sunda Engagement MUA', 'Riasan untuk acara lamaran adat sunda', 'ongoing', NULL, '2025-12-01 22:30:35', 2800000),
('a1bca107-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Trial Make Up', 'Professional Trial MUA', 'Sesi trial makeup sebelum hari H', 'ongoing', NULL, '2025-12-01 22:30:35', 1000000),
('a1bca149-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Luxury Premium', 'Luxury Bridal MUA', 'Paket MUA premium dengan produk luxury brand', 'ongoing', NULL, '2025-12-01 22:30:35', 12000000),
('a215c93e-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Sunda Bandung', 'Saung Angklung Udjo Bandung', 'Venue outdoor dan indoor untuk pernikahan adat sunda di Bandung, kapasitas 500 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 45000000),
('a2160651-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Hotel Tradisional Sunda', 'Hotel Savoy Homann Bandung', 'Ballroom dengan nuansa sunda modern, kapasitas 800 tamu di pusat kota Bandung', 'ongoing', NULL, '2025-12-01 22:23:26', 75000000),
('a2160776-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Outdoor Lembang Sunda', 'The Lodge Maribaya Bandung', 'Venue outdoor dengan pemandangan alam khas sunda, kapasitas 300 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 35000000),
('a21607f3-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Serbaguna Adat Jawa Jogja', 'Sasana Budaya Jogjakarta', 'Venue tradisional jawa di Jogja, kapasitas 600 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 40000000),
('a2160853-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Modern Jakarta Pusat', 'Grand Ballroom Plaza Indonesia', 'Ballroom mewah di Jakarta Pusat, kapasitas 1000 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 150000000),
('a21608b0-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Villa Pernikahan Bali Ubud', 'Villa Seminyak Bali', 'Villa eksklusif untuk intimate wedding di Bali, kapasitas 100 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 50000000),
('a2160902-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Pantai Wedding Venue Bali', 'Sunset Beach Bali Resort', 'Venue tepi pantai untuk pernikahan di Bali, kapasitas 200 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 80000000),
('a216094c-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Minang Padang', 'Rumah Gadang Padang', 'Venue adat minang di Padang, kapasitas 400 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 38000000),
('a2160995-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Hotel Ballroom Semarang', 'Grand Ballroom Semarang Plaza', 'Ballroom hotel di Semarang, kapasitas 700 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 60000000),
('a21609da-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Garden Bogor', 'Taman Bunga Bogor', 'Outdoor garden venue di Bogor, kapasitas 350 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 32000000),
('a2160a88-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Pendopo Adat Jawa Solo', 'Pendopo Mangkunegaran Solo', 'Venue pendopo tradisional jawa di Solo, kapasitas 500 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 55000000),
('a2160ad9-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Rooftop Venue Jakarta Selatan', 'Sky Garden Jakarta', 'Rooftop venue dengan city view di Jakarta Selatan, kapasitas 250 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 65000000),
('a2160b23-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Adat Betawi Jakarta', 'Setu Babakan Jakarta', 'Venue adat betawi di Jakarta, kapasitas 450 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 42000000),
('a2160b6e-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Villa Puncak Bogor', 'Villa Kharisma Puncak', 'Villa dengan view gunung di Puncak, kapasitas 200 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 38000000),
('a2160bba-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Hotel Surabaya', 'Grand Ballroom Surabaya Plaza', 'Ballroom mewah di Surabaya, kapasitas 900 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 85000000),
('a2160bfd-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Tepi Sungai Bandung', 'Riverside Garden Bandung', 'Venue outdoor tepi sungai di Bandung, kapasitas 300 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 35000000),
('a2160f86-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Sunda Cirebon', 'Keraton Kasepuhan Cirebon', 'Venue adat sunda-jawa di Cirebon, kapasitas 550 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 48000000),
('a2161014-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Hotel Resort Lombok', 'Beach Resort Lombok', 'Resort tepi pantai di Lombok, kapasitas 350 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 70000000),
('a2162bc6-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Perkebunan Teh Bandung', 'Kebun Teh Rancabali Bandung', 'Venue di perkebunan teh dengan suasana sunda, kapasitas 250 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 40000000),
('a2162c59-cec9-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Convention Tangerang', 'Convention Hall BSD Tangerang', 'Gedung modern di Tangerang, kapasitas 800 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 72000000),
('a2191311-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Premium 500 Pax', 'Catering Nasi Timbel Bandung', 'Menu khas sunda untuk 500 tamu di Bandung, nasi timbel, ayam bakar, ikan mas', 'ongoing', NULL, '2025-12-01 22:23:26', 42000000),
('a2192cd5-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Standard 300 Pax', 'Sari Sunda Catering Bandung', 'Menu sunda untuk 300 tamu, pepes ikan, ayam goreng, sayur asem', 'ongoing', NULL, '2025-12-01 22:23:26', 24000000),
('a2196dc2-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Eksklusif 800 Pax', 'Bumbu Desa Bandung', 'Menu sunda eksklusif untuk 800 tamu dengan live cooking', 'ongoing', NULL, '2025-12-01 22:23:26', 75000000),
('a2196edd-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Jawa 500 Pax', 'Omah Javanese Catering', 'Menu khas jawa untuk 500 tamu, gudeg, ayam bakar, tumpeng', 'ongoing', NULL, '2025-12-01 22:23:26', 38000000),
('a2196f4a-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Minang 400 Pax', 'Sederhana Catering Padang', 'Menu padang untuk 400 tamu, rendang, gulai, ayam pop', 'ongoing', NULL, '2025-12-01 22:23:26', 35000000),
('a2196fa4-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering International 600 Pax', 'Premier Catering Jakarta', 'Menu international buffet untuk 600 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 55000000),
('a2196ffc-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Betawi 350 Pax', 'Betawi Heritage Catering', 'Menu khas betawi untuk 350 tamu, kerak telor, soto betawi', 'ongoing', NULL, '2025-12-01 22:23:26', 30000000),
('a2197208-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Seafood 500 Pax', 'Ocean Fresh Catering', 'Menu seafood premium untuk 500 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 52000000),
('a219726b-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Vegetarian 200 Pax', 'Green Garden Catering', 'Menu vegetarian untuk 200 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 18000000),
('a21972b5-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Fusion 450 Pax', 'Fusion Taste Catering', 'Menu fusion indonesia-western untuk 450 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 42000000),
('a21972fe-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering BBQ 300 Pax', 'BBQ Nation Catering', 'BBQ buffet untuk 300 tamu dengan live grilling', 'ongoing', NULL, '2025-12-01 22:23:26', 35000000),
('a219736a-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Nusantara 700 Pax', 'Nusantara Kitchen Catering', 'Menu nusantara lengkap untuk 700 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 63000000),
('a21973ad-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Budget 200 Pax', 'Warung Nasi Sunda', 'Menu sunda ekonomis untuk 200 tamu di Bandung', 'ongoing', NULL, '2025-12-01 22:23:26', 14000000),
('a21976da-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Bali 400 Pax', 'Balinese Feast Catering', 'Menu khas bali untuk 400 tamu, babi guling, lawar', 'ongoing', NULL, '2025-12-01 22:23:26', 40000000),
('a219dc65-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Chinese 500 Pax', 'Golden Dragon Catering', 'Menu chinese untuk 500 tamu, dimsum, peking duck', 'ongoing', NULL, '2025-12-01 22:23:26', 48000000),
('a219dd5d-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Mediteranian 350 Pax', 'Mediterranean Delight', 'Menu mediterania untuk 350 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 45000000),
('a219ddb4-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Plus Dessert 600 Pax', 'Premium Sunda Catering Bandung', 'Menu sunda lengkap dengan dessert corner untuk 600 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 58000000),
('a219de02-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Japanese 300 Pax', 'Sakura Catering', 'Menu japanese untuk 300 tamu, sushi, ramen, teriyaki', 'ongoing', NULL, '2025-12-01 22:23:26', 42000000),
('a219de4d-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Korean 250 Pax', 'Seoul Kitchen Catering', 'Menu korean BBQ untuk 250 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 35000000),
('a219de90-cec9-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Lampung 400 Pax', 'Seruit Catering Lampung', 'Menu khas lampung untuk 400 tamu', 'ongoing', NULL, '2025-12-01 22:23:26', 32000000),
('a21c5707-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Sunda', 'Sundanese Wedding Decor Bandung', 'Dekorasi lengkap adat sunda dengan payung geulis, bale adat di Bandung', 'ongoing', NULL, '2025-12-01 22:23:26', 25000000),
('a21c76a5-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Sunda Premium', 'Parahyangan Decor Bandung', 'Dekorasi sunda premium dengan ornamen tradisional lengkap', 'ongoing', NULL, '2025-12-01 22:23:26', 35000000),
('a21c778c-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Jawa', 'Javanese Decor Jogja', 'Dekorasi pelaminan adat jawa lengkap dengan bleketepe', 'ongoing', NULL, '2025-12-01 22:23:26', 28000000),
('a21c77e5-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Minimalis Modern', 'Modern Decor Jakarta', 'Dekorasi minimalis modern dengan bunga segar', 'ongoing', NULL, '2025-12-01 22:23:26', 22000000),
('a21c783a-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Rustic Outdoor', 'Rustic Garden Decor', 'Dekorasi rustic untuk outdoor wedding dengan kayu dan burlap', 'ongoing', NULL, '2025-12-01 22:23:26', 30000000),
('a21c7889-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Garden Party', 'Green Paradise Decor', 'Dekorasi garden party dengan hanging flowers', 'ongoing', NULL, '2025-12-01 22:23:26', 28000000),
('a21c78c8-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Ballroom Mewah', 'Luxury Ballroom Decor', 'Dekorasi ballroom mewah dengan crystal chandelier', 'ongoing', NULL, '2025-12-01 22:23:26', 45000000),
('a21c7909-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Betawi', 'Betawi Decor Jakarta', 'Dekorasi pernikahan adat betawi dengan ondel-ondel', 'ongoing', NULL, '2025-12-01 22:23:26', 24000000),
('a21c7945-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Minang', 'Minang Decor Padang', 'Dekorasi adat minang dengan ornamen khas rumah gadang', 'ongoing', NULL, '2025-12-01 22:23:26', 26000000),
('a21c7985-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pantai Beach Wedding', 'Beach Wedding Decor Bali', 'Dekorasi pernikahan tepi pantai di Bali', 'ongoing', NULL, '2025-12-01 22:23:26', 32000000),
('a21c79c7-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Vintage Classic', 'Vintage Decor Studio', 'Dekorasi vintage dengan barang-barang antik', 'ongoing', NULL, '2025-12-01 22:23:26', 27000000),
('a21c7a05-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Bohemian', 'Boho Chic Decor', 'Dekorasi bohemian dengan macrame dan dreamcatcher', 'ongoing', NULL, '2025-12-01 22:23:26', 25000000),
('a21c7c16-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi International Elegant', 'Elite International Decor', 'Dekorasi international dengan bunga import', 'ongoing', NULL, '2025-12-01 22:23:26', 50000000),
('a21c7c80-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Sunda Minimalis Bandung', 'Sunda Modern Decor', 'Perpaduan dekorasi sunda dan minimalis modern di Bandung', 'ongoing', NULL, '2025-12-01 22:23:26', 30000000),
('a21c7cc3-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Bali', 'Balinese Decor Bali', 'Dekorasi pernikahan adat bali lengkap', 'ongoing', NULL, '2025-12-01 22:23:26', 38000000),
('a21c7e1e-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pelaminan Jawa Tengah', 'Javanese Central Decor', 'Dekorasi pelaminan jawa tengah dengan ukiran khas', 'ongoing', NULL, '2025-12-01 22:23:26', 29000000),
('a21c7e73-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Modern Glamour', 'Glamour Decor Jakarta', 'Dekorasi modern glamour dengan lighting dramatis', 'ongoing', NULL, '2025-12-01 22:23:26', 42000000),
('a21c7ee5-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Traditional Nusantara', 'Nusantara Heritage Decor', 'Dekorasi traditional nusantara campuran berbagai adat', 'ongoing', NULL, '2025-12-01 22:23:26', 33000000),
('a21c7f54-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Garden Sunda Bandung', 'Garden Sunda Decor', 'Dekorasi outdoor garden dengan tema sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:23:26', 28000000),
('a21c7f9d-cec9-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Romantic Pink', 'Romantic Pink Decor', 'Dekorasi romantis dengan dominan pink dan putih', 'ongoing', NULL, '2025-12-01 22:23:26', 26000000),
('a21f9b78-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pengantin Adat Sunda', 'Sunda Beauty Bandung', 'Riasan pengantin adat sunda dengan siger khas Bandung', 'ongoing', NULL, '2025-12-01 22:23:26', 4500000),
('a21fb950-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Sunda Siger Premium', 'Parahyangan Makeup Artist', 'Riasan sunda siger premium dengan aksesoris lengkap di Bandung', 'ongoing', NULL, '2025-12-01 22:23:26', 6000000),
('a21fd120-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Sunda Modern', 'Modern Sunda MUA Bandung', 'Riasan sunda modern untuk pengantin di Bandung', 'ongoing', NULL, '2025-12-01 22:23:26', 4000000),
('a21fd232-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Jawa Paes Ageng', 'Javanese MUA Jogja', 'Riasan pengantin jawa paes ageng lengkap', 'ongoing', NULL, '2025-12-01 22:23:26', 5500000),
('a21fd286-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA International Style', 'Elite Makeup Jakarta', 'Riasan pengantin gaya international modern', 'ongoing', NULL, '2025-12-01 22:23:26', 7000000),
('a21fd2ee-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Betawi', 'Betawi MUA Jakarta', 'Riasan pengantin adat betawi lengkap', 'ongoing', NULL, '2025-12-01 22:23:26', 4200000),
('a21fd364-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Minang', 'Minang Beauty Padang', 'Riasan pengantin adat minang dengan sunting', 'ongoing', NULL, '2025-12-01 22:23:26', 4800000),
('a21fd3c4-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Bali', 'Balinese Makeup Bali', 'Riasan pengantin adat bali lengkap', 'ongoing', NULL, '2025-12-01 22:23:26', 5000000),
('a21fd429-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Minimalis Natural', 'Natural Beauty Studio', 'Riasan minimalis natural untuk pengantin', 'ongoing', NULL, '2025-12-01 22:23:26', 3500000),
('a21fd48c-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Korean Style', 'K-Beauty Makeup', 'Riasan pengantin gaya korea', 'ongoing', NULL, '2025-12-01 22:23:26', 4500000),
('a21fd4ea-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Akad dan Resepsi Sunda', 'Bandung Bridal MUA', 'Paket lengkap MUA untuk akad dan resepsi adat sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:23:26', 8000000),
('a21fd55b-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Hijab Syari', 'Hijab Beauty Studio', 'Riasan pengantin muslimah dengan hijab syari', 'ongoing', NULL, '2025-12-01 22:23:26', 4000000),
('a21fd7a0-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Glam Party', 'Glam Studio Makeup', 'Riasan glamour untuk resepsi pesta', 'ongoing', NULL, '2025-12-01 22:23:26', 5500000),
('a21fd8c1-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pre-Wedding Sunda', 'Sunda Pre-Wed MUA', 'Riasan untuk sesi foto pre-wedding tema sunda', 'ongoing', NULL, '2025-12-01 22:23:26', 2500000),
('a21fd9fe-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Bridesmaid Sunda', 'Sunda Party Makeup', 'Riasan untuk bridesmaid tema sunda (per orang)', 'ongoing', NULL, '2025-12-01 22:23:26', 500000),
('a21fda92-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Family Complete', 'Family Makeup Package', 'Riasan untuk keluarga pengantin (5 orang)', 'ongoing', NULL, '2025-12-01 22:23:26', 3000000),
('a21fdafc-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Jawa Solo Basahan', 'Solo MUA Jogja', 'Riasan adat jawa solo basahan lengkap', 'ongoing', NULL, '2025-12-01 22:23:26', 5200000),
('a21fdb61-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Lamaran Sunda', 'Sunda Engagement MUA', 'Riasan untuk acara lamaran adat sunda', 'ongoing', NULL, '2025-12-01 22:23:26', 2800000),
('a21fdbc7-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Trial Make Up', 'Professional Trial MUA', 'Sesi trial makeup sebelum hari H', 'ongoing', NULL, '2025-12-01 22:23:26', 1000000),
('a21fdc2c-cec9-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Luxury Premium', 'Luxury Bridal MUA', 'Paket MUA premium dengan produk luxury brand', 'ongoing', NULL, '2025-12-01 22:23:26', 12000000),
('ac189d3a-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Pernikahan Adat Jawa', 'MC Bima Sakti', 'Jasa MC khusus untuk prosesi pernikahan adat Jawa lengkap (panggih, sungkem, dll).', 'ongoing', 'mc_adat_jawa.jpg', '2025-11-29 17:46:49', 4000000),
('ac18ba7e-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Resepsi Modern Formal (Bahasa Indonesia)', 'MC Rina Azhar', 'MC formal berbahasa Indonesia, elegan dan profesional, durasi 4 jam.', 'ongoing', 'mc_modern_formal.jpg', '2025-11-29 17:46:49', 5500000),
('ac18bb44-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Acara Akad Nikah (Semi-Formal)', 'MC Yoga Pratama', 'MC untuk acara sakral akad nikah, termasuk panduan rundown yang khidmat.', 'ongoing', NULL, '2025-11-29 17:46:49', 3000000),
('ac18bbb8-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Pernikahan Internasional (Bilingual Indo-Eng)', 'MC Rina Azhar', 'MC bilingual (Indonesia dan Inggris) untuk resepsi skala internasional.', 'ongoing', 'mc_bilingual.jpg', '2025-11-29 17:46:49', 7500000),
('ac18bc0a-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Launching Produk/Seminar Bisnis', 'MC Bima Sakti', 'MC profesional untuk acara korporat, launching, dan seminar.', 'ongoing', NULL, '2025-11-29 17:46:49', 4500000),
('ac18bc53-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Pesta Ulang Tahun Anak-anak (Fun & Games)', 'MC Cici Cerianya', 'MC spesialis anak, penuh energi, termasuk sesi permainan dan sulap ringan.', 'ongoing', NULL, '2025-11-29 17:46:49', 1500000),
('ac18bca7-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Lamaran/Tunangan (Intimate Ceremony)', 'MC Yoga Pratama', 'Jasa MC untuk acara lamaran yang intim dan hangat, durasi 3 jam.', 'ongoing', NULL, '2025-11-29 17:46:49', 2000000),
('ac18bcec-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Talkshow/Webinar Online', 'MC Rina Azhar', 'MC virtual untuk memandu talkshow atau webinar, full equipment studio.', 'airing', NULL, '2025-11-29 17:46:49', 3500000),
('ac18bd34-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Resepsi Pernikahan Adat Minang', 'MC Bima Sakti', 'MC khusus adat Minang, menguasai seluruh prosesi dan tata krama Minang.', 'ongoing', NULL, '2025-11-29 17:46:49', 4200000),
('ac18bd7b-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Pesta Malam Hari (Fun & Casual)', 'MC Yoga Pratama', 'MC dengan gaya santai dan humoris, cocok untuk pesta after-party.', 'ongoing', NULL, '2025-11-29 17:46:49', 3500000),
('ac18bdbf-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Upacara Tradisional (Satu Hari Penuh)', 'MC Bima Sakti', 'MC untuk acara adat full day, termasuk koordinasi dengan pihak keluarga.', 'ongoing', NULL, '2025-11-29 17:46:49', 6000000),
('ac18be04-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Hiburan Karaoke & Games', 'MC Cici Cerianya', 'MC untuk memandu sesi karaoke, games, dan doorprize di acara gathering.', 'ongoing', NULL, '2025-11-29 17:46:49', 2500000),
('ac18be49-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Pernikahan Intimate (Maks 100 Tamu)', 'MC Rina Azhar', 'MC fokus pada acara pernikahan kecil, lebih banyak interaksi dengan tamu.', 'ongoing', NULL, '2025-11-29 17:46:49', 4800000),
('ac18bec2-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Wisuda (Acara Kampus)', 'MC Yoga Pratama', 'MC formal untuk acara wisuda universitas, menguasai protokoler acara.', 'ongoing', NULL, '2025-11-29 17:46:49', 2800000),
('ac18bf20-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Duet (Pria & Wanita) Resepsi Modern', 'MC Duet Pro', 'Layanan dua MC (pria & wanita) untuk resepsi dengan konsep interaktif.', 'ongoing', 'mc_duet_pro.jpg', '2025-11-29 17:46:49', 8500000),
('ac18bf87-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Khitanan/Syukuran Anak', 'MC Cici Cerianya', 'MC yang ramah dan enerjik untuk acara syukuran dan khitanan anak.', 'ongoing', NULL, '2025-11-29 17:46:49', 1800000);
INSERT INTO `items` (`id`, `category_id`, `name`, `vendor`, `description`, `status`, `image_url`, `last_update`, `price`) VALUES
('ac18bff7-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Resepsi Outdoor/Garden Party', 'MC Yoga Pratama', 'MC berpengalaman di venue outdoor, menguasai tata panggung terbuka.', 'ongoing', 'mc_outdoor.jpg', '2025-11-29 17:46:49', 3800000),
('ac18c045-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Formal Kenegaraan/Pemerintahan', 'MC Bima Sakti', 'MC dengan kemampuan protokoler tinggi untuk acara instansi atau kenegaraan.', 'airing', NULL, '2025-11-29 17:46:49', 6500000),
('ac18c088-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Paket Hemat (Durasi Maks. 2 Jam)', 'MC Rina Azhar', 'Jasa MC untuk acara singkat (misalnya pembukaan/penutupan acara).', 'ongoing', NULL, '2025-11-29 17:46:49', 2500000),
('ac18c1c6-cd10-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Reuni/Malam Keakraban', 'MC Cici Cerianya', 'MC yang mampu menghidupkan suasana reuni dengan nostalgia dan interaksi.', 'ongoing', NULL, '2025-11-29 17:46:49', 2200000),
('c05a5a1d-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Sunda Bandung', 'Saung Angklung Udjo Bandung', 'Venue outdoor dan indoor untuk pernikahan adat sunda di Bandung, kapasitas 500 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 45000000),
('c05a7ab7-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Hotel Tradisional Sunda', 'Hotel Savoy Homann Bandung', 'Ballroom dengan nuansa sunda modern, kapasitas 800 tamu di pusat kota Bandung', 'ongoing', NULL, '2025-12-01 22:31:26', 75000000),
('c05a7bb2-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Outdoor Lembang Sunda', 'The Lodge Maribaya Bandung', 'Venue outdoor dengan pemandangan alam khas sunda, kapasitas 300 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 35000000),
('c05a7c5c-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Serbaguna Adat Jawa Jogja', 'Sasana Budaya Jogjakarta', 'Venue tradisional jawa di Jogja, kapasitas 600 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 40000000),
('c05a803c-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Modern Jakarta Pusat', 'Grand Ballroom Plaza Indonesia', 'Ballroom mewah di Jakarta Pusat, kapasitas 1000 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 150000000),
('c05a8183-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Villa Pernikahan Bali Ubud', 'Villa Seminyak Bali', 'Villa eksklusif untuk intimate wedding di Bali, kapasitas 100 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 50000000),
('c05a81f1-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Pantai Wedding Venue Bali', 'Sunset Beach Bali Resort', 'Venue tepi pantai untuk pernikahan di Bali, kapasitas 200 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 80000000),
('c05a8268-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Minang Padang', 'Rumah Gadang Padang', 'Venue adat minang di Padang, kapasitas 400 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 38000000),
('c05aa04c-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Hotel Ballroom Semarang', 'Grand Ballroom Semarang Plaza', 'Ballroom hotel di Semarang, kapasitas 700 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 60000000),
('c05aa0f6-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Garden Bogor', 'Taman Bunga Bogor', 'Outdoor garden venue di Bogor, kapasitas 350 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 32000000),
('c05aa15d-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Pendopo Adat Jawa Solo', 'Pendopo Mangkunegaran Solo', 'Venue pendopo tradisional jawa di Solo, kapasitas 500 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 55000000),
('c05aa1ad-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Rooftop Venue Jakarta Selatan', 'Sky Garden Jakarta', 'Rooftop venue dengan city view di Jakarta Selatan, kapasitas 250 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 65000000),
('c05aa203-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Adat Betawi Jakarta', 'Setu Babakan Jakarta', 'Venue adat betawi di Jakarta, kapasitas 450 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 42000000),
('c05aa256-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Villa Puncak Bogor', 'Villa Kharisma Puncak', 'Villa dengan view gunung di Puncak, kapasitas 200 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 38000000),
('c05aa2a4-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Hotel Surabaya', 'Grand Ballroom Surabaya Plaza', 'Ballroom mewah di Surabaya, kapasitas 900 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 85000000),
('c05aa2f2-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Tepi Sungai Bandung', 'Riverside Garden Bandung', 'Venue outdoor tepi sungai di Bandung, kapasitas 300 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 35000000),
('c05aa38b-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Sunda Cirebon', 'Keraton Kasepuhan Cirebon', 'Venue adat sunda-jawa di Cirebon, kapasitas 550 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 48000000),
('c05ab69f-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Hotel Resort Lombok', 'Beach Resort Lombok', 'Resort tepi pantai di Lombok, kapasitas 350 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 70000000),
('c05ab77f-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Perkebunan Teh Bandung', 'Kebun Teh Rancabali Bandung', 'Venue di perkebunan teh dengan suasana sunda, kapasitas 250 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 40000000),
('c05ab7eb-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Convention Tangerang', 'Convention Hall BSD Tangerang', 'Gedung modern di Tangerang, kapasitas 800 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 72000000),
('c05d6755-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Premium 500 Pax', 'Catering Nasi Timbel Bandung', 'Menu khas sunda untuk 500 tamu di Bandung, nasi timbel, ayam bakar, ikan mas', 'ongoing', NULL, '2025-12-01 22:31:26', 42000000),
('c05d72dc-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Standard 300 Pax', 'Sari Sunda Catering Bandung', 'Menu sunda untuk 300 tamu, pepes ikan, ayam goreng, sayur asem', 'ongoing', NULL, '2025-12-01 22:31:26', 24000000),
('c05d87fd-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Eksklusif 800 Pax', 'Bumbu Desa Bandung', 'Menu sunda eksklusif untuk 800 tamu dengan live cooking', 'ongoing', NULL, '2025-12-01 22:31:26', 75000000),
('c05d8965-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Jawa 500 Pax', 'Omah Javanese Catering', 'Menu khas jawa untuk 500 tamu, gudeg, ayam bakar, tumpeng', 'ongoing', NULL, '2025-12-01 22:31:26', 38000000),
('c05d8b54-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Minang 400 Pax', 'Sederhana Catering Padang', 'Menu padang untuk 400 tamu, rendang, gulai, ayam pop', 'ongoing', NULL, '2025-12-01 22:31:26', 35000000),
('c05d8bbe-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering International 600 Pax', 'Premier Catering Jakarta', 'Menu international buffet untuk 600 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 55000000),
('c05d8c0c-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Betawi 350 Pax', 'Betawi Heritage Catering', 'Menu khas betawi untuk 350 tamu, kerak telor, soto betawi', 'ongoing', NULL, '2025-12-01 22:31:26', 30000000),
('c05d8c52-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Seafood 500 Pax', 'Ocean Fresh Catering', 'Menu seafood premium untuk 500 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 52000000),
('c05d8c9a-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Vegetarian 200 Pax', 'Green Garden Catering', 'Menu vegetarian untuk 200 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 18000000),
('c05d8cdb-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Fusion 450 Pax', 'Fusion Taste Catering', 'Menu fusion indonesia-western untuk 450 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 42000000),
('c05d9f75-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering BBQ 300 Pax', 'BBQ Nation Catering', 'BBQ buffet untuk 300 tamu dengan live grilling', 'ongoing', NULL, '2025-12-01 22:31:26', 35000000),
('c05d9ff5-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Nusantara 700 Pax', 'Nusantara Kitchen Catering', 'Menu nusantara lengkap untuk 700 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 63000000),
('c05da03e-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Budget 200 Pax', 'Warung Nasi Sunda', 'Menu sunda ekonomis untuk 200 tamu di Bandung', 'ongoing', NULL, '2025-12-01 22:31:26', 14000000),
('c05da085-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Bali 400 Pax', 'Balinese Feast Catering', 'Menu khas bali untuk 400 tamu, babi guling, lawar', 'ongoing', NULL, '2025-12-01 22:31:26', 40000000),
('c05da0c6-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Chinese 500 Pax', 'Golden Dragon Catering', 'Menu chinese untuk 500 tamu, dimsum, peking duck', 'ongoing', NULL, '2025-12-01 22:31:26', 48000000),
('c05da10a-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Mediteranian 350 Pax', 'Mediterranean Delight', 'Menu mediterania untuk 350 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 45000000),
('c05da14e-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Plus Dessert 600 Pax', 'Premium Sunda Catering Bandung', 'Menu sunda lengkap dengan dessert corner untuk 600 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 58000000),
('c05da193-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Japanese 300 Pax', 'Sakura Catering', 'Menu japanese untuk 300 tamu, sushi, ramen, teriyaki', 'ongoing', NULL, '2025-12-01 22:31:26', 42000000),
('c05da1d6-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Korean 250 Pax', 'Seoul Kitchen Catering', 'Menu korean BBQ untuk 250 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 35000000),
('c05da219-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Lampung 400 Pax', 'Seruit Catering Lampung', 'Menu khas lampung untuk 400 tamu', 'ongoing', NULL, '2025-12-01 22:31:26', 32000000),
('c05feb7b-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Sunda', 'Sundanese Wedding Decor Bandung', 'Dekorasi lengkap adat sunda dengan payung geulis, bale adat di Bandung', 'ongoing', NULL, '2025-12-01 22:31:26', 25000000),
('c05ff5f9-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Sunda Premium', 'Parahyangan Decor Bandung', 'Dekorasi sunda premium dengan ornamen tradisional lengkap', 'ongoing', NULL, '2025-12-01 22:31:26', 35000000),
('c05ff6a7-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Jawa', 'Javanese Decor Jogja', 'Dekorasi pelaminan adat jawa lengkap dengan bleketepe', 'ongoing', NULL, '2025-12-01 22:31:26', 28000000),
('c05ff708-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Minimalis Modern', 'Modern Decor Jakarta', 'Dekorasi minimalis modern dengan bunga segar', 'ongoing', NULL, '2025-12-01 22:31:26', 22000000),
('c05ff75f-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Rustic Outdoor', 'Rustic Garden Decor', 'Dekorasi rustic untuk outdoor wedding dengan kayu dan burlap', 'ongoing', NULL, '2025-12-01 22:31:26', 30000000),
('c05ff7b3-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Garden Party', 'Green Paradise Decor', 'Dekorasi garden party dengan hanging flowers', 'ongoing', NULL, '2025-12-01 22:31:26', 28000000),
('c05ff7f4-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Ballroom Mewah', 'Luxury Ballroom Decor', 'Dekorasi ballroom mewah dengan crystal chandelier', 'ongoing', NULL, '2025-12-01 22:31:26', 45000000),
('c05ff83f-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Betawi', 'Betawi Decor Jakarta', 'Dekorasi pernikahan adat betawi dengan ondel-ondel', 'ongoing', NULL, '2025-12-01 22:31:26', 24000000),
('c05ff8bb-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Minang', 'Minang Decor Padang', 'Dekorasi adat minang dengan ornamen khas rumah gadang', 'ongoing', NULL, '2025-12-01 22:31:26', 26000000),
('c05ff90d-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pantai Beach Wedding', 'Beach Wedding Decor Bali', 'Dekorasi pernikahan tepi pantai di Bali', 'ongoing', NULL, '2025-12-01 22:31:26', 32000000),
('c05ff96b-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Vintage Classic', 'Vintage Decor Studio', 'Dekorasi vintage dengan barang-barang antik', 'ongoing', NULL, '2025-12-01 22:31:26', 27000000),
('c05ff9d7-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Bohemian', 'Boho Chic Decor', 'Dekorasi bohemian dengan macrame dan dreamcatcher', 'ongoing', NULL, '2025-12-01 22:31:26', 25000000),
('c05ffa3f-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi International Elegant', 'Elite International Decor', 'Dekorasi international dengan bunga import', 'ongoing', NULL, '2025-12-01 22:31:26', 50000000),
('c05ffa99-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Sunda Minimalis Bandung', 'Sunda Modern Decor', 'Perpaduan dekorasi sunda dan minimalis modern di Bandung', 'ongoing', NULL, '2025-12-01 22:31:26', 30000000),
('c05ffaf3-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Bali', 'Balinese Decor Bali', 'Dekorasi pernikahan adat bali lengkap', 'ongoing', NULL, '2025-12-01 22:31:26', 38000000),
('c05ffb56-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pelaminan Jawa Tengah', 'Javanese Central Decor', 'Dekorasi pelaminan jawa tengah dengan ukiran khas', 'ongoing', NULL, '2025-12-01 22:31:26', 29000000),
('c05ffbc9-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Modern Glamour', 'Glamour Decor Jakarta', 'Dekorasi modern glamour dengan lighting dramatis', 'ongoing', NULL, '2025-12-01 22:31:26', 42000000),
('c05ffc2b-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Traditional Nusantara', 'Nusantara Heritage Decor', 'Dekorasi traditional nusantara campuran berbagai adat', 'ongoing', NULL, '2025-12-01 22:31:26', 33000000),
('c05ffc88-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Garden Sunda Bandung', 'Garden Sunda Decor', 'Dekorasi outdoor garden dengan tema sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:31:26', 28000000),
('c05ffce6-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Romantic Pink', 'Romantic Pink Decor', 'Dekorasi romantis dengan dominan pink dan putih', 'ongoing', NULL, '2025-12-01 22:31:26', 26000000),
('c0625b19-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pengantin Adat Sunda', 'Sunda Beauty Bandung', 'Riasan pengantin adat sunda dengan siger khas Bandung', 'ongoing', NULL, '2025-12-01 22:31:26', 4500000),
('c062659a-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Sunda Siger Premium', 'Parahyangan Makeup Artist', 'Riasan sunda siger premium dengan aksesoris lengkap di Bandung', 'ongoing', NULL, '2025-12-01 22:31:26', 6000000),
('c0626648-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Sunda Modern', 'Modern Sunda MUA Bandung', 'Riasan sunda modern untuk pengantin di Bandung', 'ongoing', NULL, '2025-12-01 22:31:26', 4000000),
('c06266a0-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Jawa Paes Ageng', 'Javanese MUA Jogja', 'Riasan pengantin jawa paes ageng lengkap', 'ongoing', NULL, '2025-12-01 22:31:26', 5500000),
('c06266f2-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA International Style', 'Elite Makeup Jakarta', 'Riasan pengantin gaya international modern', 'ongoing', NULL, '2025-12-01 22:31:26', 7000000),
('c062673f-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Betawi', 'Betawi MUA Jakarta', 'Riasan pengantin adat betawi lengkap', 'ongoing', NULL, '2025-12-01 22:31:26', 4200000),
('c0626783-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Minang', 'Minang Beauty Padang', 'Riasan pengantin adat minang dengan sunting', 'ongoing', NULL, '2025-12-01 22:31:26', 4800000),
('c06267c6-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Bali', 'Balinese Makeup Bali', 'Riasan pengantin adat bali lengkap', 'ongoing', NULL, '2025-12-01 22:31:26', 5000000),
('c062680a-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Minimalis Natural', 'Natural Beauty Studio', 'Riasan minimalis natural untuk pengantin', 'ongoing', NULL, '2025-12-01 22:31:26', 3500000),
('c0626854-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Korean Style', 'K-Beauty Makeup', 'Riasan pengantin gaya korea', 'ongoing', NULL, '2025-12-01 22:31:26', 4500000),
('c0626893-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Akad dan Resepsi Sunda', 'Bandung Bridal MUA', 'Paket lengkap MUA untuk akad dan resepsi adat sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:31:26', 8000000),
('c06268d6-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Hijab Syari', 'Hijab Beauty Studio', 'Riasan pengantin muslimah dengan hijab syari', 'ongoing', NULL, '2025-12-01 22:31:26', 4000000),
('c0626917-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Glam Party', 'Glam Studio Makeup', 'Riasan glamour untuk resepsi pesta', 'ongoing', NULL, '2025-12-01 22:31:26', 5500000),
('c0626957-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pre-Wedding Sunda', 'Sunda Pre-Wed MUA', 'Riasan untuk sesi foto pre-wedding tema sunda', 'ongoing', NULL, '2025-12-01 22:31:26', 2500000),
('c062699a-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Bridesmaid Sunda', 'Sunda Party Makeup', 'Riasan untuk bridesmaid tema sunda (per orang)', 'ongoing', NULL, '2025-12-01 22:31:26', 500000),
('c06269dd-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Family Complete', 'Family Makeup Package', 'Riasan untuk keluarga pengantin (5 orang)', 'ongoing', NULL, '2025-12-01 22:31:26', 3000000),
('c0626a1b-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Jawa Solo Basahan', 'Solo MUA Jogja', 'Riasan adat jawa solo basahan lengkap', 'ongoing', NULL, '2025-12-01 22:31:26', 5200000),
('c0626a5a-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Lamaran Sunda', 'Sunda Engagement MUA', 'Riasan untuk acara lamaran adat sunda', 'ongoing', NULL, '2025-12-01 22:31:26', 2800000),
('c0626a97-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Trial Make Up', 'Professional Trial MUA', 'Sesi trial makeup sebelum hari H', 'ongoing', NULL, '2025-12-01 22:31:26', 1000000),
('c0626ad8-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Luxury Premium', 'Luxury Bridal MUA', 'Paket MUA premium dengan produk luxury brand', 'ongoing', NULL, '2025-12-01 22:31:26', 12000000),
('d75d410a-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Paket Akustik Band (4 Personil)', 'Acoustic Soul', 'Layanan musik akustik (gitar, bass, cajon, vokal), 3 set x 45 menit, meliputi lagu pop & jazz.', 'ongoing', 'ent_akustik_band.jpg', '2025-11-29 17:48:02', 4500000),
('d75d5c8a-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Full Band Wedding (7 Personil + Keyboard)', 'The Wedding Sounds', 'Band lengkap untuk resepsi (drums, gitar, bass, keyboard, 3 vokal), sound system standar.', 'ongoing', 'ent_full_band.jpg', '2025-11-29 17:48:02', 12000000),
('d75d5d1a-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Sewa Keyboard Tunggal (Pemain Profesional)', 'Harmony Keys', 'Pemain keyboard profesional untuk iringan akad nikah atau musik latar, durasi 3 jam.', 'ongoing', NULL, '2025-11-29 17:48:02', 2000000),
('d75d5d71-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Penari Tradisional (Tari Saman/Sunda)', 'Sanggar Tari Nusantara', 'Jasa penari adat untuk menyambut tamu atau mengisi jeda acara (2-3 kali penampilan).', 'ongoing', 'ent_tari_tradisional.jpg', '2025-11-29 17:48:02', 3500000),
('d75d5db9-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Penyanyi Solo Wanita (Pop & R&B)', 'Vocalist Diva', 'Penyanyi solo profesional untuk membawakan 10-15 lagu pilihan.', 'ongoing', NULL, '2025-11-29 17:48:02', 1800000),
('d75d5e0c-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Sewa Sound System Standar (Maks. 5000 Watt)', 'Decibel Audio', 'Penyewaan sound system standar untuk acara di gedung/outdoor, termasuk operator.', 'ongoing', NULL, '2025-11-29 17:48:02', 3000000),
('d75d5e4e-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Pertunjukan Sulap (Close-Up Magic)', 'Magician Reno', 'Pertunjukan sulap interaktif di antara meja tamu (close-up magic), durasi 1 jam.', 'ongoing', NULL, '2025-11-29 17:48:02', 2500000),
('d75d5e8b-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'DJ Performance (All Genre)', 'DJ Nova Beat', 'Jasa DJ profesional untuk after party atau acara ulang tahun, durasi 4 jam.', 'ongoing', 'ent_dj_performance.jpg', '2025-11-29 17:48:02', 6000000),
('d75d5ed0-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Quartet String (Klasik & Kontemporer)', 'Musica Classica', 'Kuarter gesek (biola, cello, viola) untuk musik instrumental yang elegan.', 'ongoing', 'ent_string_quartet.jpg', '2025-11-29 17:48:02', 7000000),
('d75d5f4f-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Live Music Jazz Trio (Vokal, Piano, Bass)', 'Acoustic Soul', 'Trio jazz untuk suasana intim dan mewah, membawakan lagu-lagu standar jazz.', 'ongoing', NULL, '2025-11-29 17:48:02', 5500000),
('d75d5f92-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Sewa Lighting Panggung & Efek', 'Decibel Audio', 'Penyewaan set lighting panggung standar (moving head, par LED, smoke machine).', 'ongoing', NULL, '2025-11-29 17:48:02', 4000000),
('d75d5fd5-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Pertunjukan Barongsai (2 Singa)', 'Lion Dance Community', 'Pertunjukan Barongsai untuk pembukaan acara atau Imlek, 2 singa dan tim musik.', 'ongoing', 'ent_barongsai.jpg', '2025-11-29 17:48:02', 8000000),
('d75d6017-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'MC & Penyanyi All-in One', 'Vocalist Diva', 'Layanan MC dan penyanyi solo oleh satu orang (menghemat biaya dan koordinasi).', 'ongoing', NULL, '2025-11-29 17:48:02', 3500000),
('d75d6055-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Sewa Karaoke Set (Portable)', 'Harmony Keys', 'Penyewaan set karaoke portable untuk acara di rumah atau kantor.', 'ongoing', NULL, '2025-11-29 17:48:02', 1500000),
('d75d6090-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Paket Musik Tradisional Gamelan Jawa', 'Sanggar Tari Nusantara', 'Iringan musik gamelan lengkap dengan penabuh untuk upacara adat atau resepsi.', 'ongoing', 'ent_gamelan_jawa.jpg', '2025-11-29 17:48:02', 9500000),
('d75d61b3-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Penari Latar (Dancer) untuk Panggung', 'The Wedding Sounds', 'Jasa penari latar (4 orang) untuk mengiringi penyanyi utama atau band.', 'ongoing', NULL, '2025-11-29 17:48:02', 4800000),
('d75d9ff2-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Jasa VJ (Video Jockey) Panggung', 'DJ Nova Beat', 'Jasa operator visual/video untuk ditampilkan di layar LED selama acara berlangsung.', 'ongoing', NULL, '2025-11-29 17:48:02', 2800000),
('d75da0e2-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Sewa Mesin Fotokopi/Scanner (Event Office)', 'Decibel Audio', 'Penyewaan alat kantor untuk kebutuhan registrasi atau administrasi acara.', 'airing', NULL, '2025-11-29 17:48:02', 1000000),
('d75da15f-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Paduan Suara (Choir) Gereja/Akad', 'Musica Classica', 'Paduan suara mini (8 orang) untuk mengiringi pemberkatan atau akad nikah.', 'ongoing', NULL, '2025-11-29 17:48:02', 5000000),
('d75da1d9-cd10-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Sewa Panggung Portable (Ukuran 4x6m)', 'The Wedding Sounds', 'Penyewaan dan instalasi panggung portable modular, termasuk skirting.', 'ongoing', NULL, '2025-11-29 17:48:02', 3200000),
('dfb2d6ee-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Sunda Bandung', 'Saung Angklung Udjo Bandung', 'Venue outdoor dan indoor untuk pernikahan adat sunda di Bandung, kapasitas 500 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 45000000),
('dfb2e227-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Hotel Tradisional Sunda', 'Hotel Savoy Homann Bandung', 'Ballroom dengan nuansa sunda modern, kapasitas 800 tamu di pusat kota Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 75000000),
('dfb2e2d2-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Outdoor Lembang Sunda', 'The Lodge Maribaya Bandung', 'Venue outdoor dengan pemandangan alam khas sunda, kapasitas 300 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 35000000),
('dfb2e33e-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Serbaguna Adat Jawa Jogja', 'Sasana Budaya Jogjakarta', 'Venue tradisional jawa di Jogja, kapasitas 600 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 40000000),
('dfb2e393-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Modern Jakarta Pusat', 'Grand Ballroom Plaza Indonesia', 'Ballroom mewah di Jakarta Pusat, kapasitas 1000 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 150000000),
('dfb2e3e7-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Villa Pernikahan Bali Ubud', 'Villa Seminyak Bali', 'Villa eksklusif untuk intimate wedding di Bali, kapasitas 100 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 50000000),
('dfb2e435-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Pantai Wedding Venue Bali', 'Sunset Beach Bali Resort', 'Venue tepi pantai untuk pernikahan di Bali, kapasitas 200 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 80000000),
('dfb2e481-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Minang Padang', 'Rumah Gadang Padang', 'Venue adat minang di Padang, kapasitas 400 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 38000000),
('dfb2e4c8-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Hotel Ballroom Semarang', 'Grand Ballroom Semarang Plaza', 'Ballroom hotel di Semarang, kapasitas 700 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 60000000),
('dfb2e514-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Garden Bogor', 'Taman Bunga Bogor', 'Outdoor garden venue di Bogor, kapasitas 350 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 32000000),
('dfb2e55c-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Pendopo Adat Jawa Solo', 'Pendopo Mangkunegaran Solo', 'Venue pendopo tradisional jawa di Solo, kapasitas 500 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 55000000),
('dfb2e5a5-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Rooftop Venue Jakarta Selatan', 'Sky Garden Jakarta', 'Rooftop venue dengan city view di Jakarta Selatan, kapasitas 250 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 65000000),
('dfb2e5f6-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Adat Betawi Jakarta', 'Setu Babakan Jakarta', 'Venue adat betawi di Jakarta, kapasitas 450 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 42000000),
('dfb2e63e-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Villa Puncak Bogor', 'Villa Kharisma Puncak', 'Villa dengan view gunung di Puncak, kapasitas 200 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 38000000),
('dfb2e687-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Ballroom Hotel Surabaya', 'Grand Ballroom Surabaya Plaza', 'Ballroom mewah di Surabaya, kapasitas 900 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 85000000),
('dfb2e6cf-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Tepi Sungai Bandung', 'Riverside Garden Bandung', 'Venue outdoor tepi sungai di Bandung, kapasitas 300 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 35000000),
('dfb2e715-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Pernikahan Adat Sunda Cirebon', 'Keraton Kasepuhan Cirebon', 'Venue adat sunda-jawa di Cirebon, kapasitas 550 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 48000000),
('dfb2e76a-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Hotel Resort Lombok', 'Beach Resort Lombok', 'Resort tepi pantai di Lombok, kapasitas 350 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 70000000),
('dfb2e7b2-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Venue Perkebunan Teh Bandung', 'Kebun Teh Rancabali Bandung', 'Venue di perkebunan teh dengan suasana sunda, kapasitas 250 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 40000000),
('dfb2e7ff-ceca-11f0-89f6-a036bc697a25', '3c476203-a266-42c4-b2f5-3cdd7efcf510', 'Gedung Convention Tangerang', 'Convention Hall BSD Tangerang', 'Gedung modern di Tangerang, kapasitas 800 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 72000000),
('dfb6fd4e-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Premium 500 Pax', 'Catering Nasi Timbel Bandung', 'Menu khas sunda untuk 500 tamu di Bandung, nasi timbel, ayam bakar, ikan mas', 'ongoing', NULL, '2025-12-01 22:32:19', 42000000),
('dfb70a17-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Standard 300 Pax', 'Sari Sunda Catering Bandung', 'Menu sunda untuk 300 tamu, pepes ikan, ayam goreng, sayur asem', 'ongoing', NULL, '2025-12-01 22:32:19', 24000000),
('dfb70ae1-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Eksklusif 800 Pax', 'Bumbu Desa Bandung', 'Menu sunda eksklusif untuk 800 tamu dengan live cooking', 'ongoing', NULL, '2025-12-01 22:32:19', 75000000),
('dfb70b5d-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Jawa 500 Pax', 'Omah Javanese Catering', 'Menu khas jawa untuk 500 tamu, gudeg, ayam bakar, tumpeng', 'ongoing', NULL, '2025-12-01 22:32:19', 38000000),
('dfb70bb8-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Minang 400 Pax', 'Sederhana Catering Padang', 'Menu padang untuk 400 tamu, rendang, gulai, ayam pop', 'ongoing', NULL, '2025-12-01 22:32:19', 35000000),
('dfb70c0e-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering International 600 Pax', 'Premier Catering Jakarta', 'Menu international buffet untuk 600 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 55000000),
('dfb70c5a-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Betawi 350 Pax', 'Betawi Heritage Catering', 'Menu khas betawi untuk 350 tamu, kerak telor, soto betawi', 'ongoing', NULL, '2025-12-01 22:32:19', 30000000),
('dfb70ca7-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Seafood 500 Pax', 'Ocean Fresh Catering', 'Menu seafood premium untuk 500 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 52000000),
('dfb70cf1-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Vegetarian 200 Pax', 'Green Garden Catering', 'Menu vegetarian untuk 200 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 18000000),
('dfb70d76-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Fusion 450 Pax', 'Fusion Taste Catering', 'Menu fusion indonesia-western untuk 450 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 42000000),
('dfb70dc7-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering BBQ 300 Pax', 'BBQ Nation Catering', 'BBQ buffet untuk 300 tamu dengan live grilling', 'ongoing', NULL, '2025-12-01 22:32:19', 35000000),
('dfb70e0e-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Nusantara 700 Pax', 'Nusantara Kitchen Catering', 'Menu nusantara lengkap untuk 700 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 63000000),
('dfb70e56-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Budget 200 Pax', 'Warung Nasi Sunda', 'Menu sunda ekonomis untuk 200 tamu di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 14000000),
('dfb70ea2-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Bali 400 Pax', 'Balinese Feast Catering', 'Menu khas bali untuk 400 tamu, babi guling, lawar', 'ongoing', NULL, '2025-12-01 22:32:19', 40000000),
('dfb70f11-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Chinese 500 Pax', 'Golden Dragon Catering', 'Menu chinese untuk 500 tamu, dimsum, peking duck', 'ongoing', NULL, '2025-12-01 22:32:19', 48000000),
('dfb70f74-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Mediteranian 350 Pax', 'Mediterranean Delight', 'Menu mediterania untuk 350 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 45000000),
('dfb70fe9-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Sunda Plus Dessert 600 Pax', 'Premium Sunda Catering Bandung', 'Menu sunda lengkap dengan dessert corner untuk 600 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 58000000),
('dfb7105f-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Japanese 300 Pax', 'Sakura Catering', 'Menu japanese untuk 300 tamu, sushi, ramen, teriyaki', 'ongoing', NULL, '2025-12-01 22:32:19', 42000000),
('dfb710c5-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Korean 250 Pax', 'Seoul Kitchen Catering', 'Menu korean BBQ untuk 250 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 35000000),
('dfb71159-ceca-11f0-89f6-a036bc697a25', '33126b96-7f71-4141-b9d8-999ef4fd0cb8', 'Paket Catering Lampung 400 Pax', 'Seruit Catering Lampung', 'Menu khas lampung untuk 400 tamu', 'ongoing', NULL, '2025-12-01 22:32:19', 32000000),
('dfb9a625-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Sunda', 'Sundanese Wedding Decor Bandung', 'Dekorasi lengkap adat sunda dengan payung geulis, bale adat di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 25000000),
('dfb9b235-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Sunda Premium', 'Parahyangan Decor Bandung', 'Dekorasi sunda premium dengan ornamen tradisional lengkap', 'ongoing', NULL, '2025-12-01 22:32:19', 35000000),
('dfb9b2e4-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pernikahan Adat Jawa', 'Javanese Decor Jogja', 'Dekorasi pelaminan adat jawa lengkap dengan bleketepe', 'ongoing', NULL, '2025-12-01 22:32:19', 28000000),
('dfb9b33f-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Minimalis Modern', 'Modern Decor Jakarta', 'Dekorasi minimalis modern dengan bunga segar', 'ongoing', NULL, '2025-12-01 22:32:19', 22000000),
('dfb9b394-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Rustic Outdoor', 'Rustic Garden Decor', 'Dekorasi rustic untuk outdoor wedding dengan kayu dan burlap', 'ongoing', NULL, '2025-12-01 22:32:19', 30000000),
('dfb9b3e5-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Garden Party', 'Green Paradise Decor', 'Dekorasi garden party dengan hanging flowers', 'ongoing', NULL, '2025-12-01 22:32:19', 28000000),
('dfb9b42b-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Ballroom Mewah', 'Luxury Ballroom Decor', 'Dekorasi ballroom mewah dengan crystal chandelier', 'ongoing', NULL, '2025-12-01 22:32:19', 45000000),
('dfb9b46e-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Betawi', 'Betawi Decor Jakarta', 'Dekorasi pernikahan adat betawi dengan ondel-ondel', 'ongoing', NULL, '2025-12-01 22:32:19', 24000000),
('dfb9b4b4-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Minang', 'Minang Decor Padang', 'Dekorasi adat minang dengan ornamen khas rumah gadang', 'ongoing', NULL, '2025-12-01 22:32:19', 26000000),
('dfb9c89a-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pantai Beach Wedding', 'Beach Wedding Decor Bali', 'Dekorasi pernikahan tepi pantai di Bali', 'ongoing', NULL, '2025-12-01 22:32:19', 32000000),
('dfb9c919-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Vintage Classic', 'Vintage Decor Studio', 'Dekorasi vintage dengan barang-barang antik', 'ongoing', NULL, '2025-12-01 22:32:19', 27000000),
('dfb9c960-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Bohemian', 'Boho Chic Decor', 'Dekorasi bohemian dengan macrame dan dreamcatcher', 'ongoing', NULL, '2025-12-01 22:32:19', 25000000),
('dfb9c9a5-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi International Elegant', 'Elite International Decor', 'Dekorasi international dengan bunga import', 'ongoing', NULL, '2025-12-01 22:32:19', 50000000),
('dfb9c9e6-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Sunda Minimalis Bandung', 'Sunda Modern Decor', 'Perpaduan dekorasi sunda dan minimalis modern di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 30000000),
('dfb9ca28-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Adat Bali', 'Balinese Decor Bali', 'Dekorasi pernikahan adat bali lengkap', 'ongoing', NULL, '2025-12-01 22:32:19', 38000000),
('dfb9cb36-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Pelaminan Jawa Tengah', 'Javanese Central Decor', 'Dekorasi pelaminan jawa tengah dengan ukiran khas', 'ongoing', NULL, '2025-12-01 22:32:19', 29000000),
('dfb9cbe0-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Modern Glamour', 'Glamour Decor Jakarta', 'Dekorasi modern glamour dengan lighting dramatis', 'ongoing', NULL, '2025-12-01 22:32:19', 42000000),
('dfb9cc3d-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Traditional Nusantara', 'Nusantara Heritage Decor', 'Dekorasi traditional nusantara campuran berbagai adat', 'ongoing', NULL, '2025-12-01 22:32:19', 33000000),
('dfb9cc91-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Garden Sunda Bandung', 'Garden Sunda Decor', 'Dekorasi outdoor garden dengan tema sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 28000000),
('dfb9cce8-ceca-11f0-89f6-a036bc697a25', 'b01d482d-7301-44d4-80c1-155f1866a604', 'Dekorasi Romantic Pink', 'Romantic Pink Decor', 'Dekorasi romantis dengan dominan pink dan putih', 'ongoing', NULL, '2025-12-01 22:32:19', 26000000),
('dfbcb61f-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pengantin Adat Sunda', 'Sunda Beauty Bandung', 'Riasan pengantin adat sunda dengan siger khas Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 4500000),
('dfbcc0f6-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Sunda Siger Premium', 'Parahyangan Makeup Artist', 'Riasan sunda siger premium dengan aksesoris lengkap di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 6000000),
('dfbcc1e7-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Sunda Modern', 'Modern Sunda MUA Bandung', 'Riasan sunda modern untuk pengantin di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 4000000),
('dfbcc26d-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Jawa Paes Ageng', 'Javanese MUA Jogja', 'Riasan pengantin jawa paes ageng lengkap', 'ongoing', NULL, '2025-12-01 22:32:19', 5500000),
('dfbcc2ee-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA International Style', 'Elite Makeup Jakarta', 'Riasan pengantin gaya international modern', 'ongoing', NULL, '2025-12-01 22:32:19', 7000000),
('dfbcc360-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Betawi', 'Betawi MUA Jakarta', 'Riasan pengantin adat betawi lengkap', 'ongoing', NULL, '2025-12-01 22:32:19', 4200000),
('dfbcc3ca-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Minang', 'Minang Beauty Padang', 'Riasan pengantin adat minang dengan sunting', 'ongoing', NULL, '2025-12-01 22:32:19', 4800000),
('dfbcc431-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Bali', 'Balinese Makeup Bali', 'Riasan pengantin adat bali lengkap', 'ongoing', NULL, '2025-12-01 22:32:19', 5000000),
('dfbcc49b-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Minimalis Natural', 'Natural Beauty Studio', 'Riasan minimalis natural untuk pengantin', 'ongoing', NULL, '2025-12-01 22:32:19', 3500000),
('dfbcc503-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Korean Style', 'K-Beauty Makeup', 'Riasan pengantin gaya korea', 'ongoing', NULL, '2025-12-01 22:32:19', 4500000),
('dfbcc574-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Akad dan Resepsi Sunda', 'Bandung Bridal MUA', 'Paket lengkap MUA untuk akad dan resepsi adat sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 8000000),
('dfbcc5e2-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Hijab Syari', 'Hijab Beauty Studio', 'Riasan pengantin muslimah dengan hijab syari', 'ongoing', NULL, '2025-12-01 22:32:19', 4000000),
('dfbcc643-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Glam Party', 'Glam Studio Makeup', 'Riasan glamour untuk resepsi pesta', 'ongoing', NULL, '2025-12-01 22:32:19', 5500000),
('dfbcc6a8-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Pre-Wedding Sunda', 'Sunda Pre-Wed MUA', 'Riasan untuk sesi foto pre-wedding tema sunda', 'ongoing', NULL, '2025-12-01 22:32:19', 2500000),
('dfbcc70d-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Bridesmaid Sunda', 'Sunda Party Makeup', 'Riasan untuk bridesmaid tema sunda (per orang)', 'ongoing', NULL, '2025-12-01 22:32:19', 500000),
('dfbcc776-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Family Complete', 'Family Makeup Package', 'Riasan untuk keluarga pengantin (5 orang)', 'ongoing', NULL, '2025-12-01 22:32:19', 3000000),
('dfbcc7dd-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Adat Jawa Solo Basahan', 'Solo MUA Jogja', 'Riasan adat jawa solo basahan lengkap', 'ongoing', NULL, '2025-12-01 22:32:19', 5200000),
('dfbcc854-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Lamaran Sunda', 'Sunda Engagement MUA', 'Riasan untuk acara lamaran adat sunda', 'ongoing', NULL, '2025-12-01 22:32:19', 2800000),
('dfbcc8b6-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Trial Make Up', 'Professional Trial MUA', 'Sesi trial makeup sebelum hari H', 'ongoing', NULL, '2025-12-01 22:32:19', 1000000),
('dfbcc916-ceca-11f0-89f6-a036bc697a25', '1b2f2a98-7c81-47fb-b336-c939733ada78', 'Paket MUA Luxury Premium', 'Luxury Bridal MUA', 'Paket MUA premium dengan produk luxury brand', 'ongoing', NULL, '2025-12-01 22:32:19', 12000000),
('dfbf6f4b-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Video Pernikahan Sunda', 'Bandung Wedding Photography', 'Dokumentasi lengkap foto dan video pernikahan adat sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 12000000),
('dfbf905b-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Pernikahan Sunda Premium', 'Sunda Photo Studio', 'Paket foto premium untuk pernikahan adat sunda', 'ongoing', NULL, '2025-12-01 22:32:19', 8000000),
('dfbf9179-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Video Cinematic Sunda', 'Cinematic Wedding Bandung', 'Video cinematic untuk pernikahan sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 9000000),
('dfbf91de-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Video Traditional', 'Heritage Documentation', 'Dokumentasi pernikahan traditional lengkap', 'ongoing', NULL, '2025-12-01 22:32:19', 10000000),
('dfbf923c-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Pre-Wedding Outdoor Bandung', 'Bandung Pre-Wed Photo', 'Sesi foto pre-wedding outdoor di lokasi Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 4500000),
('dfbf9293-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Video Akad Nikah', 'Akad Documentation Team', 'Dokumentasi khusus acara akad nikah', 'ongoing', NULL, '2025-12-01 22:32:19', 5000000),
('dfbf92dc-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Video Highlight 5 Menit', 'Quick Highlight Video', 'Video highlight pernikahan durasi 5 menit', 'ongoing', NULL, '2025-12-01 22:32:19', 6000000),
('dfbf9324-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Drone Video Pernikahan', 'Sky View Documentation', 'Dokumentasi video dengan drone untuk aerial view', 'ongoing', NULL, '2025-12-01 22:32:19', 3500000),
('dfbf936a-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Album Kolase Premium', 'Premium Album Maker', 'Album foto kolase premium 30 halaman', 'ongoing', NULL, '2025-12-01 22:32:19', 2500000),
('dfbf93ac-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Live Streaming Pernikahan', 'Live Stream Wedding', 'Layanan live streaming untuk pernikahan (2 kamera)', 'ongoing', NULL, '2025-12-01 22:32:19', 4000000),
('dfbf93f7-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Keluarga Besar', 'Family Photo Session', 'Sesi foto keluarga besar saat pernikahan', 'ongoing', NULL, '2025-12-01 22:32:19', 2000000),
('dfbf943e-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Video After Movie', 'After Movie Production', 'Video after movie pernikahan durasi 10 menit', 'ongoing', NULL, '2025-12-01 22:32:19', 8500000),
('dfbf948c-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Pre-Wedding Indoor Studio', 'Studio Pre-Wedding Photo', 'Sesi foto pre-wedding di studio profesional', 'ongoing', NULL, '2025-12-01 22:32:19', 3000000),
('dfbf94d6-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Video Cinematic International', 'International Cinematic', 'Video cinematic gaya international', 'ongoing', NULL, '2025-12-01 22:32:19', 15000000),
('dfbf9521-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Dokumentasi Lamaran', 'Engagement Documentation', 'Dokumentasi foto video acara lamaran', 'ongoing', NULL, '2025-12-01 22:32:19', 4000000),
('dfbf9569-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Foto Video Full Day', 'Full Day Coverage', 'Dokumentasi foto video seharian penuh', 'ongoing', NULL, '2025-12-01 22:32:19', 18000000),
('dfbf95b3-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Video Booth 360', '360 Video Booth', 'Sewa video booth 360 derajat untuk resepsi', 'ongoing', NULL, '2025-12-01 22:32:19', 5000000),
('dfbf9600-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Fotografer Tambahan', 'Additional Photographer', 'Penambahan fotografer profesional per jam', 'ongoing', NULL, '2025-12-01 22:32:19', 800000),
('dfbf9645-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Cetak Foto Jumbo', 'Jumbo Photo Print', 'Cetak foto jumbo ukuran 60x90 cm', 'ongoing', NULL, '2025-12-01 22:32:19', 500000),
('dfbf968f-ceca-11f0-89f6-a036bc697a25', 'b4c13e15-6725-49c2-946e-98f57895cad3', 'Paket Same Day Edit Video', 'Same Day Edit', 'Video highlight selesai di hari yang sama', 'ongoing', NULL, '2025-12-01 22:32:19', 10000000),
('dfc20497-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Sunda Pengantin', 'Kebaya Sunda Bandung', 'Sewa kebaya sunda untuk pengantin lengkap dengan aksesoris di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 5000000);
INSERT INTO `items` (`id`, `category_id`, `name`, `vendor`, `description`, `status`, `image_url`, `last_update`, `price`) VALUES
('dfc20f6d-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Adat Sunda Pria', 'Busana Pria Sunda', 'Sewa baju adat sunda untuk pengantin pria di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 4000000),
('dfc2104b-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Sunda Premium', 'Premium Sunda Attire', 'Kebaya sunda premium dengan bordir mewah', 'ongoing', NULL, '2025-12-01 22:32:19', 7500000),
('dfc210d7-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Jawa Pengantin', 'Javanese Wedding Attire', 'Sewa kebaya jawa untuk pengantin', 'ongoing', NULL, '2025-12-01 22:32:19', 4500000),
('dfc21151-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Beskap Jawa Pria', 'Beskap Javanese', 'Sewa beskap jawa untuk pengantin pria', 'ongoing', NULL, '2025-12-01 22:32:19', 3500000),
('dfc211cd-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Bodo Bugis', 'Bugis Traditional Attire', 'Sewa baju bodo untuk pengantin bugis', 'ongoing', NULL, '2025-12-01 22:32:19', 4000000),
('dfc2123c-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Modern Minimalis', 'Modern Kebaya Rental', 'Kebaya modern minimalis untuk resepsi', 'ongoing', NULL, '2025-12-01 22:32:19', 3500000),
('dfc212a2-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Gaun Pengantin International', 'International Bridal Gown', 'Gaun pengantin gaya international', 'ongoing', NULL, '2025-12-01 22:32:19', 8000000),
('dfc21310-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Jas Pengantin Pria', 'Groom Suit Rental', 'Jas pengantin pria model modern', 'ongoing', NULL, '2025-12-01 22:32:19', 3000000),
('dfc21378-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Adat Betawi', 'Betawi Wedding Attire', 'Baju adat betawi untuk pengantin', 'ongoing', NULL, '2025-12-01 22:32:19', 4200000),
('dfc213dc-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Adat Minang', 'Minang Traditional Dress', 'Baju adat minang lengkap dengan aksesoris', 'ongoing', NULL, '2025-12-01 22:32:19', 4500000),
('dfc2144b-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Akad Nikah', 'Akad Kebaya', 'Kebaya khusus untuk acara akad nikah', 'ongoing', NULL, '2025-12-01 22:32:19', 2500000),
('dfc214b3-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Adat Bali', 'Balinese Wedding Attire', 'Baju adat bali untuk pengantin', 'ongoing', NULL, '2025-12-01 22:32:19', 5000000),
('dfc2151e-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Bridesmaid', 'Bridesmaid Kebaya', 'Kebaya seragam untuk bridesmaid (per set)', 'ongoing', NULL, '2025-12-01 22:32:19', 800000),
('dfc21581-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Sunda Pagar Ayu', 'Pagar Ayu Dress', 'Baju sunda untuk pagar ayu (per orang)', 'ongoing', NULL, '2025-12-01 22:32:19', 600000),
('dfc215e7-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Gaun Malam Resepsi', 'Evening Gown', 'Gaun malam untuk resepsi malam', 'ongoing', NULL, '2025-12-01 22:32:19', 4500000),
('dfc2165b-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Lamaran', 'Engagement Kebaya', 'Kebaya untuk acara lamaran', 'ongoing', NULL, '2025-12-01 22:32:19', 2000000),
('dfc216bf-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Batik Couple', 'Batik Couple Set', 'Batik couple untuk acara informal', 'ongoing', NULL, '2025-12-01 22:32:19', 1500000),
('dfc21724-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Kebaya Encim Betawi', 'Kebaya Encim', 'Kebaya encim khas betawi', 'ongoing', NULL, '2025-12-01 22:32:19', 3800000),
('dfc2178b-ceca-11f0-89f6-a036bc697a25', 'ae51cb01-9c3c-4f3d-824d-61d331d57892', 'Sewa Baju Kurung Melayu', 'Malay Traditional Dress', 'Baju kurung melayu untuk pengantin', 'ongoing', NULL, '2025-12-01 22:32:19', 3500000),
('dfc482df-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Band Akustik Sunda', 'Sunda Acoustic Band', 'Band akustik dengan lagu-lagu sunda untuk pernikahan di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 8000000),
('dfc48eca-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Gamelan Sunda', 'Gamelan Sunda Bandung', 'Pertunjukan gamelan sunda untuk pernikahan adat', 'ongoing', NULL, '2025-12-01 22:32:19', 12000000),
('dfc49058-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Angklung Performance', 'Saung Angklung Bandung', 'Pertunjukan angklung untuk acara pernikahan', 'ongoing', NULL, '2025-12-01 22:32:19', 10000000),
('dfc490e6-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Band Full Jakarta', 'Jakarta Wedding Band', 'Band lengkap dengan penyanyi untuk pernikahan', 'ongoing', NULL, '2025-12-01 22:32:19', 15000000),
('dfc4917e-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'DJ dan Sound System', 'Professional DJ Service', 'DJ profesional dengan sound system lengkap', 'ongoing', NULL, '2025-12-01 22:32:19', 6000000),
('dfc4af63-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Penyanyi Solo', 'Solo Singer Performance', 'Penyanyi solo untuk acara pernikahan', 'ongoing', NULL, '2025-12-01 22:32:19', 5000000),
('dfc4b00c-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Orkestra Wedding', 'Wedding Orchestra', 'Orkestra lengkap untuk pernikahan mewah', 'ongoing', NULL, '2025-12-01 22:32:19', 25000000),
('dfc4b05d-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Gamelan Jawa', 'Javanese Gamelan', 'Gamelan jawa untuk pernikahan adat', 'ongoing', NULL, '2025-12-01 22:32:19', 11000000),
('dfc4b0a4-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Talempong Minang', 'Minang Traditional Music', 'Musik tradisional minang untuk pernikahan', 'ongoing', NULL, '2025-12-01 22:32:19', 9000000),
('dfc4b233-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Keyboard Player', 'Professional Keyboardist', 'Pemain keyboard untuk acara pernikahan', 'ongoing', NULL, '2025-12-01 22:32:19', 3000000),
('dfc4b282-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Tari Sunda Welcome Dance', 'Sunda Welcome Dancer', 'Tarian penyambutan tamu ala sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 5000000),
('dfc4b2c7-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Tari Jaipong', 'Jaipong Dance Performance', 'Pertunjukan tari jaipong khas sunda', 'ongoing', NULL, '2025-12-01 22:32:19', 6000000),
('dfc4b305-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Tari Topeng Betawi', 'Betawi Mask Dance', 'Tarian topeng betawi untuk pernikahan', 'ongoing', NULL, '2025-12-01 22:32:19', 5500000),
('dfc4b342-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Marching Band', 'Wedding Marching Band', 'Marching band untuk menyambut pengantin', 'ongoing', NULL, '2025-12-01 22:32:19', 8000000),
('dfc4b382-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'String Quartet', 'Classical String Quartet', 'String quartet untuk acara akad nikah', 'ongoing', NULL, '2025-12-01 22:32:19', 12000000),
('dfc4b3bb-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Jazz Band', 'Jazz Wedding Band', 'Band jazz untuk resepsi malam', 'ongoing', NULL, '2025-12-01 22:32:19', 13000000),
('dfc4b3fe-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Acoustic Duo', 'Acoustic Duo Performance', 'Duo akustik untuk pernikahan intimate', 'ongoing', NULL, '2025-12-01 22:32:19', 4500000),
('dfc4b439-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Tari Bali', 'Balinese Dance', 'Tarian bali untuk pernikahan', 'ongoing', NULL, '2025-12-01 22:32:19', 7000000),
('dfc4b47a-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Musik Keroncong', 'Keroncong Music', 'Musik keroncong untuk pernikahan tradisional', 'ongoing', NULL, '2025-12-01 22:32:19', 6500000),
('dfc4b4b2-ceca-11f0-89f6-a036bc697a25', 'bef9f40f-4b2b-4a33-8ef2-854566aa2dc2', 'Wayang Kulit', 'Wayang Performance', 'Pertunjukan wayang kulit untuk pernikahan adat jawa', 'ongoing', NULL, '2025-12-01 22:32:19', 15000000),
('dfc6ec92-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Pernikahan Adat Sunda', 'MC Sunda Professional Bandung', 'MC profesional untuk pernikahan adat sunda di Bandung, bisa bahasa sunda', 'ongoing', NULL, '2025-12-01 22:32:19', 3500000),
('dfc6f704-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Bilingual Sunda-Indonesia', 'Bilingual MC Bandung', 'MC bilingual sunda dan indonesia untuk pernikahan di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 4000000),
('dfc6f799-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Pernikahan Adat Jawa', 'Javanese MC', 'MC profesional untuk pernikahan adat jawa', 'ongoing', NULL, '2025-12-01 22:32:19', 3500000),
('dfc6f7e7-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Modern Jakarta', 'Modern MC Jakarta', 'MC modern untuk pernikahan kontemporer', 'ongoing', NULL, '2025-12-01 22:32:19', 4500000),
('dfc6f82b-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Bilingual Indonesia-English', 'International MC', 'MC bilingual untuk pernikahan international', 'ongoing', NULL, '2025-12-01 22:32:19', 6000000),
('dfc6f86f-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Adat Betawi', 'Betawi MC Jakarta', 'MC untuk pernikahan adat betawi', 'ongoing', NULL, '2025-12-01 22:32:19', 3800000),
('dfc6f8af-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Adat Minang', 'Minang MC', 'MC untuk pernikahan adat minang', 'ongoing', NULL, '2025-12-01 22:32:19', 4000000),
('dfc6f8fa-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Adat Bali', 'Balinese MC', 'MC untuk pernikahan adat bali', 'ongoing', NULL, '2025-12-01 22:32:19', 4200000),
('dfc6f958-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Couple Team', 'MC Couple Professional', 'Tim MC couple (pria dan wanita)', 'ongoing', NULL, '2025-12-01 22:32:19', 7000000),
('dfc6f9b2-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Celebrity', 'Celebrity MC Service', 'MC selebriti untuk pernikahan eksklusif', 'ongoing', NULL, '2025-12-01 22:32:19', 15000000),
('dfc6fa07-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Akad Nikah', 'Akad Ceremony MC', 'MC khusus untuk acara akad nikah', 'ongoing', NULL, '2025-12-01 22:32:19', 2500000),
('dfc6fa5d-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Full Day Service', 'Full Day MC', 'MC seharian penuh dari akad sampai resepsi', 'ongoing', NULL, '2025-12-01 22:32:19', 6500000),
('dfc6faaf-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Trilingual', 'Trilingual MC Service', 'MC trilingual (Indonesia, English, Mandarin)', 'ongoing', NULL, '2025-12-01 22:32:19', 8000000),
('dfc6fb0a-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Intimate Wedding', 'Intimate MC', 'MC untuk pernikahan intimate (max 100 tamu)', 'ongoing', NULL, '2025-12-01 22:32:19', 2500000),
('dfc6fb6f-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Pernikahan Tradisional Nusantara', 'Nusantara MC', 'MC untuk pernikahan traditional nusantara', 'ongoing', NULL, '2025-12-01 22:32:19', 4000000),
('dfc6fbcd-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Lamaran', 'Engagement MC', 'MC untuk acara lamaran', 'ongoing', NULL, '2025-12-01 22:32:19', 2000000),
('dfc6fc2e-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Comedian Style', 'Comedian MC', 'MC dengan gaya comedy untuk suasana ceria', 'ongoing', NULL, '2025-12-01 22:32:19', 5000000),
('dfc6fc8a-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Formal Elegant', 'Elegant MC Service', 'MC dengan gaya formal elegant', 'ongoing', NULL, '2025-12-01 22:32:19', 5500000),
('dfc6fce8-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Outdoor Wedding', 'Outdoor MC Specialist', 'MC spesialis untuk outdoor wedding', 'ongoing', NULL, '2025-12-01 22:32:19', 4000000),
('dfc6fd3e-ceca-11f0-89f6-a036bc697a25', 'b123275f-af7b-4f75-9d9f-bdd8ad899dd7', 'MC Virtual Hybrid', 'Hybrid Wedding MC', 'MC untuk pernikahan hybrid (offline + online)', 'ongoing', NULL, '2025-12-01 22:32:19', 4500000),
('dfc9758b-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Ngeuyeuk Seureuh Sunda', 'Sanggar Sunda Bandung', 'Upacara ngeuyeuk seureuh adat sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 5000000),
('dfc97f85-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Siraman Sunda', 'Adat Sunda Bandung', 'Upacara siraman adat sunda lengkap di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 6000000),
('dfc98021-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Sawer Sunda', 'Tradisi Sunda Bandung', 'Upacara sawer pengantin adat sunda', 'ongoing', NULL, '2025-12-01 22:32:19', 3000000),
('dfc98072-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Paket Lengkap Upacara Sunda', 'Complete Sunda Ceremony', 'Paket lengkap semua upacara adat sunda di Bandung', 'ongoing', NULL, '2025-12-01 22:32:19', 15000000),
('dfc980b5-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Huap Lingkung Sunda', 'Sunda Wedding Ritual', 'Upacara huap lingkung adat sunda', 'ongoing', NULL, '2025-12-01 22:32:19', 4000000),
('dfc980f1-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Penyewaan Baki Seserahan Sunda', 'Sunda Hantaran Rental', 'Sewa baki hantaran khas sunda (7 nampan)', 'ongoing', NULL, '2025-12-01 22:32:19', 1500000),
('dfc9812c-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Jasa Cok Bakal Seserahan Sunda', 'Sunda Hantaran Decorator', 'Jasa menghias seserahan adat sunda', 'ongoing', NULL, '2025-12-01 22:32:19', 1000000),
('dfc98168-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Pagar Ayu Sunda (6 orang)', 'Pagar Ayu Service', 'Penyediaan 6 pagar ayu berpakaian sunda lengkap', 'ongoing', NULL, '2025-12-01 22:32:19', 3500000),
('dfc981a6-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Sewa Payung Geulis Sunda', 'Payung Geulis Rental', 'Sewa payung geulis khas sunda (per unit)', 'ongoing', NULL, '2025-12-01 22:32:19', 200000),
('dfc981e6-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Jasa Pembaca Doa Sunda', 'Sunda Prayer Reader', 'Jasa pembaca doa dalam bahasa sunda', 'ongoing', NULL, '2025-12-01 22:32:19', 2000000),
('dfc9821e-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Midodareni Jawa', 'Javanese Midodareni', 'Upacara midodareni adat jawa', 'ongoing', NULL, '2025-12-01 22:32:19', 7000000),
('dfc98256-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Ngunduh Mantu Jawa', 'Javanese Ngunduh Mantu', 'Prosesi ngunduh mantu adat jawa', 'ongoing', NULL, '2025-12-01 22:32:19', 8000000),
('dfc98291-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Peningsetan Bali', 'Balinese Ceremony', 'Upacara peningsetan adat bali', 'ongoing', NULL, '2025-12-01 22:32:19', 9000000),
('dfc982c9-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Malam Bainai Minang', 'Minang Bainai Night', 'Upacara malam bainai adat minang', 'ongoing', NULL, '2025-12-01 22:32:19', 6000000),
('dfc98303-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Akad Nikah Betawi', 'Betawi Akad Ceremony', 'Prosesi akad nikah adat betawi lengkap', 'ongoing', NULL, '2025-12-01 22:32:19', 5000000),
('dfc9833e-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Sewa Pelaminan Adat Sunda', 'Sunda Pelaminan Rental', 'Sewa pelaminan adat sunda lengkap', 'ongoing', NULL, '2025-12-01 22:32:19', 8000000),
('dfc98383-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Adat Lengkap Jawa', 'Complete Javanese Ceremony', 'Paket lengkap upacara adat jawa', 'ongoing', NULL, '2025-12-01 22:32:19', 20000000),
('dfc983bd-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Mappacci Bugis', 'Bugis Mappacci Ceremony', 'Upacara mappacci adat bugis', 'ongoing', NULL, '2025-12-01 22:32:19', 6500000),
('dfc983f9-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Tepung Tawar Melayu', 'Malay Tepung Tawar', 'Upacara tepung tawar adat melayu', 'ongoing', NULL, '2025-12-01 22:32:19', 4500000),
('dfc98436-ceca-11f0-89f6-a036bc697a25', 'c5b14377-fa85-4345-bdc2-2a962c067fd2', 'Upacara Adat Sunda Komplit', 'Complete Sunda Traditional', 'Paket komplit semua upacara sunda dari awal sampai akhir', 'ongoing', NULL, '2025-12-01 22:32:19', 25000000);

-- --------------------------------------------------------

--
-- Table structure for table `landing_page`
--

CREATE TABLE `landing_page` (
  `id` char(36) NOT NULL DEFAULT uuid(),
  `image_url` varchar(255) NOT NULL,
  `caption` text DEFAULT NULL,
  `last_update` datetime DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `our_event`
--

CREATE TABLE `our_event` (
  `id` int(11) NOT NULL,
  `image` text DEFAULT NULL,
  `description` text DEFAULT NULL,
  `last_update` datetime DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `status` varchar(50) DEFAULT 'archived' CHECK (`status` in ('airing','archived'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `testimonials`
--

CREATE TABLE `testimonials` (
  `id` char(36) NOT NULL DEFAULT uuid(),
  `name` varchar(255) DEFAULT NULL,
  `rating` decimal(5,2) DEFAULT NULL CHECK (`rating` >= 0 and `rating` <= 5),
  `comment` text DEFAULT NULL,
  `created_at` datetime DEFAULT current_timestamp(),
  `status` varchar(50) DEFAULT 'draft' CHECK (`status` in ('airing','draft','ongoing'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `visitors`
--

CREATE TABLE `visitors` (
  `id` bigint(20) NOT NULL,
  `visit_date` date NOT NULL DEFAULT curdate(),
  `total_visits` int(11) DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `visitor_ips`
--

CREATE TABLE `visitor_ips` (
  `id` bigint(20) NOT NULL,
  `ip_address` varchar(45) NOT NULL,
  `user_agent` text DEFAULT NULL,
  `visit_date` date NOT NULL DEFAULT curdate(),
  `created_at` datetime DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `categories`
--
ALTER TABLE `categories`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `category_logs`
--
ALTER TABLE `category_logs`
  ADD PRIMARY KEY (`id`),
  ADD KEY `category_logs_category_id_fkey` (`category_id`);

--
-- Indexes for table `items`
--
ALTER TABLE `items`
  ADD PRIMARY KEY (`id`),
  ADD KEY `items_category_id_fkey` (`category_id`);

--
-- Indexes for table `landing_page`
--
ALTER TABLE `landing_page`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `our_event`
--
ALTER TABLE `our_event`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `testimonials`
--
ALTER TABLE `testimonials`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `visitors`
--
ALTER TABLE `visitors`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `visitor_ips`
--
ALTER TABLE `visitor_ips`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `category_logs`
--
ALTER TABLE `category_logs`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `our_event`
--
ALTER TABLE `our_event`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `visitors`
--
ALTER TABLE `visitors`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `visitor_ips`
--
ALTER TABLE `visitor_ips`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `category_logs`
--
ALTER TABLE `category_logs`
  ADD CONSTRAINT `category_logs_category_id_fkey` FOREIGN KEY (`category_id`) REFERENCES `categories` (`id`);

--
-- Constraints for table `items`
--
ALTER TABLE `items`
  ADD CONSTRAINT `items_category_id_fkey` FOREIGN KEY (`category_id`) REFERENCES `categories` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
