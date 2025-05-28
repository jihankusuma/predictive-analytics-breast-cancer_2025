# **Laporan Proyek Machine Learning - Jihan Kusumawardhani**

## **Domain Proyek**
Kanker payudara adalah salah satu masalah kesehatan global yang paling signifikan, menjadi salah satu penyebab utama kematian akibat kanker pada wanita. Deteksi dini merupakan faktor krusial untuk meningkatkan tingkat kelangsungan hidup dan keberhasilan pengobatan. Dengan memanfaatkan kemajuan dalam *machine learning*, kita dapat mengembangkan sistem prediktif untuk membantu para profesional medis dalam mendiagnosis tumor payudara secara akurat berdasarkan data pengukuran sel, sehingga mempercepat proses diagnosis dan mengurangi potensi kesalahan manusia.

## **Business Understanding**
Bagian laporan ini mencakup:

### **Problem Statements**
Menjelaskan pernyataan masalah latar belakang:
* Bagaimana cara membangun sebuah model *machine learning* yang mampu memprediksi secara akurat apakah sebuah tumor payudara bersifat ganas (*Malignant*) atau jinak (*Benign*) berdasarkan fitur-fitur pengukuran seluler?
* Di antara berbagai algoritma klasifikasi yang ada, manakah yang memberikan performa paling optimal untuk dataset kanker payudara ini?

### **Goals**
Menjelaskan tujuan dari pernyataan masalah:
* Membangun beberapa model klasifikasi untuk memprediksi diagnosis kanker payudara.
* Mengevaluasi dan membandingkan akurasi serta metrik relevan lainnya dari setiap model untuk mengidentifikasi algoritma yang paling efektif.

### **Solution statements**
* Menggunakan beberapa algoritma klasifikasi seperti Random Forest, K-Nearest Neighbor (KNN), Support Vector Machine (SVM), Gradient Boosting, Logistic Regression, dan Gaussian Naive Bayes.
* Metrik yang digunakan untuk evaluasi adalah *accuracy*, *precision*, *recall*, dan *F1-score* yang diambil dari *classification report*.

## **Data Understanding**
Dataset yang digunakan adalah "Breast Cancer Wisconsin (Diagnostic) Data Set" yang diperoleh dari platform Kaggle. Dataset ini berisi fitur-fitur yang diekstraksi dari citra digital aspirasi jarum halus (*Fine Needle Aspiration* - FNA) dari massa payudara.

### **URL Dataset**
Link Dataset : [https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

### **Jumlah Baris dan Kolom**
Dataset ini memiliki 569 baris data dan 32 kolom.

### **Kondisi**
Kondisi dataset sangat baik, terbukti dengan tidak adanya nilai yang hilang (*missing value*) pada seluruh kolom. Dataset ini terdiri dari 31 kolom numerik (30 `float64` dan 1 `int64`) dan 1 kolom `object` yaitu `diagnosis` yang menjadi target prediksi.

### **Variabel-variabel pada dataset adalah sebagai berikut:**

## Fitur Utama 

* `diagnosis`: **Variabel target** yang menunjukkan hasil diagnosis (**M** = Malignant/Ganas, **B** = Benign/Jinak).
* `radius_mean`: Rata-rata jari-jari dari inti sel tumor.
* `texture_mean`: Rata-rata tekstur (standar deviasi nilai *grayscale*).
* `perimeter_mean`: Rata-rata keliling inti sel.
* `area_mean`: Rata-rata luas inti sel.
* `smoothness_mean`: Rata-rata kehalusan kontur inti sel.
* `compactness_mean`: Rata-rata tingkat kepadatan inti sel (keliling² / luas — 1.0).
* `concavity_mean`: Rata-rata tingkat kecekungan pada kontur inti sel.
* `concave points_mean`: Rata-rata jumlah titik cekung pada kontur.
* `symmetry_mean`: Rata-rata simetri inti sel.
* `fractal_dimension_mean`: Rata-rata dimensi fraktal dari kontur ("coastline approximation — 1").

### Fitur Turunan

Selain fitur utama, dataset ini juga mencakup pengukuran statistik turunan untuk ke-10 fitur di atas, yang direpresentasikan dengan akhiran:

* `_se`: **Standard Error** (contoh: `radius_se`, `texture_se`, dll.).
* `_worst`: Nilai **terburuk** atau terbesar yang dihitung dari mean terbesar (contoh: `radius_worst`, `texture_worst`, dll.).

### Fitur yang Dihilangkan

* `id`: Parameter unik untuk identifikasi, **tidak relevan** untuk pemodelan dan akan dihilangkan dari dataset.

---

## **Data Preparation**

### 1. Penanganan Fitur Target (`diagnosis`)

* **Pengecekan Fitur Target**: Kolom `diagnosis` merupakan fitur target berjenis *object*. Kategori `B` (Benign) mengindikasikan kanker jinak, sementara `M` (Malignant) mengindikasikan kanker ganas. Fitur inilah yang akan menjadi target prediksi dalam proyek ini.
* **Encoding Numerik**: Untuk memungkinkan model memahami fitur ini, kami mengubah tipe data `diagnosis` dari *object* ke numerik. Kanker jinak (`B`) di-*mapping* menjadi `0`, dan kanker ganas (`M`) di-*mapping* menjadi `1`.
* **Distribusi Kelas**: Setelah *mapping*, dilakukan perhitungan jumlah baris untuk setiap kategori pada kolom target untuk melihat distribusinya.

### 2. Pembagian Dataset

Dataset dibagi menjadi dua bagian:

* **Data Latih (80%)**: Digunakan untuk melatih model agar dapat mempelajari pola dari data.
* **Data Uji (20%)**: Digunakan untuk menguji performa model yang telah dilatih pada data yang belum pernah dilihat sebelumnya.

Pembagian ini dilakukan menggunakan fungsi `train_test_split` dari pustaka `scikit-learn`.

### 3. Standardisasi Data

Semua fitur numerik dalam dataset distandardisasi. Proses ini bertujuan untuk membuat semua fitur berada dalam skala yang seragam (biasanya dengan rata-rata 0 dan standar deviasi 1). Standardisasi dilakukan menggunakan `StandardScaler` dari `scikit-learn` dengan rumus:

$$z = \frac{(x - \mu)}{\sigma}$$

di mana:
* $z$ adalah nilai terstandardisasi
* $x$ adalah nilai fitur asli
* $\mu$ adalah rata-rata fitur
* $\sigma$ adalah standar deviasi fitur


## **Modeling**
Model yang digunakan untuk proyek ini dijelaskan secara rinci sebagai berikut.

### **1. Logistic Regression**
Logistic Regression adalah metode klasifikasi statistik yang digunakan untuk memprediksi hasil biner (dua kelas). Model ini bekerja dengan menerapkan fungsi logistik (sigmoid) pada kombinasi linear dari fitur-fitur input untuk menghasilkan probabilitas.

#### **Cara Kerja**
Fungsi Logistik (Sigmoid): Logistic regression menggunakan fungsi logistik untuk memetakan hasil regresi linear ke dalam rentang probabilitas antara 0 dan 1. Proses pelatihan melibatkan pencarian bobot (*weights*) dan bias yang meminimalkan fungsi kerugian (*loss function*), yang biasanya menggunakan *binary cross-entropy loss*.

$$P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + ... + \beta_nX_n)}}$$

#### **Parameter**
* **Koefisien (weights)**: Parameter yang menentukan seberapa besar pengaruh setiap fitur input terhadap prediksi output. Nilai koefisien yang besar menunjukkan pengaruh yang kuat.
* **Bias (intercept)**: Parameter tambahan yang memungkinkan model untuk memiliki fleksibilitas dan tidak harus melewati titik asal (0,0).
* **Solver**: Algoritma yang digunakan dalam proses optimisasi untuk menemukan parameter terbaik (misalnya, 'liblinear', 'lbfgs').

#### **Kelebihan**
* Sangat cepat, efisien secara komputasi, dan tidak memerlukan banyak sumber daya.
* Hasilnya mudah diinterpretasikan karena koefisien dapat menjelaskan hubungan antara setiap fitur dengan probabilitas hasil.
* Berfungsi sangat baik untuk masalah yang dapat dipisahkan secara linear.

#### **Kekurangan**
* Cenderung memiliki performa yang kurang baik pada masalah dengan hubungan non-linear yang kompleks.
* Rentan terhadap *underfitting* jika batas keputusan antar kelas terlalu rumit.

### **2. Random Forest**
Random Forest adalah algoritma *ensemble learning* yang membangun banyak pohon keputusan (*decision trees*) pada berbagai sub-sampel dari dataset dan menggunakan *voting* untuk meningkatkan akurasi prediksi dan mengontrol *overfitting*.

#### **Cara Kerja**
* **Bootstrap Aggregating (Bagging)**: Algoritma ini membuat banyak dataset baru dengan mengambil sampel secara acak dengan pengulangan (*with replacement*) dari dataset asli. Setiap dataset *bootstrap* ini digunakan untuk melatih satu pohon keputusan.
* **Pembangunan Pohon Keputusan dengan Fitur Acak**: Saat membangun setiap pohon, pada setiap *split* atau percabangan, model hanya mempertimbangkan sebagian kecil fitur yang dipilih secara acak, bukan semua fitur. Ini memastikan bahwa pohon-pohon yang dibangun bervariasi.
* **Voting/Averaging**: Setelah semua pohon selesai dilatih, Random Forest menggabungkan hasil prediksi dari seluruh pohon. Untuk klasifikasi (seperti pada proyek ini), prediksi akhir adalah kelas yang paling banyak dipilih (*majority vote*) oleh semua pohon.

#### **Parameter**
* **n\_estimators**: Jumlah pohon keputusan yang akan dibangun dalam hutan. Semakin banyak pohon, semakin stabil hasilnya, namun waktu komputasi juga meningkat.
* **max\_features**: Jumlah maksimum fitur yang akan dipertimbangkan untuk setiap *split* pada pohon.
* **max\_depth**: Kedalaman maksimum setiap pohon. Ini membatasi jumlah *split* yang dapat dilakukan, membantu mengontrol *overfitting*.
* **min\_samples\_split**: Jumlah minimum sampel yang diperlukan untuk membagi sebuah *node*.
* **min\_samples\_leaf**: Jumlah minimum sampel yang harus ada di sebuah daun (*leaf node*) setelah *split*.

#### **Kelebihan**
* Menghasilkan akurasi yang sangat tinggi dan merupakan salah satu algoritma klasifikasi terbaik.
* Sangat kuat terhadap *overfitting* berkat proses *bagging* dan pemilihan fitur acak.
* Mampu menangani data dalam jumlah besar dengan ribuan fitur tanpa perlu melakukan seleksi fitur.

#### **Kekurangan**
* Cenderung merupakan model *black box*, artinya sulit untuk menginterpretasikan bagaimana model membuat prediksi secara spesifik.
* Membutuhkan sumber daya komputasi dan memori yang lebih besar dibandingkan model tunggal.

### **3. Support Vector Machine (SVM)**
SVM adalah algoritma klasifikasi yang bertujuan untuk menemukan *hyperplane* terbaik dalam ruang N-dimensi (N adalah jumlah fitur) yang secara jelas memisahkan titik-titik data ke dalam kelas-kelas yang berbeda dengan margin semaksimal mungkin.

#### **Cara Kerja**
* **Hyperplane dan Margin**: SVM mencari sebuah batas keputusan (*hyperplane*) yang tidak hanya memisahkan dua kelas, tetapi juga memiliki jarak (margin) yang paling besar ke titik data terdekat dari masing-masing kelas.
* **Support Vectors**: Titik-titik data yang berada tepat di batas margin disebut *support vectors*. Titik-titik inilah yang menentukan posisi dan orientasi dari *hyperplane*.
* **Kernel Trick**: Untuk data yang tidak dapat dipisahkan secara linear, SVM menggunakan fungsi *kernel* (misalnya Linear, RBF, Polinomial) untuk memetakan data ke ruang dimensi yang lebih tinggi di mana pemisahan linear menjadi mungkin.

#### **Parameter**
* **C (Regularization parameter)**: Mengontrol keseimbangan antara memaksimalkan margin dan meminimalkan kesalahan klasifikasi. Nilai C yang kecil menciptakan margin yang besar namun mengizinkan beberapa kesalahan klasifikasi, sementara nilai C yang besar mencoba mengklasifikasikan semua sampel dengan benar yang berisiko *overfitting*.
* **Kernel**: Fungsi yang digunakan untuk mengubah data. Pilihan kernel ('linear', 'rbf', 'poly') sangat menentukan performa model.
* **Gamma**: Parameter untuk kernel non-linear (seperti RBF). Gamma mendefinisikan seberapa jauh pengaruh dari satu sampel pelatihan. Nilai gamma yang tinggi berarti pengaruhnya dekat, yang dapat menyebabkan batas keputusan yang lebih kompleks dan berisiko *overfitting*.

#### **Kelebihan**
* Efektif dalam ruang berdimensi tinggi, bahkan jika jumlah dimensi lebih besar dari jumlah sampel.
* Hemat memori karena hanya subset dari titik pelatihan (*support vectors*) yang digunakan dalam fungsi keputusan.
* Sangat serbaguna berkat berbagai pilihan fungsi *kernel*.

#### **Kekurangan**
* Tidak cocok untuk dataset yang sangat besar karena waktu pelatihannya bisa sangat lama (kompleksitas $O(n^3)$ atau $O(n^2)$).
* Performa sangat bergantung pada pemilihan parameter C dan jenis *kernel* yang tepat.

### **4. K-Nearest Neighbors (KNN)**
KNN adalah algoritma non-parametrik sederhana yang mengklasifikasikan data baru berdasarkan mayoritas kelas dari 'k' tetangga terdekatnya di ruang fitur.

#### **Cara Kerja**
* **Penyimpanan Data**: KNN menyimpan seluruh dataset pelatihan selama fase "pelatihan".
* **Perhitungan Jarak**: Saat memprediksi data baru, algoritma menghitung jarak (misalnya, Jarak Euclidean) antara titik data baru tersebut dengan semua titik data dalam dataset pelatihan.
* **Menemukan Tetangga**: Algoritma mengidentifikasi 'k' titik data pelatihan dengan jarak terpendek (tetangga terdekat).
* **Voting Kelas**: Titik data baru kemudian diklasifikasikan ke dalam kelas yang paling umum (mayoritas) di antara 'k' tetangga terdekat tersebut.

#### **Parameter**
* **n\_neighbors (k)**: Jumlah tetangga terdekat yang akan digunakan untuk *voting*. Ini adalah parameter paling krusial.
* **metric**: Metrik jarak yang digunakan untuk mengukur kedekatan antar titik data (misalnya, 'euclidean', 'manhattan').
* **weights**: Bobot yang diberikan pada setiap tetangga dalam *voting*. Bisa 'uniform' (semua tetangga punya bobot sama) atau 'distance' (tetangga yang lebih dekat punya pengaruh lebih besar).

#### **Kelebihan**
* Sangat mudah dipahami dan diimplementasikan.
* Tidak memerlukan fase pelatihan eksplisit, membuatnya cepat untuk mulai digunakan (*lazy learner*).
* Secara alami dapat menangani masalah klasifikasi multi-kelas.

#### **Kekurangan**
* Proses prediksi bisa menjadi sangat lambat dan mahal secara komputasi pada dataset besar.
* Sangat sensitif terhadap skala fitur, sehingga standardisasi atau normalisasi data adalah langkah wajib.
* Performa menurun drastis pada data berdimensi tinggi (*curse of dimensionality*).

### **5. Gradient Boosting Machines (GBM)**
Gradient Boosting adalah teknik *ensemble* yang membangun model secara sekuensial, di mana setiap model baru berfokus untuk memperbaiki kesalahan dari model sebelumnya.

#### **Cara Kerja**
* **Model Awal**: Proses dimulai dengan model sederhana, sering kali hanya prediksi rata-rata dari nilai target.
* **Pembelajaran Iteratif**: Secara iteratif, algoritma melatih model baru (biasanya pohon keputusan) untuk memprediksi *residual* (selisih antara nilai aktual dan prediksi) dari model sebelumnya.
* **Pembaruan Model**: Prediksi dari model baru ini kemudian ditambahkan ke prediksi model gabungan sebelumnya, disesuaikan dengan *learning rate* untuk mencegah *overfitting*. Proses ini pada dasarnya adalah penurunan gradien pada fungsi kerugian.

#### **Parameter**
* **n\_estimators**: Jumlah total model (pohon) yang akan dibangun secara berurutan.
* **learning\_rate**: Faktor penyusutan (antara 0 dan 1) yang mengontrol seberapa besar kontribusi setiap pohon terhadap hasil akhir.
* **max\_depth**: Kedalaman maksimum dari setiap pohon keputusan individu untuk mengontrol kompleksitas model.
* **subsample**: Fraksi dari sampel pelatihan yang digunakan untuk melatih setiap pohon, menambahkan unsur keacakan (Stochastic Gradient Boosting).

#### **Kelebihan**
* Sering kali menghasilkan akurasi prediksi yang sangat tinggi, salah satu yang terbaik di kelasnya.
* Sangat fleksibel dan dapat dioptimalkan untuk berbagai fungsi kerugian.

#### **Kekurangan**
* Rentan terhadap *overfitting* jika tidak di-tuning dengan hati-hati (terutama `n_estimators` dan `learning_rate`).
* Pelatihan bisa lambat karena sifatnya yang sekuensial (tidak dapat diparalelkan).
* Lebih sulit untuk diinterpretasikan dibandingkan model yang lebih sederhana.

### **6. Gaussian Naive Bayes (GNB)**
Naive Bayes adalah klasifikasi probabilistik yang didasarkan pada Teorema Bayes dengan asumsi "naif" bahwa semua fitur bersifat independen satu sama lain. Varian Gaussian digunakan ketika fitur-fitur bersifat kontinu dan diasumsikan mengikuti distribusi normal (Gaussian).

#### **Cara Kerja**
* **Teorema Bayes**: Model ini menghitung probabilitas posterior $P(Kelas | Fitur)$, yaitu probabilitas sebuah data masuk ke kelas tertentu dengan adanya fitur-fitur tersebut.
* **Asumsi Independensi Naif**: Diasumsikan bahwa setiap fitur memberikan kontribusi independen terhadap probabilitas. Ini menyederhanakan perhitungan menjadi: $P(Kelas | Fitur) \propto P(Kelas) \times \prod_{i=1}^{n} P(Fitur_i | Kelas)$.
* **Distribusi Gaussian**: Untuk fitur kontinu, probabilitas $P(Fitur_i | Kelas)$ dihitung menggunakan rumus Probability Density Function (PDF) dari distribusi Gaussian, yang memerlukan rata-rata dan varians dari setiap fitur untuk setiap kelas.

#### **Parameter**
* Gaussian Naive Bayes hampir tidak memiliki *hyperparameter* untuk di-tuning. Parameter utamanya (rata-rata dan varians untuk setiap fitur per kelas) dipelajari langsung dari data pelatihan.
* **var\_smoothing**: Sejumlah kecil nilai yang ditambahkan ke varians untuk tujuan stabilitas numerik, terutama jika ada varians yang bernilai nol.

#### **Kelebihan**
* Sangat cepat, efisien, dan bekerja dengan baik pada dataset yang sangat besar.
* Memerlukan data pelatihan yang relatif sedikit.
* Berkinerja baik bahkan jika asumsi independensi tidak sepenuhnya terpenuhi.

#### **Kekurangan**
* Asumsi independensi yang "naif" adalah kelemahan utamanya, karena fitur di dunia nyata seringkali saling terkait.
* Jika ada kategori dalam data uji yang tidak ada dalam data latih, model akan memberikan probabilitas nol dan gagal membuat prediksi (masalah *zero-frequency*).

## **Evaluation**
Metrik evaluasi yang digunakan untuk mengukur performa model adalah:

* **Accuracy (Akurasi)**: Mengukur proporsi prediksi yang benar dari keseluruhan jumlah data.
  

    $$Accuracy =  \frac{TP + TN}{TP + TN + FP + FN}$$
  
    *Keterangan: TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative*

* **Precision (Presisi)**: Mengukur seberapa banyak dari prediksi positif yang benar-benar positif.

   $$Precision =  \frac{TP}{TP + FP}$$

* **Recall (Sensitivitas)**: Mengukur seberapa banyak dari data positif yang sebenarnya berhasil diidentifikasi oleh model.

   $$Recall =  \frac{TP}{TP + FN}$$

* **F1-Score**: Rata-rata harmonik dari *precision* dan *recall*, memberikan keseimbangan antara keduanya.

   $$F1-Score = 2  \times \frac{Precision \times Recall}{Precision + Recall}$$
  
  

### **Hasil Metrik Evaluasi**

Berdasarkan hasil pengujian pada 114 data uji, performa dari masing-masing model dirangkum dalam tabel berikut. Kelas 0 adalah Jinak (*Benign*), dan Kelas 1 adalah Ganas (*Malignant*).

| Model               | Accuracy | Class 0 - F1-Score | Class 0 - Precision | Class 0 - Recall | Class 1 - F1-Score | Class 1 - Precision | Class 1 - Recall |
| :------------------ | :------- | :----------------- | :------------------ | :--------------- | :----------------- | :------------------ | :--------------- |
| **Random Forest (RF)** | **0.973684** | 0.977778           | 0.956522            | 1.000000         | 0.967742           | 1.000000            | 0.937500         |
| **Logistic Regression** | **0.973684** | 0.977444           | 0.970149            | 0.984848         | 0.968421           | 0.978723            | 0.958333         |
| **K-Nearest Neighbors (KNN)** | 0.964912 | 0.970588           | 0.942857            | 1.000000         | 0.956522           | 1.000000            | 0.916667         |
| **Support Vector Machine (SVM)** | 0.964912 | 0.970149           | 0.955882            | 0.984848         | 0.957447           | 0.978261            | 0.937500         |
| **Gradient Boosting Machine (GBM)** | 0.964912 | 0.970588           | 0.942857            | 1.000000         | 0.956522           | 1.000000            | 0.916667         |
| **Gaussian Naive Bayes** | 0.938596 | 0.947368           | 0.940299            | 0.954545         | 0.926316           | 0.936170            | 0.916667         |

---
---


### **Apakah solusi yang dikembangkan sudah menjawab setiap problem statement, berhasil mencapai seluruh goals yang diharapkan, dan memberikan dampak sesuai dengan solusi yang direncanakan? Jelaskan?**
- **Problem Statement**: Model berhasil menjawab tantangan untuk memprediksi diagnosis tumor (ganas atau jinak) dan secara kuantitatif menentukan algoritma klasifikasi mana yang paling optimal.
- **Goals Tercapai**: Ya, beberapa model klasifikasi berhasil dibangun dan dievaluasi secara komprehensif, dengan Random Forest serta Logistic Regression teridentifikasi sebagai yang paling efektif berdasarkan metrik evaluasi.
- **Dampak Solusi**: Penggunaan berbagai algoritma memungkinkan perbandingan objektif untuk menemukan solusi terbaik, sementara pemakaian metrik evaluasi yang beragam (accuracy, precision, dll.) memastikan model yang terpilih terbukti andal dan seimbang dalam kemampuannya mendeteksi kedua kelas.


## **Kesimpulan**
Dari proyek yang telah dilaksanakan, dapat disimpulkan bahwa *machine learning* dapat diterapkan secara efektif untuk memprediksi diagnosis kanker payudara dengan akurasi yang sangat tinggi. Di antara enam model yang dievaluasi, **Random Forest** dan **Logistic Regression** menjadi yang paling unggul dengan performa identik, mencapai akurasi sebesar 0.973684 pada data uji. Keduanya hanya salah mengklasifikasikan 3 dari 114 kasus uji. Tingkat akurasi yang tinggi ini menegaskan bahwa solusi prediktif berbasis data dapat menjadi aset berharga dalam bidang medis untuk mendukung deteksi dini kanker payudara.
