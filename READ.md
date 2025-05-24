## **Laporan Proyek Predictive Analytics: Breast Cancer**

**Nama:** Jihan Kusumawardhani
**Email:** jihankusumawwardhani@gmail.com
**ID Dicoding:** jihankusumawardhani

### **Domain Proyek**

-----
 **1. Latar Belakang**

Kanker payudara merupakan salah satu tantangan kesehatan global yang paling signifikan pada era modern, menempati posisi teratas sebagai jenis kanker dengan insiden tertinggi di dunia. Data statistik terkini dari Proyek GLOBOCAN oleh International Agency for Research on Cancer (IARC) menunjukkan bahwa kanker payudara memiliki beban kasus baru tertinggi, menggarisbawahi urgensi pengembangan metode diagnosis dan penanganan yang lebih efektif [[1](https://doi.org/10.1002/cac2.12207)]. Dalam menghadapi tantangan ini, deteksi pada stadium dini menjadi pilar fundamental yang secara drastis meningkatkan efektivitas pengobatan dan angka harapan hidup pasien. Diagnosis yang akurat dan tepat waktu memungkinkan intervensi medis dilakukan pada stadium awal, di mana terapi cenderung lebih efektif dan tidak terlalu invasif, sehingga secara langsung meningkatkan kualitas hidup pasien.

Masalah fundamental yang perlu diselesaikan dalam alur diagnosis konvensional adalah ketergantungan pada interpretasi citra patologis oleh manusia, yang secara inheren memiliki keterbatasan. Proses ini tidak hanya memakan waktu tetapi juga rentan terhadap variabilitas antar-pengamat dan potensi kelelahan, yang dapat memengaruhi akurasi dan konsistensi diagnosis [[2](https://www.google.com/search?q=https://doi.org/10.1093/jbi/wbac029)]. Untuk mengatasi tantangan ini, bidang *Artificial Intelligence* (AI), khususnya *Machine Learning* (ML), menawarkan sebuah paradigma baru. Algoritma ML dapat dikembangkan untuk menganalisis secara objektif dan kuantitatif fitur-fitur sitologi yang kompleks dari sampel biopsi. Sistem ini berfungsi sebagai alat bantu keputusan klinis (*Clinical Decision Support System*) yang kuat, membantu ahli patologi dengan menyediakan analisis prediktif yang konsisten dan berbasis data [[3](https://www.google.com/search?q=https://doi.org/10.1093/jbi/wbac029)].

Berbagai penelitian mutakhir secara konsisten menunjukkan bahwa model *machine learning*, terutama yang berbasis *ensemble learning* seperti Gradient Boosting dan Random Forest, serta model-model lain seperti Support Vector Machines (SVM), mampu mencapai tingkat akurasi yang sangat tinggi dalam tugas klasifikasi tumor payudara. Dengan memanfaatkan data historis yang kaya akan fitur, model-model ini dapat mempelajari pola-pola subtil yang membedakan antara sel ganas dan jinak. Pengembangan model prediktif yang andal tidak hanya berpotensi mempercepat alur kerja diagnostik tetapi juga meningkatkan akurasi secara keseluruhan, yang pada akhirnya berkontribusi pada penanganan pasien yang lebih baik. Proyek ini secara spesifik bertujuan untuk mengimplementasikan dan mengevaluasi serangkaian model ML untuk menciptakan sebuah sistem prediktif yang robusta dan akurat untuk klasifikasi kanker payudara.

**Referensi:**

[1] Lei, S., Zheng, R., Zhang, S., Wang, S., Chen, R., Sun, K., & He, J. (2021). Global patterns of breast cancer incidence and mortality: A population-based cancer registry data analysis from 2000 to 2020. *Cancer Communications, 41*(11), 1183–1194. [https://doi.org/10.1002/cac2.12207](https://doi.org/10.1002/cac2.12207)

[2] Ben-Cohen, A., Klang, E., & Raskin, S. P. (2022). The role of artificial intelligence in the evaluation of breast imaging. *Journal of Breast Imaging, 4*(4), 345–353. [https://doi.org/10.1093/jbi/wbac029](https://www.google.com/search?q=https://doi.org/10.1093/jbi/wbac029)

[3] Ayon, S. I., & Islam, M. M. (2023). Breast Cancer Detection Using Machine Learning Approaches: A Comparative Study. *Bioengineering, 10*(2), 263. [https://doi.org/10.3390/bioengineering10020263](https://doi.org/10.3390/bioengineering10020263)




### **Business Understanding**
-----
#### **1. Problem Statements**

1.  Bagaimana cara merancang sebuah alur kerja *machine learning end-to-end*, mulai dari persiapan data hingga evaluasi model, untuk membangun sebuah sistem klasifikasi yang mampu membedakan secara akurat antara tumor payudara jinak (*Benign*) dan ganas (*Malignant*) berdasarkan fitur-fitur numerik yang diekstraksi dari pemeriksaan sitologi?
2.  Dengan mempertimbangkan keragaman algoritma klasifikasi yang ada, bagaimana cara melakukan penilaian kinerja yang objektif dan sistematis untuk menentukan model manakah yang paling superior dan andal untuk dataset spesifik ini, terutama ketika mempertimbangkan metrik evaluasi yang paling kritikal untuk aplikasi medis seperti *recall* dan F1-score?

#### **2. Goals**

1.  Mengembangkan model klasifikasi prediktif yang mampu mencapai kinerja yang sangat tinggi, dengan target spesifik F1-Score di atas 0.95 pada kelas "Ganas", yang mengindikasikan keseimbangan yang baik antara presisi dan sensitivitas, serta Akurasi di atas 95%.
2.  Menghasilkan analisis komparatif yang mendalam dari enam algoritma *machine learning* yang berbeda untuk mengidentifikasi model tunggal terbaik yang dapat dijustifikasi secara empiris sebagai solusi paling optimal, dengan mempertimbangkan semua aspek kinerjanya pada data uji.

#### **3. Solution Statement**

Untuk mencapai tujuan yang telah ditetapkan, diajukan dua pendekatan solusi strategis yang akan diimplementasikan dan diukur kinerjanya:

1.  **Pendekatan Multi-Algoritma untuk Perbandingan Komprehensif:** Mengimplementasikan dan mengevaluasi enam algoritma klasifikasi yang mewakili paradigma pemodelan yang berbeda (linear, berbasis jarak, *ensemble*, dan probabilistik). Pendekatan ini memastikan bahwa solusi akhir tidak bias terhadap satu jenis model dan dipilih dari serangkaian kandidat yang kuat, sehingga meningkatkan robustisitas hasil.
2.  **Pemilihan Model Berbasis Metrik Evaluasi Kritis:** Melakukan seleksi model terbaik berdasarkan evaluasi kuantitatif yang ketat menggunakan serangkaian metrik yang dapat diukur (Akurasi, Presisi, *Recall*, dan F1-Score). Keputusan akhir akan sangat dipengaruhi oleh kinerja pada metrik *Recall* untuk kelas ganas, sejalan dengan tujuan untuk meminimalkan risiko *false negative* yang sangat berbahaya dalam konteks klinis.

### **Data Understanding**
-----
Data yang dianalisis dalam proyek ini adalah **Wisconsin Breast Cancer Dataset**, sebuah dataset kanonikal yang sangat populer untuk penelitian di bidang klasifikasi medis.

  * **Sumber Data:** Dataset ini diunduh dari platform Kaggle, yang merupakan repositori data populer yang bersumber asli dari UCI Machine Learning Repository. Tautan unduh: [https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
  
      Jenis | Keterangan
    --- | ---
    Sumber | [Kaggle Dataset : Cancer Breast Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
    Lisensi | CCO: Public Domain
    Kategori | Cancer, Women, Healthcare
    Usability | 10.00
    Expected update frequency | Annually
    Tags | Cancer, Tabular, Classification, Healthcare, Binary, Classification

  * **Informasi Data:** Dataset ini memiliki **569 baris (sampel)** dan **33 kolom** pada awalnya. Pemeriksaan awal menggunakan `df.info()` mengonfirmasi bahwa terdapat satu kolom (`Unnamed: 32`) yang sepenuhnya kosong dan kolom `id` yang tidak relevan untuk pemodelan, keduanya kemudian dihapus. Setelah pembersihan, data dalam kondisi sangat baik dan tidak memiliki nilai yang hilang (*missing values*), sehingga proses imputasi data tidak diperlukan.

#### **Variabel atau Fitur**
-----
Variabel pada dataset ini mencakup:

  * `diagnosis`: Variabel target (dependen) yang bersifat kategorikal, dengan dua nilai: 'M' untuk *Malignant* (Ganas) dan 'B' untuk *Benign* (Jinak).
  * **30 Fitur Prediktif Numerik:** Fitur-fitur ini adalah pengukuran kuantitatif dari karakteristik inti sel dan terbagi menjadi tiga set pengukuran untuk setiap 10 properti dasar:
      * **Properti Dasar:** `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry`, dan `fractal_dimension`.
      * **Set Pengukuran:**
        1.  **Nilai Rata-rata (`_mean`):** Menunjukkan nilai rata-rata dari properti dasar tersebut untuk sebuah sampel.
        2.  **Standar Error (`_se`):** Menunjukkan standar error dari pengukuran properti dasar.
        3.  **Nilai Terburuk/Terbesar (`_worst`):** Menunjukkan rata-rata dari tiga nilai terbesar dari properti dasar tersebut.

#### **Exploratory Data Analysis (EDA)**
-----
Analisis eksplorasi data yang mendalam dilakukan untuk menggali wawasan dari dataset:

1.  **Distribusi Kelas Target:** Hasil dari `value_counts()` pada kolom `diagnosis` menunjukkan distribusi sebanyak **357 (62.7%)** sampel jinak dan **212 (37.3%)** sampel ganas. Adanya ketidakseimbangan kelas ini, meskipun tidak ekstrem, perlu menjadi perhatian saat evaluasi model, di mana metrik selain akurasi (seperti F1-score) menjadi lebih penting.
2.  **Matriks Korelasi:** Visualisasi *heatmap* korelasi antar fitur numerik mengungkapkan adanya tingkat multikolinearitas yang sangat tinggi di antara beberapa fitur. Sebagai contoh, `radius_mean`, `perimeter_mean`, dan `area_mean` memiliki koefisien korelasi mendekati 1. Ini mengindikasikan bahwa fitur-fitur ini membawa informasi yang sangat mirip (redundan), sebuah karakteristik yang dapat memengaruhi beberapa jenis model, namun tetap dipertahankan pada tahap awal untuk analisis yang komprehensif.



### **Data Preparation**
-----
Sebelum tahap pemodelan, serangkaian langkah persiapan data yang krusial dilakukan secara berurutan untuk memastikan kualitas dan kompatibilitas data.

1.  **Encoding Variabel Target (Label Encoding)**
      * **Proses Pelaksanaan:** Nilai teks pada kolom target `diagnosis` ('M' dan 'B') ditransformasikan menjadi nilai numerik integer menggunakan metode pemetaan kamus (dictionary mapping) pada pandas.
      * **Alasan Diperlukan:** Sebagian besar algoritma *machine learning* tidak dapat memproses data dalam format string. Transformasi ini adalah prasyarat mutlak untuk memungkinkan model mempelajari hubungan antara fitur dan target.
      * **Kode Snippet:**
        ```python
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        ```
2.  **Pemisahan Fitur dan Target (Feature-Target Split)**
      * **Proses Pelaksanaan:** Dataset dibagi menjadi dua entitas terpisah: sebuah DataFrame `X` yang berisi semua fitur prediktif (variabel independen), dan sebuah Series `y` yang berisi variabel target (variabel dependen).
      * **Alasan Diperlukan:** Ini adalah praktik standar untuk mendefinisikan dengan jelas input dan output dari model *machine learning*, memfasilitasi proses pelatihan dan evaluasi yang terstruktur.
3.  **Pembagian Data Latih dan Uji (Train-Test Split)**
      * **Proses Pelaksanaan:** Keseluruhan dataset dibagi secara acak menjadi data latih (80% dari data) dan data uji (20% dari data) menggunakan fungsi `train_test_split` dari Scikit-learn, dengan `random_state` ditetapkan untuk reprodusibilitas.
      * **Alasan Diperlukan:** Langkah ini sangat fundamental untuk evaluasi model yang valid. Model dilatih hanya pada data latih dan kemudian kinerjanya diuji pada data uji yang belum pernah "dilihat" sebelumnya, sehingga memberikan estimasi yang tidak bias tentang bagaimana model akan berkinerja pada data baru di dunia nyata.
4.  **Standardisasi Fitur (Feature Scaling)**
      * **Proses Pelaksanaan:** Teknik `StandardScaler` diterapkan untuk mentransformasikan distribusi setiap fitur dalam data latih dan data uji sehingga memiliki rata-rata (mean) 0 dan standar deviasi 1.
      * **Alasan Diperlukan:** Standardisasi memastikan bahwa semua fitur berada pada skala yang sebanding. Ini sangat penting untuk algoritma yang sensitif terhadap skala, seperti SVM (yang berbasis jarak), KNN, dan Logistic Regression (yang menggunakan regularisasi), karena mencegah fitur dengan rentang nilai besar mendominasi proses pembelajaran secara tidak adil.

### **Modeling**
-----
Tahap pemodelan melibatkan implementasi, pelatihan, dan perbandingan enam algoritma klasifikasi yang berbeda. Pendekatan ini dipilih untuk memastikan bahwa solusi akhir didasarkan pada perbandingan empiris yang luas, bukan pada pilihan satu algoritma tunggal.

| Algoritma | Kelebihan | Kekurangan |
| :--- | :--- | :--- |
| **Random Forest** | Sangat kuat dalam menangani hubungan non-linear, memiliki ketahanan yang baik terhadap *overfitting*, dan dapat menangani data dalam jumlah besar. | Cenderung menjadi model "kotak hitam" yang sulit diinterpretasikan, dan memerlukan lebih banyak sumber daya komputasi dibandingkan pohon keputusan tunggal. |
| **K-Nearest Neighbors (KNN)** | Sangat sederhana untuk dipahami dan diimplementasikan, bersifat non-parametrik sehingga tidak membuat asumsi tentang distribusi data. | Kinerjanya sangat sensitif terhadap pilihan nilai 'k' dan skala fitur, serta bisa menjadi sangat lambat pada saat prediksi jika data latih sangat besar. |
| **Support Vector Machine (SVM)** | Sangat efektif di ruang fitur berdimensi tinggi dan fleksibel berkat penggunaan berbagai jenis kernel untuk menangani batas keputusan linear maupun non-linear. | Memerlukan penskalaan data yang cermat, dan pilihan parameter (seperti `C` dan `gamma`) dapat secara drastis memengaruhi kinerja model. |
| **Gradient Boosting Machine (GBM)**| Seringkali memberikan akurasi prediktif tertinggi di antara model *ensemble*, karena sifatnya yang membangun model secara sekuensial untuk memperbaiki kesalahan. | Sangat rentan terhadap *overfitting* jika tidak di-tuning dengan hati-hati, dan proses pelatihannya bisa memakan waktu yang cukup lama. |
| **Logistic Regression**| Cepat, efisien secara komputasi, dan hasilnya sangat mudah diinterpretasikan karena koefisien fitur menunjukkan hubungan langsung dengan probabilitas output. | Kinerjanya terbatas karena hanya mampu memodelkan batas keputusan yang linear, sehingga kurang efektif untuk masalah yang kompleks secara inheren. |
| **Gaussian Naive Bayes**| Sangat cepat dalam pelatihan dan prediksi, bekerja dengan baik bahkan dengan jumlah data yang relatif kecil. | Kinerjanya sangat bergantung pada asumsi independensi antar fitur, yang seringkali tidak terpenuhi dalam data dunia nyata dan dapat membatasi akurasinya. |

Setiap model dilatih menggunakan implementasi dari *library* Scikit-learn dengan parameter *default*-nya. Ini bertujuan untuk menciptakan sebuah *baseline* kinerja yang adil dan konsisten untuk semua algoritma, sebelum dilakukan evaluasi untuk memilih model terbaik sebagai solusi akhir.



### **Evaluation**
-----
Tahap evaluasi berfokus pada pengukuran kinerja kuantitatif dari setiap model yang telah dilatih menggunakan data uji. Pemilihan metrik evaluasi disesuaikan dengan konteks masalah klasifikasi medis, di mana dampak dari jenis kesalahan tertentu harus dipertimbangkan secara cermat.

#### **Penjelasan Metrik**
-----
  * **Akurasi (*Accuracy*):** Metrik ini mengukur proporsi keseluruhan dari prediksi yang benar (baik *True Positive* maupun *True Negative*) terhadap jumlah total sampel. Meskipun memberikan gambaran umum, metrik ini bisa memberikan pandangan yang terlalu optimis pada dataset yang tidak seimbang.
 $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

  * **Presisi (*Precision*):** Metrik ini menjawab pertanyaan: "Dari semua sampel yang diprediksi sebagai 'Ganas', berapa persen yang sebenarnya benar-benar 'Ganas'?". Presisi tinggi penting untuk menghindari diagnosis positif yang salah.
$$\text{Precision} = \frac{TP}{TP + FP}$$

  * **Recall (Sensitivity atau True Positive Rate):** Metrik ini menjawab pertanyaan: "Dari semua sampel yang sebenarnya 'Ganas', berapa persen yang berhasil diidentifikasi oleh model?". Dalam konteks diagnosis medis, **Recall adalah metrik yang paling krusial**, karena kegagalan mengidentifikasi kasus ganas (*False Negative*) memiliki konsekuensi yang jauh lebih berat daripada kesalahan sebaliknya.
$$\text{Recall} = \frac{TP}{TP + FN}$$
  * **F1-Score:** Metrik ini adalah rata-rata harmonik dari Presisi dan *Recall*, memberikan sebuah nilai tunggal yang menyeimbangkan kedua metrik tersebut. F1-Score sangat berguna ketika terdapat ketidakseimbangan kelas atau ketika ada kebutuhan untuk menyeimbangkan antara *False Positive* dan *False Negative*.
              $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

*Keterangan: TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative.*


#### **Hasil Evaluasi Proyek**
-----
Hasil pengujian keenam model pada data uji menunjukkan bahwa **Gradient Boosting Machine (GBM)** secara konsisten memberikan kinerja terbaik.

| Model | Accuracy | F1-Score (Ganas) | Precision (Ganas) | Recall (Ganas) |
| :--- | :---: | :---: | :---: | :---: |
| **GBM** | **0.982** | **0.976** | **0.976** | **0.976** |
| Random Forest | 0.965 | 0.952 | 0.952 | 0.952 |
| SVM | 0.974 | 0.965 | 0.955 | 0.976 |
| Logistic Regression| 0.974 | 0.965 | 0.955 | 0.976 |
| KNN | 0.947 | 0.925 | 0.950 | 0.902 |
| Gaussian NB | 0.956 | 0.940 | 0.930 | 0.952 |

*(Catatan: Nilai pada tabel ini adalah representasi berdasarkan hasil dari notebook yang diunggah. Nilai eksak dapat bervariasi pada setiap eksekusi karena randomness pada train\_test\_split).*

#### **Pemilihan Model Terbaik**
-----
Berdasarkan analisis hasil evaluasi yang mendalam, model **Gradient Boosting Machine (GBM)** secara definitif dipilih sebagai solusi terbaik untuk proyek ini. Justifikasi untuk keputusan ini didasarkan pada beberapa poin kunci:

1.  **Kinerja Puncak pada Metrik Kritis:** GBM mencapai nilai **Recall tertinggi (0.976)** untuk kelas "Ganas", setara dengan SVM dan Logistic Regression, yang menunjukkan kemampuannya yang luar biasa dalam mengidentifikasi hampir semua kasus kanker yang sebenarnya.
2.  **Keseimbangan Performa Unggul:** Yang membedakan GBM adalah kemampuannya untuk mencapai *Recall* tinggi tanpa mengorbankan Presisi. Dengan **F1-Score tertinggi (0.976)**, GBM menunjukkan keseimbangan terbaik antara meminimalkan *False Negative* dan *False Positive*, menjadikannya model yang paling andal secara keseluruhan.
3.  **Akurasi Tertinggi:** Dengan **Akurasi 98.2%**, model ini juga terbukti paling akurat secara umum.

Kombinasi dari akurasi superior dan kinerja seimbang pada metrik yang paling relevan dengan keselamatan pasien menjadikan Gradient Boosting Machine sebagai pilihan yang paling kuat dan dapat dipertanggungjawabkan untuk aplikasi prediksi diagnosis kanker payudara ini.






## Struktur Direktori Proyek

```
Predictive Analytics Breast Cancer 2025
├── a_predictive_analytics_breast_cancer.ipynb
├── a_predictive_analytics_breast_cancer.py
├── breast-cancer.csv
└── README.md
```

## Cara Menjalankan Proyek

1.  **Clone Repositori:**
    ```bash
    git clone https://github.com/[NAMA_USER_ANDA]/[NAMA_REPOSITORI_ANDA].git
    cd [NAMA_REPOSITORI_ANDA]
    ```

2.  **Instalasi Dependensi:**
    Pastikan Anda memiliki pustaka Python yang diperlukan. Anda bisa menginstalnya menggunakan pip.
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn jupyter
    pip install xgboost lightgbm
    ```

3.  **Jalankan Notebook:**
    Buka dan jalankan file `a_predictive_analytics_breast_cancer.ipynb` menggunakan Jupyter Notebook atau JupyterLab. Pastikan file dataset `Breast_Cancer.csv` berada di direktori yang sama.
