# Laporan Proyek Machine Learning - Leni Gustia
----
## Domain Proyek
### **Latar Belakang**
Penyakit jantung koroner, khususnya serangan jantung akut, masih menjadi penyebab utama kematian di dunia. Menurut [World Health Organization. (2021)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)), penyakit kardiovaskular menyebabkan sekitar 17,9 juta kematian setiap tahun atau setara dengan 31% dari seluruh kematian global. Kondisi ini mengindikasikan pentingnya deteksi dan penanganan dini, mengingat gejalanya sering tidak spesifik dan muncul secara mendadak.

Serangan jantung atau infark miokard terjadi ketika aliran darah ke otot jantung terhambat akibat penyumbatan arteri koroner, sehingga menyebabkan kerusakan jaringan jantung [Thygesen et al., (2019)](https://academic.oup.com/eurheartj/article/40/3/237/5079081?login=false). Gejala umum meliputi nyeri dada, sesak napas, dan rasa tidak nyaman di tubuh bagian atas. Biomarker seperti troponin dan CK-MB menjadi indikator utama dalam mendeteksi kerusakan otot jantung secara klinis [Collet et al., (2021)](https://academic.oup.com/eurheartj/article/42/14/1289/5898842?login=false). Selain itu, tekanan darah, detak jantung, dan kadar gula darah juga berperan penting dalam penilaian risiko serangan jantung.

Seiring berkembangnya teknologi di bidang data science dan kesehatan, pendekatan machine learning telah dimanfaatkan secara luas untuk membangun model prediktif dalam mendeteksi penyakit secara lebih akurat dan efisien. Berbagai studi menunjukkan keberhasilan pendekatan ini dalam konteks penyakit jantung.

Penelitian oleh [Deogire et al., (2024)](https://www.ijcaonline.org/archives/volume186/number21/prediction-of-risk-of-heart-attack-using-machine-learning-techniques/?utm_source)  menggunakan metode Naïve Bayes, K‑Nearest Neighbor (K-NN), dan Random Forest untuk memprediksi serangan jantung berdasarkan variabel klinis seperti tekanan darah, detak jantung, dan gula darah. Hasilnya menunjukkan bahwa algoritma K‑NN mencapai akurasi tertinggi sebesar 96,4%, sementara Random Forest dan Naïve Bayes tetap menunjukkan performa yang kompetitif.

Selanjutnya, studi yang dilakukan oleh [Srinivasan et al., (2023)](https://www.nature.com/articles/s41598-023-40717-1?utm_source) menggabungkan algoritma Decision Tree dan Naïve Bayes dalam analisis dataset penyakit kardiovaskular. Studi ini menekankan keunggulan interpretabilitas dari Decision Tree dalam memberikan aturan keputusan yang jelas bagi klinisi, serta kecepatan dan efisiensi Naïve Bayes dalam klasifikasi. Kombinasi keduanya mampu menghasilkan akurasi prediksi yang tinggi, serta mendukung pengambilan keputusan klinis secara cepat dan tepat.

Melengkapi temuan-temuan tersebut, studi komparatif oleh [Alariyibi et al., (2023)](https://arxiv.org/abs/2312.04595?utm_source=) membandingkan tiga model machine learning, yaitu J48 (Decision Tree), Random Forest, dan Naïve Bayes, dalam konteks prediksi penyakit jantung. Hasil penelitian menunjukkan bahwa Random Forest mencapai akurasi tertinggi sebesar 99,24%, diikuti oleh J48 dan Naïve Bayes. Hal ini menunjukkan bahwa model ensemble seperti Random Forest memiliki keunggulan dalam menangani kompleksitas dan variabilitas data medis.

Ketiga studi tersebut menunjukkan bahwa pendekatan berbasis machine learning, terutama dengan algoritma Decision Tree, Naïve Bayes, dan Random Forest, sangat efektif dalam mendeteksi risiko serangan jantung. Oleh karena itu, pemanfaatan data klinis seperti yang tersedia pada dataset Zheen Hospital di Erbil, Irak, dengan fitur seperti usia, tekanan darah, denyut jantung, glukosa darah, CK-MB, dan troponin, sangat relevan untuk dikembangkan lebih lanjut dalam membangun sistem prediktif yang akurat dan aplikatif di dunia medis.

## Business Understanding

#### Problem Statements
1. Bagaimana membangun sistem prediksi yang mampu mengestimasi kemungkinan seseorang mengalami serangan jantung dengan memanfaatkan data parameter kesehatan seperti usia, jenis kelamin, tekanan darah, kadar glukosa darah, serta biomarker jantung?
2. Seberapa efektif dan akurat algoritma pembelajaran mesin seperti Decision Tree, Naïve Bayes, dan Random Forest dalam mengidentifikasi risiko serangan jantung jika dibandingkan dengan metode diagnosis tradisional yang selama ini digunakan?

#### Goals
1. Mengembangkan model prediktif berbasis machine learning yang dapat memproyeksikan risiko serangan jantung dengan memanfaatkan kombinasi variabel klinis individu.
2. Melakukan optimasi model melalui tuning hyperparameter untuk meningkatkan presisi dan ketepatan dalam klasifikasi risiko serangan jantung.
3. Menilai dan membandingkan akurasi prediksi dari algoritma machine learning (Decision Tree, Naïve Bayes, dan Random Forest) dalam konteks deteksi dini penyakit jantung.

#### Solution Statement
1. Menerapkan berbagai algoritma machine learning, seperti Decision Tree, Naïve Bayes, dan Random Forest, dalam proses pelatihan model prediksi berbasis data kesehatan pasien guna mengidentifikasi individu yang berisiko tinggi terkena serangan jantung.
2. Melakukan evaluasi performa setiap model menggunakan metrik kuantitatif seperti akurasi, precision, recall, F1-score, dan confusion matrix, agar diperoleh pemahaman yang menyeluruh terhadap efektivitas prediksi masing-masing algoritma dalam konteks medis.


## Data Understanding

**Heart Attack Prediction Dataset** merupakan kumpulan data medis yang dikumpulkan dari Rumah Sakit Zheen di Erbil, Irak, selama periode Januari hingga Mei 2019. Dataset ini terdiri dari 1319 baris data dengan 9 fitur klinis utama yang digunakan untuk memprediksi kemungkinan terjadinya serangan jantung pada pasien. Label target pada data ini berupa nilai biner: 1 untuk pasien yang mengalami serangan jantung, dan 0 untuk yang tidak mengalaminya. Dataset ini telah dibersihkan dan tidak mengandung nilai hilang (missing value), sehingga siap digunakan untuk proses analisis dan pemodelan. **[Kaggle](https://www.kaggle.com/datasets/fatemehmohammadinia/heart-attack-dataset-tarik-a-rashid)**.

### Variabel pada Heart Attack Prediction Dataset:
- **Age**: Usia pasien pada saat pemeriksaan, dengan rentang antara 0 hingga 80 tahun. Usia merupakan salah satu faktor risiko penting dalam serangan jantung.
- **Gender**: Jenis kelamin biologis pasien, dikodekan sebagai 1 untuk laki-laki dan 0 untuk perempuan.
- **Heart Rate**: Jumlah detak jantung per menit (bpm), mencerminkan kondisi ritme dan fungsi jantung pasien.
- **Systolic Blood Pressure**: Tekanan darah saat jantung berkontraksi (sistolik), diukur dalam satuan mmHg.
- **Diastolic Blood Pressure**: Tekanan darah saat jantung berelaksasi antara dua kontraksi (diastolik), juga diukur dalam mmHg.
- **Blood Sugar**: Kadar glukosa darah pasien, dikodekan sebagai 1 jika >120 mg/dL dan 0 jika ≤120 mg/dL, digunakan sebagai indikator status glikemik.
- **CK-MB**: Enzim jantung yang dilepaskan ketika terjadi kerusakan pada otot jantung; menjadi indikator awal adanya infark miokard.
- **Troponin**: Biomarker protein yang sangat spesifik dan sensitif terhadap kerusakan otot jantung; dianggap sebagai “gold standard” dalam diagnosis serangan jantung.
- **Result**: Label hasil diagnosis, menunjukkan apakah pasien mengalami serangan jantung (1) atau tidak (0).

### Visualisasi Distribusi Data Numerik
![Visualisasi Data Numerik](/Gambar/Gambar_Numerik.png)

Distribusi usia (age) memuncak sekitar 60 tahun, menunjukkan dominasi pasien usia lanjut yang berisiko tinggi terkena serangan jantung. Jenis kelamin (gender) didominasi laki-laki (kode 1), yang cenderung lebih rentan terhadap penyakit jantung. Denyut jantung (heart rate) sebagian besar berada pada kisaran normal (60–120 bpm), namun terdapat outlier ekstrem yang kemungkinan merupakan kesalahan data. Tekanan darah sistolik (systolic blood pressure) terdistribusi simetris pada kisaran umum 100–140 mmHg, sedangkan tekanan diastolik (diastolic blood pressure) sedikit miring ke kanan, dominan pada 60–90 mmHg, dengan beberapa outlier. Kadar gula darah (blood sugar) sangat condong ke kanan, mayoritas di bawah 150 mg/dL, tetapi terdapat nilai ekstrem di atas 500 mg/dL. Nilai CK-MB dan troponin umumnya rendah, namun keduanya menunjukkan distribusi miring ke kanan, mengindikasikan sebagian kecil pasien mengalami peningkatan signifikan akibat kerusakan otot jantung.

### **Visualisasi Distribusi Kelas Target**
![Kategori](/Gambar/Gambar_Result.png)

Visualisasi ini menggambarkan distribusi dua kategori pada variabel Result. Terlihat bahwa jumlah individu dengan label positive (mengalami serangan jantung) lebih tinggi dibandingkan dengan label negative (tidak mengalami serangan jantung). Kelas positive mendominasi dengan frekuensi yang lebih besar, sedangkan kelas negative tercatat lebih sedikit. Ketimpangan ini menunjukkan adanya ketidakseimbangan kelas dalam dataset, yang merupakan hal umum pada kasus prediksi penyakit, di mana pasien yang mengalami kondisi tertentu (seperti serangan jantung) biasanya lebih sedikit dibandingkan yang tidak.



### **Visualisasi Kernel Density Estimation**
![KDE](/Gambar/Gambar_KDE.png)
Visualisasi KDE memperlihatkan pola hubungan antar fitur dalam dataset. Scatter plot di bawah diagonal menunjukkan sebaran data antar pasangan variabel, sedangkan diagonal menampilkan distribusi masing-masing variabel. Fitur seperti Tekanan Darah dan Denyut Jantung menunjukkan distribusi yang mendekati normal, menandakan pola yang stabil. Gula Darah dan Troponin memiliki distribusi yang terpusat, menunjukkan variasi nilai yang sempit. Sementara itu, variabel biner seperti Jenis Kelamin memperlihatkan distribusi yang terpisah dan tidak memiliki korelasi kuat dengan fitur lain.



### **Visualisasi Correlation Matrix**
![CM](/Gambar/Gambar_CM.png)

Hasil analisis matriks korelasi menunjukkan bahwa Systolic Blood Pressure dan Diastolic Blood Pressure memiliki korelasi cukup kuat sebesar 0,59, yang menandakan adanya keterkaitan erat antara kedua jenis tekanan darah tersebut. Sebaliknya, Age hanya menunjukkan korelasi yang sangat lemah dengan Troponin (0,09) dan hampir tidak berkorelasi dengan Systolic Blood Pressure (0,02), mengindikasikan bahwa usia tidak berpengaruh besar terhadap indikator jantung tertentu dalam dataset ini. Sementara itu, Blood Sugar tampak berdiri sendiri karena memiliki korelasi yang sangat rendah dengan semua fitur lainnya. Korelasi antara Heart Rate dan Diastolic Blood Pressure juga rendah (0,11), namun keduanya tetap relevan sebagai indikator kesehatan jantung. Selain itu, nilai CK-MB tidak menunjukkan hubungan yang kuat dengan fitur lainnya, mencerminkan bahwa enzim ini bervariasi secara independen dalam data.

## Data Preparation

- **`Handling Outlier`** : Nilai-nilai pencilan (outlier) pada fitur numerik diidentifikasi menggunakan metode Interquartile Range (IQR). Setelah ditemukan, nilai-nilai yang terlalu jauh dari rentang normal tidak dihapus, melainkan dibatasi menggunakan teknik clipping, yaitu membatasi nilai agar tidak melebihi ambang tertentu. Tujuannya agar data tetap utuh namun tidak memberikan pengaruh berlebihan terhadap model.
  **Alasan Penerapan** : Outlier dapat memicu bias estimasi, terutama pada model yang sensitif seperti Decision Tree. Dengan membatasi nilai-nilai tersebut, proses pelatihan menjadi lebih akurat dan tidak terdistorsi oleh data yang tidak representatif.
```python
# Variabel numerik
variabel_numerik = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure',
                    'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']

# Salin data asli sebelum pemangkasan outlier
data_sebelum_clip = data_leni[variabel_numerik].copy()
daftar_outlier = {}

# Clipping outlier dengan metode IQR
for fitur in variabel_numerik:
    kuartil_1 = data_leni[fitur].quantile(0.25)
    kuartil_3 = data_leni[fitur].quantile(0.75)
    iqr = kuartil_3 - kuartil_1
    lower = kuartil_1 - 1.5 * iqr
    upper = kuartil_3 + 1.5 * iqr

    # Simpan data outlier
    daftar_outlier[fitur] = data_leni.loc[(data_leni[fitur] < lower) | (data_leni[fitur] > upper), fitur]

    # Lakukan clipping
    data_leni[fitur] = np.clip(data_leni[fitur], lower, upper)
```

- **`Standarisasi`** : Proses standarisasi dilakukan pada fitur numerik menggunakan StandardScaler untuk memastikan setiap nilai fitur berada dalam skala yang seragam. Hal ini sangat penting karena perbedaan skala antar fitur dapat memengaruhi kinerja model pembelajaran mesin, terutama algoritma yang sensitif terhadap nilai numerik, seperti model berbasis jarak maupun gradient-based. Dengan standarisasi, proses optimasi dapat berjalan lebih cepat dan stabil, serta mencegah dominasi fitur tertentu yang memiliki rentang nilai jauh lebih besar dibanding fitur lainnya.
  **Alasan Penerapan**: Penyeragaman skala diperlukan agar model tidak berat sebelah terhadap fitur dengan skala besar. Meskipun model seperti Decision Tree dan Random Forest relatif tahan terhadap skala, penerapan standarisasi tetap bermanfaat untuk menjaga konsistensi preprocessing dan mendukung eksperimen dengan berbagai jenis algoritma lain yang lebih sensitif terhadap skala data.
```python
# Variabel numerik pada data_leni
fitur_numerik = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure',
                 'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']

# Standarisasi fitur numerik menggunakan StandardScaler
scaler = StandardScaler()
data_leni[fitur_numerik] = scaler.fit_transform(data_leni[fitur_numerik])
```
- **`Spliting Data`** : Dataset dibagi menjadi dua subset, yakni 80% digunakan untuk pelatihan model dan 20% sisanya untuk pengujian. Kolom Result dijadikan sebagai label target yang diprediksi dalam proses ini.
**Alasan Penerapan**:Pemisahan ini penting untuk menilai performa model terhadap data yang tidak pernah dilihat sebelumnya. Hal ini membantu menguji kemampuan generalisasi model serta meminimalkan risiko overfitting. Dengan demikian, model dapat diuji seolah-olah sedang diterapkan pada data nyata di dunia luar.
```python
# Pisahkan fitur dan target
X = data_leni.drop(columns=["Result"])
y = data_leni["Result"]

# Split data menjadi train dan test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```
  
## Modeling

Dalam proyek ini, digunakan tiga algoritma klasifikasi untuk memprediksi kemungkinan terjadinya serangan jantung berdasarkan atribut medis yang tersedia, yakni Decision Tree, Random Forest, dan Naive Bayes. Ketiga model ini dipilih karena memiliki karakteristik yang sesuai untuk pengolahan data kesehatan dan klasifikasi berbasis data riil pasien. Penjelasan masing-masing model sebagai berikut:

- **Decision Tree**:  Merupakan algoritma yang menyusun struktur pohon keputusan berdasarkan pembagian data secara bertahap dari atribut yang paling berpengaruh, menggunakan ukuran seperti Gini Impurity atau Entropy. Setiap cabang mencerminkan keputusan atas suatu fitur, sementara daun pohon menunjukkan hasil klasifikasi. Model ini mudah dipahami dan cocok untuk mengidentifikasi pola non-linear. Namun, apabila struktur pohonnya terlalu kompleks, model ini rentan terhadap overfitting. Untuk mengatasinya, dapat digunakan teknik pemangkasan (pruning) atau pembatasan kedalaman pohon.
- **Random Forest**: Merupakan teknik ensemble learning yang membentuk kumpulan pohon keputusan dari data dan fitur yang dipilih secara acak. Hasil prediksi diperoleh melalui mekanisme suara terbanyak (majority voting). Keunggulan Random Forest terletak pada kestabilannya serta kemampuannya mengurangi risiko overfitting dibandingkan Decision Tree tunggal. Meski begitu, metode ini membutuhkan sumber daya komputasi yang lebih besar karena jumlah model yang dilibatkan cukup banyak.
- **Naive Bayes**: Berdasarkan pada Teorema Bayes, model ini menghitung probabilitas setiap kelas berdasarkan fitur yang tersedia, dengan anggapan bahwa fitur-fitur tersebut saling bebas (independen). Model ini dikenal karena kesederhanaan dan kecepatannya dalam mengolah data berskala besar. Walaupun asumsi independensi sering kali tidak sepenuhnya sesuai dengan kenyataan, Naive Bayes tetap efektif untuk klasifikasi dasar dan kasus medis dengan kompleksitas rendah.

##### Tahapan yang Dilakukan dalam Proses Pemodelan:

1.  **`Load model`**
- **Decision Tree** dimuat dengan parameter `random_state=42`
```python
# Inisialisasi model Decision Tree
Model_Decision_Tree = DecisionTreeClassifier(random_state=42)
```

- **Random Forest** dimuat dengan parameter `n_estimator= 100` dan `random_state=42`
```python
# Membangun dan melatih model Random Forest
Model_Random_Forest = RandomForestClassifier(n_estimators=100, random_state=42)
```

- **Naive Bayes**   :
```python
# Membangun dan melatih model Naive Bayes
Model_Naive_Bayes = GaussianNB()
```

2.  **`Pelatihan Model`**:

- **Decision Tree** dilatih dengan menggunakan data pelatihan `X_train dan y_train` 
```python
Model_Decision_Tree.fit(X_train, y_train)
```
- **Random Forest** dilatih dengan menggunakan data pelatihan `X_train dan y_train` 
```python
Model_Random_Forest.fit(X_train, y_train)
```

- **Naive Bayes** dilatih dengan menggunakan data pelatihan `X_train dan y_train` 
```python
Model_Naive_Bayes.fit(X_train, y_train)
```

3.  **`Evaluasi`**:
    Ketiga model yang telah dilatih dievaluasi dan dibandingkan menggunakan berbagai metrik pengukuran, seperti akurasi, presisi, recall, dan F1-score, guna menentukan model yang memberikan performa terbaik. Berdasarkan hasil evaluasi tersebut, Random Forest menunjukkan kinerja prediksi yang paling unggul dibandingkan Decision Tree dan Naive Bayes. Oleh karena itu, model ini dipilih sebagai model terbaik karena mampu memberikan hasil prediksi yang lebih akurat dalam mengidentifikasi risiko serangan jantung pada pasien.

### Evaluation
**Evaluasi model** dilakukan dengan memanfaatkan sejumlah metrik utama yang sesuai untuk klasifikasi biner, seperti **Accuracy**, **Precision**, **Recall**, **F1-Score**, dan **Confusion Matrix**. Metrik-metrik ini dipilih karena konteks data yang berkaitan dengan prediksi kondisi kesehatan (kemungkinan serangan jantung), sehingga penting untuk memperhatikan keseimbangan antara identifikasi kasus positif dan negatif. Tujuan dari evaluasi ini adalah untuk memastikan bahwa model tidak hanya akurat, tetapi juga mampu meminimalkan kesalahan dalam mengidentifikasi pasien yang berpotensi berisiko tinggi.

**Metrik Evaluasi yang Digunakan**
1.  **`Accuracy Score`** :
   - **Accuracy**: Menunjukkan proporsi prediksi yang tepat dibandingkan dengan seluruh jumlah data. Metrik ini memberikan pandangan umum terhadap performa model, namun dalam situasi tertentu seperti distribusi kelas yang tidak seimbang, akurasi bisa memberikan kesan yang menyesatkan.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$
- Untuk Decision Tree
```python
print(f"Akurasi Decision Tree: {accuracy_score(y_test, y_pred_dt):.4f}")
```
- Untuk Random Forest
```python
print(f"Akurasi Random Forest: {accuracy_score(y_test, y_pred_rf):.4f}")
```
- Untuk Naive Bayes
```python
print(f"Akurasi Naive Bayes: {accuracy_score(y_test, y_pred_nb):.4f}")
```
2. **`Classification Report`** :

a. **Precision**: Menggambarkan proporsi prediksi positif yang terbukti benar. Dalam konteks ini, precision menunjukkan seberapa akurat model dalam mengidentifikasi pasien yang benar-benar mengalami serangan jantung dari seluruh pasien yang diprediksi positif. Nilai precision yang tinggi berarti model jarang memberikan hasil positif palsu (false positive), sehingga mengurangi kemungkinan kesalahan dalam mendiagnosis seseorang terkena serangan jantung.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

b. **Recall (Sensitivity)**: Menunjukkan sejauh mana model mampu mengenali seluruh kasus positif yang ada. Dalam kasus medis, metrik ini sangat krusial karena semakin tinggi recall, semakin sedikit kasus serangan jantung yang terlewatkan. Kegagalan mendeteksi pasien yang benar-benar sakit (false negative) dapat membawa konsekuensi serius, sehingga recall menjadi prioritas utama.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

c. **F1-Score**: Merupakan rata-rata harmonis dari precision dan recall, yang memberikan ukuran kinerja model secara menyeluruh, khususnya saat distribusi kelas tidak seimbang. F1-score membantu mengevaluasi sejauh mana model menjaga keseimbangan antara kemampuan mengidentifikasi kasus positif secara benar dan menghindari kesalahan klasifikasi.

$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
- Untuk Decision Tree
```python
print("\nClassification Report Decision Tree:\n", classification_report(y_test, y_pred_dt))
```
- Untuk Random Forest
```python
print("\nClassification Report Random Forest:\n", classification_report(y_test, y_pred_rf))
```
- Untuk Naive Bayes
```python
print("\nClassification Report Naive Bayes:\n", classification_report(y_test, y_pred_nb))
```

3. **`Confusion Matrix`** : Matriks ini menyajikan gambaran detail tentang hasil prediksi model dengan membagi output ke dalam empat kategori: true positive, true negative, false positive, dan false negative. Alat ini sangat berguna untuk memahami pola kesalahan model, seperti apakah model lebih sering melewatkan kasus serangan jantung yang sebenarnya terjadi, atau justru salah mengklasifikasikan individu sehat sebagai penderita serangan jantung.

   |                    | Predicted Negatif (0) | Predicted Positif (1) |
   | ------------------ | --------------------- | --------------------- |
   | Actual Negatif (0) | True Negative (TN)    | False Positive (FP)   |
   | Actual Positif (1) | False Negative (FN)   | True Positive (TP)    |

- Untuk Decision Tree
 ```python
   cm_dt = confusion_matrix(y_test, y_pred_dt)
   ```
- Untuk Random Forest
```python
cm_rf = confusion_matrix(y_test, y_pred_rf)
```
- Untuk Naive Bayes
```python
cm_nb = confusion_matrix(y_test, y_pred_nb)
```

#### Berikut adalah ringkasan hasil evaluasi berdasarkan prediksi pada data :

 **1. Accuracy dan Classification Report :**

| Model         | Accuracy | Precision | Recall | F1-Score |
| ------------- | -------- | --------- | ------ | -------- |
| Decision Tree | 0.9773   | 0.98      | 0.98   | 0.98     |
| Random Forest | 0.9811   | 0.98      | 0.98   | 0.98     |
| Naive Bayes   | 0.8712   | 0.86      | 0.88   | 0.87     |


**Analisis Hasil:**
- **Akurasi**:  
  Ketiga model menghasilkan tingkat akurasi yang tinggi. **Random Forest** mencatat akurasi tertinggi sebesar 0.9811, disusul oleh **Decision Tree** dengan 0.9773, dan **Naive Bayes** dengan 0.8712. Meskipun selisihnya kecil antara dua model teratas, **Random Forest** sedikit lebih unggul, sementara **Naive Bayes** tertinggal cukup jauh dalam hal akurasi.

- **Presisi**:  
  Semua model menunjukkan nilai **precision** yang baik, terutama **Random Forest** dan **Decision Tree** yang keduanya berada pada angka 0.98. **Naive Bayes** memiliki precision sebesar 0.86. Hal ini menunjukkan bahwa model cukup andal dalam meminimalkan kesalahan prediksi positif palsu.

- **Recall**:  
  Model **Random Forest** memperoleh nilai recall tertinggi sebesar 0.98, diikuti oleh **Decision Tree** dengan 0.98, dan **Naive Bayes** dengan 0.88 (dari macro average). Ini menunjukkan bahwa Random Forest dan Decision Tree mampu mendeteksi lebih banyak kasus positif, yang sangat penting dalam konteks medis seperti prediksi serangan jantung.

- **F1-Score**:  
  Metrik **F1-Score**, yang menggabungkan precision dan recall, mencerminkan performa seimbang dari model. **Random Forest** dan **Decision Tree** masing-masing mencapai F1-score 0.98, sedangkan **Naive Bayes** mencatat F1-score 0.87. Artinya, Naive Bayes masih cukup baik, namun tidak seoptimal dua model lainnya dalam menyeimbangkan antara menghindari false positives dan menangkap kasus positif.

### **Visualisasi Hasil 3 Klasifikasi**

![Perbandingan](/Gambar/Gambar_Perbandingan.png)

Berdasarkan hasil evaluasi model, **Random Forest** dipilih sebagai model terbaik meskipun **Decision Tree** juga menunjukkan performa yang sangat baik dalam menyeimbangkan precision dan recall. Random Forest unggul dengan akurasi tertinggi sebesar **0.9811**, serta nilai **precision dan recall** yang tinggi (**0.98 dan 0.98**), menandakan kemampuannya dalam mengidentifikasi kasus positif maupun negatif secara akurat dan konsisten. Meskipun Decision Tree memiliki akurasi yang hampir sama (**0.9773**) dan F1-score identik (**0.98**), Random Forest tetap menunjukkan sedikit keunggulan dari sisi stabilitas dan generalisasi. Hal ini menjadikannya lebih ideal dalam konteks medis, terutama untuk kebutuhan deteksi dini dan pengambilan keputusan yang cepat. Kemampuannya dalam menangani variabilitas data serta menghasilkan prediksi yang andal menjadikan **Random Forest** sebagai pilihan yang paling sesuai untuk memprediksi risiko serangan jantung.

### **2. Analisis Berdasarkan Confusion Matrix**

| Model         | Actual               | Predicted Negatif (0) | Predicted Positif (1) |
|---------------|----------------------|------------------------|------------------------|
| Decision Tree | Actual Negatif (0)   | 98                     | 3                      |
| Decision Tree | Actual Positif (1)   | 3                      | 160                    |
| Random Forest | Actual Negatif (0)   | 98                     | 3                      |
| Random Forest | Actual Positif (1)   | 2                      | 161                    |
| Naive Bayes   | Actual Negatif (0)   | 94                     | 7                      |
| Naive Bayes   | Actual Positif (1)   | 27                     | 136                    |


Berdasarkan hasil dari confusion matrix untuk ketiga model, berikut adalah poin-poin penting yang dapat disimpulkan:

### 1. Kinerja dalam Mengklasifikasikan Kelas Positif (1)
Kemampuan model dalam mengenali pasien yang benar-benar mengalami serangan jantung (positif) memperlihatkan perbedaan mencolok:

- **Random Forest** unggul dengan **161 True Positive (TP)** dan hanya **2 False Negative (FN)**, menjadikannya model paling sensitif dalam mendeteksi kasus positif.
- **Decision Tree** juga cukup baik, dengan **160 TP** dan **3 FN**, menunjukkan tingkat deteksi positif yang hampir setara dengan Random Forest.
- Sebaliknya, **Naive Bayes** menunjukkan kelemahan signifikan, dengan hanya **136 TP** dan **27 FN**, sehingga berisiko tinggi melewatkan pasien yang benar-benar berisiko terkena serangan jantung.

### 2. Kinerja dalam Mengklasifikasikan Kelas Negatif (0)
Ketiga model menunjukkan performa yang sangat baik dalam mengidentifikasi kasus negatif (tidak mengalami serangan jantung), yang tercermin dari tingginya nilai **True Negative (TN)** dan rendahnya **False Positive (FP)**.

- **Random Forest** dan **Decision Tree** masing-masing berhasil mengklasifikasikan **98 kasus negatif** dengan benar dan hanya melakukan **3 kesalahan klasifikasi (FP)**.
- **Naive Bayes** sedikit tertinggal, dengan **94 TN** dan **7 FP**, yang menunjukkan kecenderungan lebih besar dalam memberikan prediksi positif yang keliru pada pasien sehat.


### 3. Keunggulan Random Forest

Berdasarkan evaluasi performa secara keseluruhan, **Random Forest** muncul sebagai model terbaik. Berikut beberapa alasan yang mendukung hal tersebut:
#### a. Stabilitas Model
Sebagai model **ensemble**, Random Forest menggabungkan beberapa pohon keputusan sehingga lebih tahan terhadap overfitting, berbeda dengan Decision Tree tunggal yang rawan bias terhadap data latih.
#### b. Minim Risiko False Negative
Dengan hanya **2 kasus False Negative**, Random Forest memiliki potensi deteksi yang sangat tinggi terhadap pasien yang benar-benar sakit—hal yang sangat krusial dalam konteks medis.
#### c. Akurasi Tinggi
Random Forest mencatat **akurasi tertinggi sebesar 98.11%**, menandakan kemampuannya dalam mengklasifikasikan pasien secara akurat, baik pada kelas negatif maupun positif.
#### d. Keseimbangan Precision dan Recall
Dengan nilai **precision sebesar 0.98** dan **recall 0.99**, Random Forest menjaga keseimbangan dalam mendeteksi kasus positif dengan tingkat kesalahan minimum.
#### e. Kinerja Konsisten pada Data Kompleks
Random Forest dapat menangani fitur-fitur medis yang saling berkorelasi dan memiliki struktur data yang kompleks, menjadikannya cocok untuk prediksi serangan jantung yang melibatkan banyak parameter klinis.

---

### Kesimpulan

Secara menyeluruh, **Random Forest** menggabungkan **akurasi tinggi**, **kemampuan generalisasi**, dan **ketahanan terhadap overfitting**, menjadikannya pilihan terbaik untuk aplikasi medis seperti prediksi serangan jantung. Model ini memberikan prediksi yang sangat akurat dan dapat diandalkan, yang krusial dalam pengambilan keputusan klinis secara dini dan tepat.
