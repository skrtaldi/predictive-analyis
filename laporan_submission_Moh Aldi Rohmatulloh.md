# Laporan Proyek Machine Learning - Moh. Aldi Rohmatulloh MC525D5Y0147

## Domain Proyek
Penyakit Jantung merupakan penyebab kematian nomor satu secara global. Menurut data dari WHO(2023) menunjukkan sekitar 17,9 juta jiwa meninggal setiap tahun akibat penyakit kardiovasklar, yang mana lebih dari 75% kematian ini terjadi di negara-negara berpendapatan menengah kebawah. Salah satu tantangan utama dalam penanganan penyakit jantung adalah seringkali diagnosi baru ditegakkan pada stadium lanjut, karena gejala awal yang tidak spesifik.

Seiring berkembangnya teknologi, pemanfaatan pendekatan berbasis data(data-driven) dan kecerdasan buatan(AI) mulai digunakan dalam dunia medis untuk mendukung proses diagnosa, termasuk prediksi risiko penyakit jantung. Deteksi dini penyakit jantung memiliki potensi besar untuk menyelamatkan nyawa pasien. Namun, ketersediaan fasisilitas diagnostik canggih tidak merata, terutama di fasilitas kesehatan dengan sumber daya terbatas. Dalam konteks ini, pendekatan berbasis Machine Learning menawarkan solusi deteksi awal yang relatif murah dan efisien.

Sistem prediksi Machine Learning dapat berfungsi sebagai second opinion yang berharga bagi dokter, sehingga berpotensi meningkatkan akurasi diagnosis secara keseluruhan. Identifikasi pasien pada tahap awal penyakit jantung tidak hanya meningkatkan peluang keberhasilan pengobatan, tetapi juga dapat menekan biaya pengobatan jangka panjang. Aspek ini sangat relevan dalam konteks sistem kesehatan nasional maupun asuransi kesehatan swasta, dimana efisiensi biaya dan aksebilitas layanan kesehtaan menjadi prioritas.

## Business Understanding
Setelah memahami latar belakang dan pentingnya permasalahan yang ingin diselesaikan, langkah selanjutnya adalah mengklarifikasi masalah secara lebih terstruktur. Klarifikasi ini bertujuan untuk merumuskan pernyataan masalah yang spesifik, menetapkan tujuan yang ingin dicapai, serta menyusun solusi yang memungkinkan untuk mengatasi permasalahan tersebut

### Problem Statements
- Bagaimana cara memprediksi kemungkinan seseorang mengidap penyakit jantung berdasarkan data mereka?
- Faktor apa saja yang paling berpengaruh terhadap terjadinya penyakit jantung?
- Bagaimana kita dapat membangun model prediktif yang akurat untuk mendeteksi risiko penyakit jantung secara otomatis?

### Goals
- Mendidentifikasi individu berisiko penyakit jantung dengan mampu mengklasifikasikan individu kedalam kelompok risiko tinggi dan rendah penyakit jantung berdasarkan data klinis sederhana.
- Menentukan faktor risiko dominan dengan mengidentifikasi dan menentukan urutan kepentingan faktor faktor(misalnya, tekanan darah, kolesterol, detak jantung) yang paling berpengaruh terhadap risiko terjadinya penyakit jantung.
- Mengembangkan dan menerapkan model Machine Learning yang akurat untuk mengklasifikasikan risiko penyakit jantung secara otomatis berdasarkan data.

### Solution statements
- Mengembangkan dan membandingkan beberapa model klasifikasi seperti Decision Tree, Random Forest, K-Nearest Neighbours, Logistic Regression dengan tujuan untuk memilih model dengan performa terbaik berdasarkan evaluasi metrik seperti *accuraccy*, *Precision*, *Recall*, *F1 Score*, dan *ROC AUC*
- Berdasarkan beberapa evaluasi metrik tersebut dipilih dua metrik evaluasi yang menjadi faktor penting dalam memilih model dengan performa terbaik yaitu F1-score dan ROC AUC karena kedua metrik ini dapat memberikan gambaran yang lebih seimbang dan menyeluruh tentang performa model klasifikasi, terutama dalam kondisi tertentu seperti data yang tidak seimbang dan dapat menunjukkan kemampuan model dalam membedakan kelas positif dan negatif berdasarkan probabilitas
- Model akan diuji dan divalidasi untuk memastikan kinerjanya kemudian dapat diintegrasikan ke dalam sistem pendukung keputusan sebagai alat bantu dalam proses diagnosis penyakit jantung

## Data Understanding
Dataset ini menyediakan informasi klinis dari sejumlah pasien, yang mencakup berbagai fitur yang diduga berkorelasi dengan risiko penyakit jantung. Setiap entri (baris) dalam dataset mewakili data seorang pasien, sementara setiap kolom mencatat atribut medis atau demografis yang relevan. Struktur dataset terdiri dari total 918 observasi (baris) dan 12 variabel (atribut). Dataset ini dapat diunduh dan dieksplorasi lebih lanjut melalui tautan
[Kagle](https://www.kaggle.com/code/ahmedmohamed33116/heart-failure-prediction)

### Variabel-variabel pada klasifikasi penyakit jantung adalah sebagai berikut:
- age : merupakan usia pasien dalam satuan tahun dengan tipe data numerik
- sex : merupakan data kategorikal jenis kelamin pasien 'M' untuk Male dan 'F' untuk Female
- ChestPainType : merupakan data kategorikal tipe nyeri dada pada pasien dengan kode (ATA, NAP, ASY, dan TA)
- RestingBP : merupakan data numerik tekanan darah saat istirahat(mm Hg)
- cholesterol : merupakan data numerik kadar kolesterol serum dalam satuan mg/dl.
- fastingBS : merupakan data kategorikal apakah (gula darah puasa > 120mg/dl) 0 mewakili false dan 1 mewakili TRUE.
- RestingECG : merupakan data kategorikal hasil elektrodiogram saat istirahat.
- MaxHR : merupakan data numerik detak jantung maksimum yang dicapai.
- ExerciseAngina : merupakan data kategorikal angina akibat olahraga(Y/N)
- Oldpeak : merupakan data numerik depresi ST setelah olahraga dibandingkan saat istirahat(dalam mm).
- ST_Slope : merupakan data kategorikal kemiringan segmen ST selama latihan.
- HeartDisease : merupakan kolom biner target klasifikasi yang menjadi indikator apakah pasien memiliki peyakit jantung.

Melakukan tahapan Exploratory Data Analysis(EDA) untuk memahami dataset. Berikut hasil EDA yang sudah dilakukan:
### Cek Distribusi Target
Tahapan ini dilakukan untuk mengetahui distribusi target HeartDisease. Berdasarkan grafik distribusi target HeartDisease, diketahui bahwa jumlah individu yang tidak memiliki penyakit jantung lebih banyak dibandingkan yang memiliki penyakit. Meski begitu, perbandingan jumlahnya masih cukup proporsional sekitar 57% dan 43%, sehingga data dianggap relatif seimbang. Kondisi ini mendukung validitas data untuk training model klasifikasi dan menegaskan pentingnya prediksi yang akurat terhadap kasus penyakit jantung yang jumlahnya cukup signifikan dalam dataset.
### Distribusi Usia Pasien
Grafik menunjukkan bahwa penyakit jantung lebih sering terjadi pada individu berusia 50 tahun keatas. Sementara pada usia dibawah 40, kasus penyakit jantung masih tergolong rendah. Hal ini mengindikasikan bahwa usia merupakan faktor risiko penteing terhadap penyakit jantung, sehingga upaya pencegahan dan deteksi dini sebaiknya difokuskan pada kelompok usia paruh baya ke atas.
### Distribusi Jenis Kelamin terhadap HeartDisease
Berdasarkan grafik distribusi jenis kelamin terhadap status penyakit jantung, terlihat bahwa male mendominasi jumalh penderita penyakit jantung dibandingkan female. Sebagian female dalam dataset dalam dataset tidak menderita penyakit jantung. Temuan ini mendukung ipotesis bahwa jenis kelamin berpotensi menjadi salah satu faktor risiko terhadap penyakit jantung, khususnya pada laki-laki.
### Distribusi ChestPainType terhadap HeartDisease
Berdasarkan grafik distribusi jenis nyeri dada terhadap status penyakit jantung, ditemukan bahwa jenis dada Asymptomatic(ASY) paling banyak terkait dengan kasus penyakit jantung. Hal ini menunjukkan bahwa sebagian besar penderita jantung dalam dataset tidak menunjukkan gejala nyeri dada yang khas. Sebaliknya, nyeri dada tipe ATA dan NAP justru lebih umum ditemukan pada individu tanpa penyakit jantung, sehingga bisa dikatakan bahwa jenis nyeri dada tertentu dapat menjadi indikator penting dalam mendeteksi risiko penyakit jantung.
### Corellation Matrix
Heatmap korelasi menunjukkan bahwa fitur Oldpeak dan MaxHR memiliki hubungan yag paling kuat terhadap penyakit jantung. Oldpeak memiliki korelasipostif positif, artinya semakin tinggi nilainya, semakin besar kemungkinan pasien menderita penyakit jantung. Sebaliknya, MacHR memiliki korelasi negatif, mengindikasikan bahwa pasien dengan denyut jantung maksimum yang lebih tinggi saat latihan cenderung lebih sehat. Korelasi fitur lain terhadap penyakit jantung tegolong lemah, meskipun tetap relevan untuk diperhitungkan dalam permodelan prediktif.

## Data Preparation
Proses data preparation yang dilakukan dalam membangun model klasifikasi ini antara lain:
### Cek Duplikasi Data
Menggunakan kode heart_disease.duplicated().sum(), kode ini digunakan untuk mengecek berapa banyak baris yang mengalami duplikasi. heart_disease.drop_duplicates(), digunakan untuk menghapus baris baris yang terdeteksi sebagai duplikat. Proses ini dilakukan untuk menghindari bias pada data yang berpotensi membuat model belajar berlebihan pada pola yang sebenarnya tidak terlalu penting karena sering muncul berulang sehingga performa menurun terutama pada data baru. Selain itu tahapan ini juga merupakan tahap data cleaning untuk memastikan dataset bersih dan hanya berisi data unik yang benar benar dibutuhkan.
### Cek Missing Values
print(heart_disease.isna().sum()), proses ini dapat mengidentifikasi apakah ada nilai Nan pada setiap kolom datafram heart_disease yang kemudian hasilnya ditambahkan jumlahnya. Proses ini dilakukan untuk menjamin kualitas data sebelum mempersiapkan data untuk permodelan karena banyak algoritma machine learning yang tidak bisa menangai data kosong secara langsung 
### Encoding Kategorikal Fitur
Proses ini diawali dengan mendeklarasikan label encoder sebagai teknik encoding untuk fitur kategorikal. Label encoder digunakan untuk fitur Jenis Kelamin dan Exercise Angina karena kedua fitur tersebut hanya memiliki dua nilai kategorikal yaitu M/F dan Y/N, sehingga bisa diubah menjadi angka 0 dan1 menggunakan label encoder. Untuk fitur kategorikal dengan lebih dari dua kategori, digunakan One-Hot Encoding dengan pd.get_dummies. Sedangkan drop_first=True digunakan untuk menghindari multikolinearitas dengan membuang salah satu kolom dummy pertama dari setiap fitur.Tujuan dilakukan Encoding adalah menyiapkan data kategorikal agar bisa digunakan oleh algoritma machine learning, karena sebagian besar algoritma hanya menerima input numerik dan tujuan selanjutnya adalah menghindari bias numerik dimana label encoding cocok untuk data biner, sementara one-hot encoding mencegah model menganggap adanya urutan atau bobot pada kategori tidak berurutan. 
### Feature Scaling
Proses ini diawal dengan menginisialisasi objek StandardScaler, yaitu metode normalisasi yang mengubah nilai fitur numerik agar memiliki rata-rata 0 dan standar deviasi 1. Selanjutnya adalah melakukan proses transformasi pada kolom numerik yang digunakan yang akan menghailkan data yang sudah distandarisasi dan menggantikan nilai sebelumnya. Tujuan dilakukan feature scaling adalah untuk membuat semua fitur numerik berada dalam skala yang sebanding sehingga dapat meningkatkan performa dan kecepatan konvergensi algoritma selama training model dan untuk menghindari dominasi fitur dengan nilai besar terhadap fitur dengan nilai kecil dalam perhitungan jarak atau bobot. Hal ini penting dilakukan untuk algoritma yang sensitif terhadap skala seperti, KNN, SVM, Logistic Regression dll.
### Data Spliting
Proses ini membagi dataset menjadi 80% data latih yang digunakan untuk melatih model dan 20% data uji yang digunakan untuk mengevaluasi performa model. Kode random_state=42 digunakan untuk memastikan pembagian data selalu konsisten jika kode dijalankan ulang. Tujuan dilakukan data spliting adalah untuk melatih dan menguji model secara adil dimana data latih digunakan untuk membangun model dan data uji digunakan untuk mengevaluasi kinerja model terhadap data yang belum pernah dilihat sbeelumnya. Proses ini dapat mencegah terjadinya overfitting dan menentukkan akurasi performa awal.

## Modeling
### K-Nearest Neighbors (KNN)
Parameter yang digunakan dalam proses permodelan menggunakan KNN adalah n_neighbors=5 yang berarti model akan mengklasifikasikan data berdasarkan mayoritas label dari 5 tetangga terdekat. Model ini dilatih menggunakan dataset pelatihan (X_train, y_train) dan diuji pada X_test. Kelebihan dari model KNN adalah model ini mudah dipahami dan diimplementasikan karena tidak memiliki asumsi terhadap distribusi data dan cocok untuk dataset kecil hingga menengah. Sedangkan kekurangan dari model KNN adalah lambat pada dataset bear karena harus menghitun jarak terhadap semua data, sendisitif terhadap skala dan noise dan memerlukan memori besar karena menyimpan semua data training.
### Decision Tree
Tahapan permodelan menggunakan Decision Tree adalah dengan membuat objek model DecisionTreeClassifier dengan parameter random_state=42 yang digunakan afgar asil dapat direproduksi. Kemudian melatih model pada data latih (X_train, y_train). Selanjutnya melakukan prediksi terhadap data ui yang menghasilkan label 0 atau 1. y_probe_dt digunakan untuk mengambil probabilitas kelas positif 1 untuk digunakan dalam ROC AUC Score. Kelebihan model Decision Tree adalah mudah dipahami dan divisualisasikan, tidak perlu normalisasi fitur, dan bisa menangani data kategorikal dan numerik. Sedangkan kekurangannua adalah model mudah mengalami overfitting, terutama pada data kecil atau kompleks sehingga membuat sensitif terhadap perubahan kecil pada data dan berpotensi membuat performa kurang stabil.
### Logistic Regression
Tahapan untuk membuat model Logistic Regression adalah dengan membuat objeknya terkebih dahulu lalu menambahkan oarameter max_iter=1000 yang digunakan agar proses training mencapai konvergensi, terutama jika datanya kompleks. Parameter selanjutnya adalah random_state=41 digunakan untuk memastikan hasil konsisten. Setelahnya dilakukan pelatihan model pada data training(X_train, y_train) yang selanjutnya dapat memprediksi label kelas 0 atau 1 berdasarkan data uji. Terakhir mengambil probabilitas kelas postif, yang digunakan dalam ROC AUC Score. Metrik Evaluasi yang digunakan untuk mengevaluasi model ini antara lain, Accuraccy, Precision, Recall, F1-Scroe, ROC AUC. Kelebihan dari model logistic regression adalah model cepat dilarih dan cocok untuk baseline model. Memberikan probabilitas, cocok untuk ROC AUC. Sedangkan kekurangan dari model ini adalah model ini tidak mampu menangkap hbungan non-linear yang kompleks sehingga performa model bisa buruk jika terdapat outlier ekstrem atau korelasi multikolinearitas tinggi.
### Random Forest
Tahapan ini diawali dengan menginsialisasi model RandomForestClassifier dengan parameter random_state=42. Selanjutnya melatih model pada data training(X_train, y_train). Variabl y_pred_rf digunakan untuk memberikan prediksi kelas 0 atau 1 dan variabel y_proba_rf digunakan untuk menghitung probabilitas kelas positif yang digunakan untuk ROC AUC Score. Metriks evaluasi yang digunakan untuk mengevaluasi model inia antara lain, Accuracy, Precision, Recall, F1-Score dan ROC AUC. Kelebihan dari model random forest adalah model cukup akurat dan kuat dalam menghadapai overfitiing dan model ini dapat menangani fitur numerik & kategorikal. Selain itu, model ini cukup stabil terhadap outlier dan noise. Sedangkan kekurangan model random forest ini adalah training yang lebih lambat daripada model lain sehingga memakan memori yang lebih besar dan membuat model ini kurang interpretable dibanding model lain seperti Logistic Regression.

Model terbaik berdasarkan metrik evaluasi adalah Random Forest karena memiliki nilai metrik tertinggi daripada model lain. Model ini cenderung lebih stabil karena merupakan kombinasi banyak pohon keputusan. Dan model ini dpaat menangani fitur numerik dan kategorikal, serta tidak terlalu sensitif terhadap scaling data. 

## Evaluation
Dalam proyek klasifikasi ini, digunakan lima metrik utama untuk mengevaluasi performa model klasifikasi. metrik yang digunakan seperti accuraccy, precision, recall, F1-score, dan ROC AUC. Pemilihan metrik ini disesuaikan dengan tujuan mengevaluasi performa prediksi model dalam mengklasifikasikan kelas dengan seimbang dan akurat.
Accuraccy mengukur presentase prediksi yang benar terhadap total seluruh data uji. Meskipun umum digunakan, metrik ini bisa menjadi kurang representatif jika data tidak seimbang. Oleh karena itu, metrik precision dan recall juga digunakan. Precision menilai berapa banyak dari nilai prediksi positif yang benar benar positif, yang penting ketika kita ingin meminimalkan kesalahan dalam memprediksi positif (false positive). Sebaliknya, recall mengukur seberapa banyak data postif yang berhasil terdeteksi oleh model, yang penting jika kesalahan melewatkan data positif(false negatif) sangat berdampak. Untuk menyeimbangkan keduanya F1-Score digunakan karena merupakan rata-rata harmonis dari precision dan recall, sehingga memberikan gamvaran menyeluruh atas keampuan model dalam menanganai ketidakseimbangan kelas.
Selain keempat metrik tersebut, pproyek kalsifikasi ini juga menggunakan ROC AUC sebagai metrik evaluasi tambahan. Metrik ini sangat penting dalam konteks klasifikasi biner karena menunjukkan kemampuan model dalam membedakan kelas positif dan negatif berdasarkan probabilitas. Semakin tinggi nilai ROC AUC, maka semakin baik model dalam mengidentifikasi dua kelas dengan benar. Nilai ROC AUC berkisar dari 0.5 hingga 1.0  
​
Empat algoritma yang digunakan dalam proyek ini adalah K-Nearest Neighbors(KNN), Decision Tree, Logistic Regression, dan Random Forest. Berikut adalah hasil metrik evaluasi dari empat algoritma klasifikasi yang telah dilatih:

| Model                         | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| ----------------------------- | -------- | --------- | ------ | -------- |---------|
| **K-Nearest Neighbors (KNN)** | 0.92     | 0.92      | 0.92   | 0.92     | 0.98    |
| **Decision Tree**             | 0.96     | 0.96      | 0.96   | 0.96     | 0.96    |
| **Logistic Regression**       | 0.98     | 0.98      | 0.98   | 0.98     | 0.98    |
| **Random Forest**             | 0.99     | 0.99      | 0.99   | 0.99     | 0.99    |

Berdasarkan hasil evaluasi, **Random Forest** memberikan nilai tertinggi dalam metrik evaluasi utama, yang menunjukkan bahwa model ini memiliki performa paling stabil dan mampu mengklasifikasikan data dengan akurat. Logistic Regression juga memberikan hasil yang baik, terutama pada ROC AUC, dan memiliki keunggulan dalam hal interperabilitas. Decision Tree memiliki performa yang cukup baik, namun rentan terhadap overfitting. Sementara itu, model KNN menunjukkan performa yang cukup, tetapi dapat terpengaruh oleh distribusi dan skala data.

Berdasarkan hasil tersebut, **Random Forest** dipilih sebagai model terbaik karena mampu memberikan hasil evaluasi tertinggi dalam metrik yang paling penting untuk kaus ini, yaitu F1-Score dan ROC AUC. Selain itu, Random Forest memiliki keunggulan dalam menangani fitur yang kompleks dan memberikan generalisasi yang baik terhadap data baru. Model ini menjadi solusi akhir yang digunakan dalam proyek klasifikasi ini.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

