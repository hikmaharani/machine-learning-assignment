import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.title("Student Performance Factors")
st.write("Project ini dibuat oleh Hikmah Maharani untuk menyelesaikan Study Case Machine Learning GDG on Campus Universitas Sriwijaya")

st.header("Tentang Dataset")
st.write("Deskripsi: Dataset ini memberikan gambaran menyeluruh tentang berbagai faktor yang mempengaruhi kinerja siswa dalam ujian. Data ini mencakup informasi mengenai kebiasaan belajar, kehadiran, keterlibatan orang tua, dan aspek-aspek lain yang memengaruhi keberhasilan akademik.")
st.write("Link Dataset: [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)")

st.header("1. Data Wrangling")
st.subheader("a. Menampilkan 10 baris pertama dari dataset")

df = pd.read_csv('https://drive.google.com/uc?id=1RU2r4Lt80AhmB8WDcV4GvJwwS4Bp0_7W')
st.write(df.head(10))

st.subheader("Deskripsi Kolom")
st.write("""
         1. Hours_Studied: Jumlah jam yang dihabiskan untuk belajar per minggu.
         2. Attendance: Persentase kehadiran di kelas.
         3. Parental_Involvement: Tingkat keterlibatan orang tua dalam pendidikan siswa (Rendah, Sedang, Tinggi).
         4. Access_to_Resources: Ketersediaan sumber daya pendidikan (Rendah, Sedang, Tinggi).
         5. Extracurricular_Activities: Partisipasi dalam kegiatan ekstrakurikuler (Ya, Tidak).
         6. Sleep_Hours: Jumlah rata-rata jam tidur per malam.
         7. Previous_Scores: Nilai dari ujian sebelumnya.
         8. Motivation_Level: Tingkat motivasi siswa (Rendah, Sedang, Tinggi).
         9. Internet_Access: Ketersediaan akses internet (Ya, Tidak).
         10. Tutoring_Sessions: Jumlah sesi bimbingan belajar yang diikuti per bulan.
         11. Family_Income: Tingkat pendapatan keluarga (Rendah, Sedang, Tinggi).
         12. Teacher_Quality: Kualitas guru (Rendah, Sedang, Tinggi).
         13. School_Type: Jenis sekolah yang diikuti (Negeri, Swasta).
         14. Peer_Influence: Pengaruh teman sebaya terhadap prestasi akademik (Positif, Netral, Negatif).
         15. Physical_Activity: Jumlah rata-rata jam aktivitas fisik per minggu.
         16. Learning_Disabilities: Adanya gangguan belajar (Ya, Tidak).
         17. Parental_Education_Level: Tingkat pendidikan tertinggi orang tua (SMA, Perguruan Tinggi, Pascasarjana).
         18. Distance_from_Home: Jarak dari rumah ke sekolah (Dekat, Sedang, Jauh).
         19. Gender: Jenis kelamin siswa (Laki-laki, Perempuan).
         20. Exam_Score	: Nilai ujian akhir.
         """)

st.subheader("b. Informasi Dataset")

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.write(""" 
         Pada bagian ini menampilkan informasi terkait dataset mulai dari total baris, jumlah kolom, tipe data dan memori yang digunakan.
         1. Total baris: 6607
         2. Jumlah kolom: 20
         3. Tipe data: Object((14), int64(7)
         4. Memori: 1.0+ MB
         """)

st.header("2. Data Availability")
st.subheader("a. melihat nilai missing values")

st.write(df.isnull().sum())

st.write("Pada dataset ini terdapat missing values pada kolom Teacher_Quality, Parental_Education_Level, dan Distance_from_Home dengan jumlah 78, 90, dan 67 respectively.")
st.write("Tekan tombol dibawah ini untuk menghilangkan nilai missing values")
if st.button("Hapus Missing Values"):
    df = df.dropna()
    st.write("Dataset Setelah Menghapus Missing Values:")
    st.dataframe(df.isnull().sum())
    
st.subheader("b. melihat nilai duplikat")

st.write("Nilai duplikat:", df.duplicated().sum())

st.write("Terlihat pada dataset ini tidak memiliki nilai duplikat, sehingga tidak perlu dilakukan penghapusan duplikat.")

st.header("3. Exploratory Data Analysis (EDA)")
st.subheader("a. melihat deskripsi data")

st.write(df.describe())

st.write("""
            1. count : menunjukkan jumlah data untuk setiap kolom yaitu 6607 data

            2. mean : menunjukkan nilai rata-rata dari setiap kolom
            - Hours_Studied: Rata-rata siswa belajar selama 19.98 jam.
            - Attendance: Kehadiran rata-rata siswa adalah 79.98%.
            - Sleep_Hours: Siswa rata-rata tidur 7.02 jam.
            - Previous_Scores: Rata-rata nilai sebelumnya adalah 75.07.
            - Tutoring_Sessions: Rata-rata siswa mengikuti 1.49 sesi tutor.
            - Physical_Activity: Rata-rata aktivitas fisik siswa adalah 2.97.
            - Exam_Score: Rata-rata skor ujian adalah 67.24.

            3. std (standar deviasi) : menunjukkan seberapa besar data tersebar dari nilai rata-rata
            - Kolom Hours_Studied memiliki deviasi sebesar 5.99 jam, menunjukkan variasi waktu belajar yang cukup besar di antara siswa.
            - Nilai Exam_Score memiliki deviasi 3.89, artinya skor ujian antar siswa relatif homogen.
            - Kolom seperti Physical_Activity memiliki deviasi lebih rendah, menunjukkan distribusi data yang lebih seragam.

            4. min (nilai minimum): nilai terkecil dalam setiap kolom
            - Hours_Studied: Ada siswa yang hanya belajar 1 jam.
            - Attendance: Kehadiran minimum adalah 60%.
            - Sleep_Hours: Siswa yang tidur paling sedikit tidur selama 4 jam.
            - Exam_Score: Nilai terendah adalah 55.
            
            5. 25% (Q1): nilai di bawah 25% dari data
            - Hours_Studied: 25% siswa belajar kurang dari 16 jam.
            - Attendance: Kehadiran 25% siswa kurang dari 70%.
            - Exam_Score: Skor ujian 25% siswa di bawah 65.

            6. 50% (Q2): nilai tengah dari data
            - Hours_Studied: 50% siswa belajar kurang dari 20 jam.
            - Attendance: Kehadiran 50% siswa adalah 80%.
            - Exam_Score: 50% siswa memiliki skor ujian 67 atau lebih rendah.

            7. 75% (Q3): nilai di bawah 75% dari data
            - Hours_Studied: 75% siswa belajar kurang dari atau sama dengan 24 jam.
            - Attendance: Kehadiran 75% siswa adalah 90% atau lebih rendah 
            - Exam_Score: Skor ujian 75% siswa di bawah 69.

            8. max (nilai maksimum): nilai terbesar dalam setiap kolom
            - Hours_Studied: Ada siswa yang belajar hingga 44 jam.
            - Attendance: Kehadiran maksimum adalah 100%.
            - Exam_Score: Nilai ujian tertinggi adalah 101 (terdapat data outlier)
            """)

st.subheader("b. Heatmap korelasi")

corr = df[["Hours_Studied","Attendance","Previous_Scores","Sleep_Hours", "Tutoring_Sessions", "Physical_Activity", "Exam_Score"]].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.write("Heatmap ini menunjukkan korelasi setiap kolom numerikal dengan Exam_Score yang dikelompokkan menjadi korelasi kuat, korelasi sedang, dan korelasi lemah. Dengan ini dapat diketahui hubungan antar fitur-fitur tersebut dengan skor ujian.")
st.write("""
         1. Korelasi kuat: menunjukkan hubungan yang kuat antara dua variabel
         - Attendance dan Exam_Score meiliki korelasi sebesar 0.58 menunjukkan hubungan atau korelasi yang kuat. Ini berarti tingkat persentase kehadiran yang tinggi akan meningkatkan skor ujian.
         - Hours_Studied dan Exam_Score memiliki korelasi sebesar 0.45 yanng artinya, semakin banyak waktu belajar, skor ujian akan meningkat.
         
         2. Korelasi sedang: menunjukkan hubungan yang sedang antara dua variabel
         - Previous_Scores dan Exam_Score memiliki korelasi sebesar 0.18 berarti nilai ujian sebelumnya memiliki hubungan yang sedang atau tidak terlalu kuat dengan skor ujian saat ini.
         - Tutoring_Sessiions dan Exam_Score memiliki korelasi sebesar 0.16 yang menunjukkan bahwa jumlah sesi bimbingan tidak memiliki hubungan yang terlalu dengan skor ujian.
         
         3. Korelasi lemah: menunjukkan hubungan yang lemah antara dua variabel
         - Physical_Activity dan Exam_Score memiliki korelasi sebesar 0.028 yang menunjukkan bahwa aktivitas fisik tidak memiliki hubungan yang signifikan dengan skor ujian.
         - Sleep_Hours dan Exam_Score memiliki korelasi sebesar -0.017 yang menunjukkan bahwa kualitas tidur hampir tidak memiliki hubungan yang signifikan dengan skor ujian
         """)

st.subheader("c. Boxplot numerikal dan katagorikal terhadap Exam_Score")
st.subheader("1. Boxplot numerikal terhadap Exam_Score")

numerical_columns = ['Hours_Studied', 'Attendance', 'Previous_Scores','Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity']

fig, axes = plt.subplots(3, 2, figsize=(30, 15), constrained_layout=True) 
axes = axes.flatten()

for i, col in enumerate(numerical_columns):
    sns.boxplot(x=df[col], y=df['Exam_Score'], ax=axes[i])
    axes[i].set_title(f'Exam Score vs {col}', fontsize=12)
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel('Exam Score', fontsize=10)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')
st.pyplot(fig)
st.write("Boxplot ini menunjukkan distribusi fitur numerikal terhadap Exam_Score")
st.write("""
         1. Hours_Studied terhadap Exam_Score
         - boxplot menunjukkan bahwa skor ujian meningkat dengan bertambahnya jam belajar
         - outlier yang muncul pada jam belajar yang tinggi, menunjukkan beberapa siswa yang jam belajar nya lebih lama tetapi tidak mendapat nilai yang tinggi
         
         2. Attendance terhadap Exam_Score 
         - tidak terlihat pola yanng jelas antara kehadiran dan nilai ujian siswa, tetapi median nilai stabil pada tingkat kehadiran tertentu
         - outlier merata di seluruh tingkat kehadiran, ini berarti menunjukkan bahwa faktor lain selain Attendance juga berpengaruh
         
         3. Previous_Scores terhadap Exam_Score 
         - boxplot menunjukkan bahwa semakin tinggi nilai sebelumnya maka semakin tinggi pula nilai ujian
         - outlier yang muncul pada nilai sebelumnya yang rendah, menunjukkan bahwa beberapa siswa bisa meningkatkan nilai ujian mereka meskipun nilai sebelumnya rendah
         
         4. Sleep_Hours terhadap Exam_Score 
         - boxplot menunjukkan nilai ujian relatif stabil di seluruh jam tidur. median nilai tidak terlalu berbeda untuk jam tidur yang berbeda
         - outlier meningkat pada siswa yang tidur lebih sedikit, menunjukkan efek terhadap nilai ujian
         
         5. Tutoring_Sessions terhadap Exam_Score 
         - boxplot menunjukkan sedikit efek terhadap peningkatan median nilai ujian hingga kategori tertentu (sekitar 4-6 sesi), setelah itu tidak ada peningkatan yang signifikan atau cenderung menurun
         - outlier yang muncul pada siswa yang mengikuti sesi bimbingan yang lebih sedikit
         
         6. Physical_Activity terhadap Exam_Score
         - boxplot menunjukkan bahwa aktivitas fisik tidak menunjukkan pola yang signifikan terhadapa nilai ujiaan
         - outlier merata dan median tidak menunjukkan nilai yang berbeda pada aktivitas fisik yang berbeda
         """)

st.subheader("2. Boxplot Katagorikal terhadap Exam_Score")

categorical_vars = [ 
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
    'School_Type', 'Peer_Influence', 'Learning_Disabilities',
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]

fig, axes = plt.subplots(5, 3, figsize=(20, 25), constrained_layout=True)
axes = axes.flatten()

for i, col in enumerate(categorical_vars):
    sns.boxplot(x=df[col], y=df['Exam_Score'], ax=axes[i])
    axes[i].set_title(f'Exam Score by {col}', fontsize=12)
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel('Exam Score', fontsize=10)
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')
st.pyplot(fig)

st.write("Boxplot ini menunjukkan distribusi fitur katagorikal terhadap Exam_Score")
st.write("""
         1. Parental_Involvement terhadap Exam_Score
         - boxplot menunjukkan nilai ujian meningkat dengan tingkat keterlibatan orang tua (low ke high)
         - outlier lebih banyak pada kategori Low dan Medium, yang menunjukkan pengaruh kuat dari keterlibatan orang tua terhadap performa siswa.
         
         2. Access_to_Resources terhadap Exam_Score
         - siswa yang memiliki akses sumber daya pendidikan yang tinggi memiliki median yang lebih tinggi dibanding dengan medium dan low
         - siswa dengan akses lebih rendah memiliki lebih banyak outlier dengan nilai rendah
         
         3. Extracurricular_Activities terhadap Exam_Score
         - siswa yang mengikuti kegiatan ekstrakurikuler memiliki median yang sedikit lebih tinggi dibandingkan dengan yang tidak mengikuti
         - perbedaan tidak signifikan dan variasi nilai di kedua kelompok cukup besar
         
         4. Motivation_Level terhadap Exam_Score
         - siswa yang memiliki motivasi tinggi memiliki median yang lebih tinggi dibandingkan dengan medium dan low
         - siswa dengan motivasi rendah memiliki lebih banyak outlier dengan nilai rendah
         
         5. Internet_Access terhadap Exam_Score
         - siswa yang memiliki akses internet memiliki median yang lebih tinggi dibandingkan dengan yang tidak memiliki akses internet 
         - outlier lebih banyak pada kategori Tidak memiliki akses internet menunjukkan nilai rendah
         
         6. Family_Income terhadap Exam_Score
         - siswa dari keluarga dengan pendapatan tinggi memiliki median yang lebih tinggi dibanding medium dan low
         - outlier dengan nilai rendah lebih banyak pada kategori Low 
         
         7. Teacher_Quality terhadap Exam_Score
         - kualitas guru yang tinggi memiliki median yang lebih tinggi dibanding medium dan low
         - kualitas mengajar guru menunjukkan pengaruh kuat terhadap performa siswa
         
         8. School_Type terhadap Exam_Score
         - siswa yang bersekolah di sekolah jenis swasta (private) memiliki median yang lebih tinggi dibandingkan dengan sekolah negeri (public)
         - siswa disekolah negeri menunjukkan lebih banyak outlier dengan nilai rendah
         
         9. Parental_Education_Level terhadap Exam_Score
         - orang tua dengan tingkat pendidikan postgraduate cenderung memiliki anak dengan nilai median yang lebih tinggi
         - siswa dengan orang tua yang berpendidikan high school menunjukkan lebih banyak outlier dengan nilai rendah
         
         10. Peer_Influence terhadap Exam_Score
         - siswa dengan teman dan lingkungan positif memiliki media nilai yang tinggi 
         - siswa dengan lingkungan negatif menunjukkan lebih banyak outlier dengan nilai rendah
         
         11. Learning_Ds=isabilities terhadap Exam_Score
         - siswa yang tidak memiliki gangguan belajar memiliki median yang lebih tinggi dibandingkan dengan yang memiliki gangguan belajar
         - ini menunjukkan bahwa gangguan belajar dapat mempengaruhi performa siswa
         
         12. Distance_from_Home terhadap Exam_Score
         - jarak rumah yang dekat (near) memiliki median yang lebih tinggi dibanding medium dan far
         - tidak ada perbedaan signifikan pada distribusi katagori
         
         13. Gender terhadap Exam_Score
         - median nilai untuk siswa laki laki dan perempuan hampir sama
         - tidak terdapat perbedaan yang signifikan antar kedua gender
         """)

st.subheader("d. Distribusi kolom kategorikal Terhadap Exam Score ")
categories = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
              'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
              'School_Type', 'Peer_Influence', 'Learning_Disabilities',
              'Parental_Education_Level', 'Distance_from_Home', 'Gender']

category_descriptions = {
    'Parental_Involvement': "Dukungan orang tua yang tinggi menunjukkan bahwa skor ujian siswa lebih tinggi dibanding dengan siswa yang mendapatkan dukungan orang tua yang rendah",
    'Access_to_Resources': "Ketersediaan sumber daya pendidikan yang tinggi menunjukkan skor ujian siswa yang lebih tinggi",
    'Extracurricular_Activities': "Aktivitas ekstrakulikuler tidak menunjukkan perbedaan yang signifikan dengan skor ujian",
    'Motivation_Level': "Tingkat motivasi siswa yang tinggi cenderung mempengaruhi skor ujian menjadi lebih tinggi",
    'Internet_Access': "Siswa yang memiliki akses internet cenderung memiliki skor ujian yang lebih tinggi dibanding dengan siswa yang tidak memiliki akses internet",
    'Family_Income': "Siswa dari keluarga dengan pendapatan tinggi memiliki skor ujian yang lebih tinggi",
    'Teacher_Quality': "Kualitas guru yang lebih tinggi mempengaruhi skor ujian siswa menjadi lebih tinggi",
    'School_Type': "Tipe sekolah tidak menunjukkan perbedaan yang signifikan dengan skor ujian",
    'Peer_Influence': "Pengaruh teman yang baij dan lingkungan sosial yang positif apat meningkat kan skor ujian",
    'Learning_Disabilities': "Siswa yang tidak memiliki gangguan belajar cenderung memiliki skor ujian yang lebih tinggi",
    'Parental_Education_Level': "Siswa yang memiliki orang tua dengan tingkat pendidikan yang tinggi cenderung memiliki skor ujian yang lebih tinggi",
    'Distance_from_Home': "Jarak antara rumah siswa ke sekolah tidak menunjukkan perbedaan yang signifikan dengan skor ujian",
    'Gender': "Tidak ada perbedaan yang signifikan antara gender dan skor ujian"
}

selected_category = st.selectbox("Pilih", categories)
st.subheader(f"Distribusi {selected_category} terhadap Exam Score")
st.write(category_descriptions[selected_category])

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=selected_category, y='Exam_Score', data=df, ax=ax)
ax.set_title(f'Distribusi {selected_category}')
ax.set_xlabel(selected_category)
ax.set_ylabel('Exam Score')
st.pyplot(fig)


st.subheader("e. distribusi kolom numerikal")

numerikal = ['Hours_Studied','Attendance', 'Sleep_Hours','Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']

select_num = st.selectbox("Pilih", numerikal)
st.subheader(f"Distribusi {select_num}")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df[select_num], kde=True, ax=ax)
ax.set_title(f'Distribution {select_num}')
ax.set_xlabel(select_num)
ax.set_ylabel('Frequency')
st.pyplot(fig)

st.subheader("f. Pairplot Hours_Studied, Attendance, dan Previous_Scores terhadap Exam_Score")

sns.set(style="ticks")
pairplot_fig = sns.pairplot(df, x_vars=['Hours_Studied', 'Attendance', 'Previous_Scores'], y_vars='Exam_Score')
st.pyplot(pairplot_fig)

st.write("""
         1. Hours_Studied dan Exam_Score terdapat pola positif dimana semakin banyak jam belajar maka skor ujian meningkat.
         2. Attendance dan Exam_Score menunjukkan pola positif dimana kehadiran yang lebih tinggi cenderung menghasilkan nilai ujian yang lebih baik, meskipun hubungannya tidak terlalu kuat.
         3. Previous_Scores dan Exam_Score menunjukkan data terlihat lebih tersebar tanpa pola yang jelas, berarti skor sebelumnya tidak menjadi indikator utama skor ujian.
         """)

st.subheader("Kesimpulan")
st.write("""
         - Faktor seperti jam belajar dan kehadiran memiliki korelasi positif terhadap skor ujian.
         - Dukungan eksternal seperti kualitas guru, akses sumber daya, dan dukungan orang tua sangat berpengaruh pada performa.
         - Faktor internal seperti motivasi siswa juga berperan penting dalam pencapaian nilai ujian.
         """)