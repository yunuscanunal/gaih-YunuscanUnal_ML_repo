1) Makine Öğrenimini nasıl tanımlarsınız?
answer: 
Makine öğrenimi, tükettikleri verilere göre öğrenen ya da performansı iyileştiren sistemler oluşturmaya odaklanan bir yapay zeka (AI) alt kümesidir.

2) Denetimli ve Denetimsiz Öğrenim arasındaki farklar nelerdir? Bunların her biri için örnek 3 algoritma yazınız ve nasıl çalıştıkları hakkında kısaca bilgi veriniz.
answer:
gözetimli öğrenme; girdi verilerinden ve ilgili labellardan model eğitme. Linear regresyon, Logistic regresyon, Karar Ağacı
gözetimsiz öğrenme; işaretlenmemiş veri üzerinden bilinmeyen bir yapıyı öğrenmek için bir algoritma kullanan makine öğrenmesi tekniği. Hierarşik clustering,density-based clustering, K means

3) Eğitim, test ve doğrulama seti nedir ve neden onları kullanmamız gerekir?
answer:
Eğitim Seti : Modelin eğitildiği veri kümesidir.
Test Seti : Bir eğitim kümesinde geliştirilen modeli değerlendirmek için kullanılan bir veri kümesidir.
Doğrulama seti: Eğitim aşamasında elde edilen modelin performansını değerlendirmek için kullanılan alt bir veri setidir.
    Modelimizi eğitmek için eğitim veri setini kullanırız. Eğitim veri setini anlamlandırmak için eğitim seti ile model kuruyoruz ve bu anlamlandrımayı ölçmek , skorlamak için de test setini kullanıyoruz. Doğrulama setini de modelimizin ezberlememesi için kullanıyoruz.
    
4) Temel ön işleme (pre-processing) adımları nelerdir? Bu adımları ayrıntılı olarak açıklayınız ve verilerimizi neden ön işleme sokmamız gerektiği hakkında bilgi veriniz.
answer:
Bir modeli eğitmek için kullanmadan önce verileri işlemek.
	1.Veri seçme: Verilerin hepsini değil bizim için gerekli veriler seçilmeli
	2.Veri ön işleme: Biçimlendirme; veriler csv gibi bir veri formatına çevrilir.
        Temizleme; eksik veriler istatiksel yöntemlerle düzenlenebilir.
        Örnekleme; çok fazla veri varsa, bu veri setinden örnek olabilecek daha küçük veri seti alınır.
	3.Veri dönüştürme:Ölçekleme, Ayrıştırma, Toplama
    
5) Sürekli (continuous) ve ayrık (discrete) değişkenleri nasıl belirleyebilir ve analiz edebilirsiniz?
answer:
Discrete(ayrık); Tam sayı değerler.
Countionus(sürekli): Bir aralıktaki herhangibir değer alabilir.
Discrete için 
    
6) Makine öğrenimi içinde kullanılan temel istatiksel kavramlar nelerdir?
answer:
Korelasyon
Z score
Mean - median - std
Outliers (IQR)

7) Aşağıda verilen grafiği analiz ediniz. (Grafik ve değişken türü nedir, dağılımı nedir ve hangi ön işlemlerden geçmelidir?)
answer:
Seaborn kütüphanesinden distplot grafiği. sns.distplot(dataframe). Normal Dağılım grafiğidir, değişken türü sürekli değişkendir.
