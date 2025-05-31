# Kalp Krizi Riski Tahmini

Bu proje, makine öğrenmesi ve derin öğrenme yöntemleri kullanarak bireylerin kalp krizi geçirme riskini tahmin etmeyi amaçlamaktadır. Proje kapsamında Flask tabanlı bir web arayüzü ile kullanıcıdan alınan verilerle risk tahmini yapılabilmektedir.

## Literatür Bilgisi

Kalp-damar hastalıkları, dünya genelinde en yaygın ölüm nedenlerinden biridir. Erken teşhis ve risk analizi, hastalığın önlenmesi ve tedavi sürecinin iyileştirilmesi açısından büyük önem taşır. Makine öğrenmesi ve derin öğrenme teknikleri, tıbbi verilerden anlamlı sonuçlar çıkararak hastalık risklerinin tahmin edilmesinde yaygın olarak kullanılmaktadır. Özellikle yapay sinir ağları, karar ağaçları, rastgele ormanlar ve lojistik regresyon gibi yöntemler, kalp hastalığı tahmininde başarılı sonuçlar vermektedir (Daha fazla bilgi için: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)).

## Proje İçeriği

- **Veri Seti:** Projede [Heart Disease](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) veri seti kullanılmıştır.
- **Ön İşleme:** Kategorik veriler sayısal değerlere dönüştürülmüş, eksik veriler temizlenmiş ve SMOTE yöntemiyle veri dengelenmiştir.
- **Modeller:** Yapay Sinir Ağı, Random Forest, Karar Ağacı ve Lojistik Regresyon modelleri ile tahminler yapılmıştır.
- **Web Arayüzü:** Flask ile geliştirilen web arayüzü üzerinden kullanıcıdan alınan bilgilerle kalp krizi riski tahmini yapılabilmektedir.

## Kullanılan Kütüphaneler

- Python 3.x
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- tensorflow / keras
- flask

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
    ```
    pip install -r requirements.txt
    ```

2. Proje dosyalarını bir klasöre çıkarın ve `heart.csv` veri setinin bulunduğundan emin olun.

3. Uygulamayı başlatın:
    ```
    python app.py
    ```

4. Tarayıcınızda `http://127.0.0.1:5000` adresine giderek uygulamayı kullanabilirsiniz.

## Dosya Açıklamaları

- `app.py`: Flask tabanlı web uygulaması.
- `modeltest.py`: Farklı makine öğrenmesi modellerinin eğitim ve test kodları.
- `heart.csv`: Kalp hastalığı veri seti.
- `requirements.txt`: Gerekli Python kütüphaneleri.

## Katkı ve Lisans

Bu proje eğitim amaçlıdır. Katkıda bulunmak için lütfen bir pull request gönderin.

---

**Not:** Bu uygulama tıbbi teşhis amacıyla kullanılmamalıdır. Sonuçlar yalnızca bilgilendirme amaçlıdır.

## Geliştirici

- [Berkay Seyman](https://github.com/symNistaken)
