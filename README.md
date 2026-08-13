# Stock Price Prediction with PyTorch

10 hissenin (AAPL, MSFT, NVDA, META, PLTR, GOOGL, TTWO, TSLA, AMZN, AMD) geçmiş fiyat verileri kullanılarak, PyTorch ile geliştirilen bir LSTM (Long Short-Term Memory) modeliyle bir sonraki günün kapanış (Close) fiyatını tahmin etmeyi amaçlayan bir zaman serisi projesi. Her hisse için ayrı bir model eğitilir. Eğitim ve çıkarım tamamen CPU üzerinde çalışacak şekilde tasarlanmıştır (GPU/CUDA gerektirmez).

## Özellikler

- **Veri çekme:** `yfinance` ile 10 hisse için son 5 yıllık günlük OHLCV (Open, High, Low, Close, Volume) verisi otomatik olarak indirilir.
- **Veri temizliği:** Eksik değer kontrolü ve doldurma, z-score tabanlı anormal fiyat sıçraması (outlier) tespiti.
- **Teknik indikatörler:** 7 günlük (MA7) ve 21 günlük (MA21) hareketli ortalama, günlük getiri (daily return).
- **PyTorch LSTM modeli:** 2 katmanlı, dropout'lu, tamamen parametrik (hidden size, learning rate, epoch, batch size) bir LSTM mimarisi.
- **Hiperparametre denemeleri:** Farklı learning rate / epoch / hidden size kombinasyonlarının train loss üzerinden karşılaştırılması.
- **Değerlendirme ve görselleştirme:** Test seti üzerinde RMSE/MAE/MAPE metrikleri ve gerçek vs. tahmin karşılaştırma grafiği.

## Kurulum

Python 3.12 gereklidir.

```bash
# Virtual environment oluştur
python -m venv venv

# Aktive et (Windows PowerShell)
venv\Scripts\Activate.ps1

# Bağımlılıkları kur (CPU tabanlı torch dahil)
pip install -r requirements.txt
```

## Kullanım

### Tüm hisseler için tek komutla (önerilen)

`src/run_all.py`, 10 hissenin hepsi için fetch → preprocess → prepare_dataset → train → evaluate → plot adımlarını sırayla, bilinen en iyi hiperparametrelerle (`learning_rate=0.0005`, `epoch=30`, `hidden_size=64`) otomatik çalıştırır ve sonunda tüm hisseleri karşılaştıran bir özet tablo üretir:

```bash
python src/run_all.py   # 10 hisse icin tum pipeline -> outputs/summary_results.csv
```

CPU üzerinde 10 hisse için birkaç dakika sürebilir.

### Tek bir hisse için adım adım

Her script `ticker` parametresi alır; script'ler `src/` klasöründe bulunur ve aşağıdaki sırayla çalıştırılmalıdır (varsayılan olarak `__main__` bloklarında `"AAPL"` kullanılır, farklı bir hisse için ilgili fonksiyonu kendi ticker'ınızla çağırabilirsiniz):

```bash
python src/fetch_data.py       # Tum TICKERS listesini indirir -> data/{TICKER}_raw.csv
python src/preprocess.py       # Temizler, MA7/MA21/daily_return ekler -> data/{TICKER}_processed.csv
python src/prepare_dataset.py  # Normalize eder, train/test split + LSTM sequence'ları oluşturur -> data/{TICKER}_X_train.npy, y_train.npy, X_test.npy, y_test.npy, scaler.pkl
python src/train.py            # Hiperparametre kombinasyonlarını dener, en iyi modeli kaydeder -> models/{TICKER}_stock_lstm.pt
python src/evaluate.py         # Test seti üzerinde RMSE/MAE/MAPE hesaplar
python src/plot_results.py     # Gerçek vs. tahmin grafiğini üretir -> outputs/{TICKER}_prediction_vs_actual.png
```

## Proje Yapısı

```
stock-price-prediction/
├── data/                          # Ham/işlenmiş veri ve model girdileri (npy/pkl dosyaları git'e dahil değildir)
│   ├── {TICKER}_raw.csv           # yfinance'ten çekilen ham OHLCV verisi (ör. AAPL_raw.csv, MSFT_raw.csv, ...)
│   ├── {TICKER}_processed.csv     # Temizlenmiş veri + MA7, MA21, daily_return
│   ├── {TICKER}_X_train.npy / {TICKER}_y_train.npy
│   ├── {TICKER}_X_test.npy / {TICKER}_y_test.npy
│   └── {TICKER}_scaler.pkl        # O hisseye ozel fit edilmiş MinMaxScaler (tahminleri geri ölçeklemek için)
├── models/
│   └── {TICKER}_stock_lstm.pt     # Her hisse icin ayri egitilmiş LSTM model ağırlıkları
├── outputs/
│   ├── {TICKER}_prediction_vs_actual.png  # Her hisse icin gerçek vs. tahmin karşılaştırma grafiği
│   └── summary_results.csv        # 10 hissenin RMSE/MAE/MAPE ozet karsilastirma tablosu
├── notebooks/                     # Keşifsel analiz için ayrılmış klasör
├── src/
│   ├── fetch_data.py              # yfinance ile 10 hisse icin veri çekme (TICKERS listesi)
│   ├── preprocess.py              # Veri temizliği ve teknik indikatörler (ticker parametreli)
│   ├── prepare_dataset.py         # Normalizasyon, train/test split, sequence windowing (ticker parametreli)
│   ├── model.py                    # StockLSTM (PyTorch nn.Module) tanımı
│   ├── dataset.py                 # StockDataset ve DataLoader'lar (ticker parametreli)
│   ├── train.py                    # Training loop, hiperparametre denemeleri, train_and_save (ticker parametreli)
│   ├── evaluate.py                # Test seti değerlendirmesi (RMSE/MAE/MAPE, ticker parametreli)
│   ├── plot_results.py            # Gerçek vs. tahmin grafiği (ticker parametreli)
│   └── run_all.py                 # 10 hisse icin tum pipeline'i otomatik calistirir, ozet tablo uretir
├── requirements.txt
└── README.md
```

## Sonuçlar

Her hisse için aynı hiperparametrelerle (`learning_rate=0.0005`, `epoch=30`, `hidden_size=64`) ayrı ayrı eğitilen modellerin test seti sonuçları, MAPE'ye göre en iyiden en kötüye sıralı:

| Ticker | RMSE ($) | MAE ($) | MAPE (%) |
|---|---|---|---|
| AAPL | 7.75 | 6.06 | 2.10 |
| TSLA | 14.59 | 11.37 | 2.90 |
| AMZN | 9.84 | 7.23 | 3.00 |
| TTWO | 8.90 | 6.84 | 3.03 |
| MSFT | 18.94 | 13.22 | 3.09 |
| NVDA | 7.77 | 6.28 | 3.19 |
| GOOGL | 15.37 | 12.50 | 3.79 |
| META | 36.79 | 26.98 | 4.24 |
| PLTR | 10.52 | 7.60 | 5.28 |
| AMD | 39.72 | 30.41 | 7.77 |

Tam tablo `outputs/summary_results.csv` dosyasında.

**Gözlem:** Model, AAPL, TSLA, AMZN, TTWO, MSFT ve NVDA gibi göreli olarak daha istikrarlı fiyat hareketine sahip hisselerde belirgin şekilde daha düşük hata veriyor (MAPE %2-3 civarı). Buna karşılık PLTR ve özellikle AMD gibi daha volatil, ani ve sert fiyat sıçramaları yaşayan hisselerde hata payı gözle görülür şekilde artıyor (PLTR %5.28, AMD ise %7.77 ile en yüksek hata oranına sahip). Bu, modelin **hisse volatilitesine duyarlı** olduğunu gösteriyor: LSTM geçmiş fiyat hareketlerinden öğrendiği için, fiyatı daha öngörülebilir/trend takip eden hisselerde daha isabetli, ani rejim değişiklikleri yaşayan hisselerde ise daha isabetsiz tahminler üretiyor.

En iyi performans gösteren AAPL için gerçek vs. tahmin grafiği:

![AAPL Gerçek vs Tahmin](outputs/AAPL_prediction_vs_actual.png)

## Sınırlamalar / Geliştirme Fikirleri

- Model, fiyatlardaki **keskin dönüşlerde (ani yükseliş/düşüşlerde) gecikme (lag)** yaşıyor — bu, LSTM'lerin bir önceki değere yakın tahmin üretme eğiliminden kaynaklanan, zaman serisi modellerinde sık görülen bir davranış.
- Şu an 10 hisseyle sınırlı ve her hisse için **bağımsız bir model** eğitiliyor (paylaşılan/çoklu hisse öğrenmesi yok); daha geniş bir evrene veya farklı sektörlere genelleme test edilmedi.
- Ek **teknik indikatörler** (RSI, MACD, Bollinger Bantları, hacim tabanlı göstergeler vb.) eklenerek modelin daha fazla sinyal görmesi sağlanabilir.
- **Farklı model mimarileri** (GRU, Transformer tabanlı zaman serisi modelleri, çok değişkenli/çok hisseli modeller) denenebilir.
- Şu anki yaklaşım yalnızca **tek adım ileri (bir sonraki gün)** tahmin yapıyor; çok adımlı (multi-step) tahmin genişletilebilir.
- Haber/duyarlılık (sentiment) verisi gibi fiyat dışı sinyaller modele dahil edilmedi.
