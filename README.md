# **Master's Project Animal Re-Identification Application**


---

## **Instalacija**

Za postavljanje aplikacije na vašem računalu, slijedite sljedeće korake:

1. Klonirajte Git repozitorij s uključenim podmodulima:
   ```bash
   git clone --recursive-submodules https://github.com/matejmaricIA/Animal-Re-Identification---MSc-Project.git
   ```
2. Ažurirajte podmodule:
   ```bash
   git submodule update --init --recursive
   ```
3. Kreirajte virtualno okruženje za Python:
   ```bash
   python3 -m venv venv
   ```
4. Aktivirajte virtualno okruženje:
   - **Linux/MacOS**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
5. Instalirajte potrebne biblioteke:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Korištenje aplikacije**

Aplikacija podržava dva glavna načina rada: **treniranje modela** i **inferenciju (predikciju)**.

### **1. Treniranje modela**

Za treniranje modela na odabranom datasetu (npr. ATRW), koristite naredbu:
```bash
python main.py --train --ds ATRW
```

- **`--train`**: Pokreće aplikaciju u načinu treniranja.
- **`--ds`**: Specificira dataset koji će se koristiti za treniranje (npr. `ATRW`).

Tijekom treniranja:
- Podaci će biti podijeljeni u trening i test skupove.
- Modeli **PCA** i **GMM** bit će istrenirani na značajkama.
- Evaluacija će prikazati točnost i top-N točnost na testnom skupu.

Rezultati treniranja bit će spremljeni u definirane direktorije u projektu.

### **2. Inferencija (predikcija)**

Za izvođenje predikcija na novim slikama koristite naredbu:
```bash
python main.py --predict --image_location /path/to/dir
```

- **`--predict`**: Omogućuje način rada za predikciju.
- **`--image_location`**: Specificira direktorij sa slikama za analizu.

Tijekom predikcije:
- Pozadina slika bit će uklonjena, a slike će biti obrađene.
- Generirat će se Fisher vektori za svaku sliku.
- Predikcije će uključivati predviđenu klasu i top-N podudaranja.

---

## **Struktura podataka**

- **Dataset**: `./data/ATRW/`
- **Segmentirani podaci**: `./data/segmented_dataset/`
- **Trenirani modeli i značajke**:
  - PCA model: `./data/pca_model.pkl`
  - GMM model: `./data/gmm_model.pkl`
  - Fisher vektori: `./data/fisher_vectors.pkl`

---

## **Napomene**

- **Podrška za GPU**: Aplikacija koristi GPU za ubrzanje rada. Ako GPU nije dostupan, automatski će se koristiti CPU.

## **Rezultati**
Evaluacija modela pokazala je sljedeće rezultate na testnom skupu od 723 slike:

- **Točnost (Accuracy)**: 81.33%
- **Top-5 Točnost**: 89.35%

```
              precision    recall  f1-score   support

   accuracy                           0.81       723
  macro avg       0.82      0.79      0.78       723
weighted avg       0.83      0.81      0.81       723
```

---




Za dodatne informacije ili pomoć, obratite se na [kontakt](mailto:matej.maric99@gmail.com).
