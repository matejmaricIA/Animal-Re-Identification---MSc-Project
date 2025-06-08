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
python main.py --train --ds ATRW --save_eval
```

- **`--train`**: Pokreće aplikaciju u načinu treniranja.
- **`--ds`**: Specificira dataset koji će se koristiti za treniranje (npr. `ATRW`).
- **`--save_eval`**: Sprema rezultate evaluacije u direktorij `data/evaluations`.
 **`--version`**: Oznaka verzije metode koja se koristi. U kombinaciji s
  postavkama uklanjanja pozadine i tone mappinga kreira se "tag" prema kojem se
  rezultati spremaju u poddirektorij unutar `evaluations/`.

  
Tijekom treniranja:
- Podaci će biti podijeljeni u trening i test skupove.
- Modeli **PCA** i **GMM** bit će istrenirani na značajkama.
- Evaluacija će prikazati točnost i top-N točnost na testnom skupu.

Rezultati treniranja bit će spremljeni u definirane direktorije u projektu.

### **2. Inferencija (predikcija)**

Za izvođenje predikcija na novim slikama koristite naredbu:
```bash
python main.py --predict --ds ATRW --image_location /path/to/dir
```

- **`--predict`**: Omogućuje način rada za predikciju.
- **`--ds`**: Dataset korišten za treniranje na kojem je bazirana baza podataka.
- **`--image_location`**: Specificira direktorij sa slikama za analizu.

Tijekom predikcije:
- Pozadina slika bit će uklonjena, a slike će biti obrađene.
- Generirat će se Fisher vektori za svaku sliku.
- Predikcije će uključivati predviđenu klasu i top-N podudaranja.

---

## **Struktura podataka**

- **Dataset**: `./data/<IME_DATASETA>/`
- **Segmentirani podaci**: `./data/<IME_DATASETA>/segmented_dataset/`
- **Trenirani modeli i značajke**:
  - PCA model: `./data/<IME_DATASETA>/pca_model.pkl`
  - GMM model: `./data/<IME_DATASETA>/gmm_model.pkl`
  - Fisher vektori: `./data/<IME_DATASETA>/fisher_vectors.pkl`

---

## **Napomene**

- **Podrška za GPU**: Aplikacija koristi GPU za ubrzanje rada. Ako GPU nije dostupan, automatski će se koristiti CPU.

## **Rezultati**

Pregled rezultata na testiranim skupovima podataka:

![Pregled rezultata](evaluations/version_1_backg_rem_False_tone_mapping_False_closed/visualizations/evaluation_comparison.png)

Pregled skupova podataka:
![Pregled skupova podataka](evaluations/version_1_backg_rem_False_tone_mapping_False_closed/visualizations/dataset_statistics.png)

### Tablica Rezultata

![Rezultati evaluacije](evaluations/version_1_backg_rem_False_tone_mapping_False_closed/visualizations/results_table.png)

Za dodatne informacije ili pomoć, obratite se na [kontakt](mailto:matej.maric99@gmail.com).
