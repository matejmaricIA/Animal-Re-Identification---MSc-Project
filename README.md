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

![Pregled rezultata](evaluations/visualizations/evaluation_comparison.png)

Pregled skupova podataka:
![Pregled skupova podataka](evaluations/visualizations/dataset_statistics.png)

### Tablica Rezultata

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./evaluations/visualizations/evaluation_table.md) -->
<!-- The below code snippet is automatically added from ./evaluations/visualizations/evaluation_table.md -->
```md
| Dataset | Accuracy | Top-N Accuracy | Weighted F1 | Samples | Classes |
|---|---|---|---|---|---|
| CowDataset | 0.8888888888888888 | 0.9528619528619529 | 0.8884844364762517 | 297 | 13 |
| DogFaceNet | 0.26385681293302543 | 0.3504618937644342 | 0.21888475603377755 | 1732 | 1393 |
| ELPephants | 0.09976798143851508 | 0.16937354988399073 | 0.08600154679040989 | 431 | 273 |
| HyenaID2022 | 0.33174603174603173 | 0.4873015873015873 | 0.31580712868306854 | 630 | 256 |
| GiraffeZebraID | 0.1377245508982036 | 0.33532934131736525 | 0.12158036758835163 | 1503 | 1304 |
| StripeSpotter | 0.8414634146341463 | 0.926829268292683 | 0.837094852580864 | 164 | 44 |
| SealID | 0.39568345323741005 | 0.6091127098321343 | 0.383574015394555 | 417 | 57 |
| BelugaID | 0.1816758026624902 | 0.3547376664056382 | 0.16231183769754298 | 1277 | 665 |
| Giraffes | 0.4626865671641791 | 0.6604477611940298 | 0.42478354978354976 | 268 | 178 |
| ATRW | 0.813953488372093 | 0.92 | 0.8130835574369896 | 1075 | 182 |
| IPanda50 | 0.4850909090909091 | 0.6690909090909091 | 0.4831979756512429 | 1375 | 50 |
| LionData | 0.09032258064516129 | 0.16774193548387098 | 0.07360983102918588 | 155 | 94 |
| AmvrakikosTurtles | 0.06 | 0.18 | 0.04333333333333333 | 50 | 50 |
| CZoo | 0.46808510638297873 | 0.6855791962174941 | 0.46248464263441236 | 423 | 24 |
| NyalaData | 0.06074766355140187 | 0.1822429906542056 | 0.054913829680184816 | 428 | 236 |
```
<!-- MARKDOWN-AUTO-DOCS:END -->


Za dodatne informacije ili pomoć, obratite se na [kontakt](mailto:matej.maric99@gmail.com).
