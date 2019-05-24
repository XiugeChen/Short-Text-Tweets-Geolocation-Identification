## Results Summary

### Preprocessing

| Method          | Data Set | Original Size (word) | Reduced Size (word) | Time ()            |
| --------------- | -------- | -------------------- | ------------------- | ------------------ |
| 1. noStem       | Train    | 123822               | 92107               | 275.5418601036072  |
| 2. noRmStop     | Dev      | 60944                | 51566               | 54.84501504898071  |
| 3. noCheckSpell | Test     | 99722                | 96275               | 306.0435256958008  |
| 1. noStem       | Train    |                      | 91957               | 303.17842411994934 |
| 2. rmStop       | Dev      |                      | 51419               | 67.37629795074463  |
| 3. noCheckSpell | Test     |                      | 96123               | 334.8346338272095  |
| 1. stem         | Train    |                      | 59039               | 213.81338596343994 |
| 2. noRmStop     | Dev      |                      | 33629               | 55.65715718269348  |
| 3. noCheckSpell | Test     |                      | 60710               | 204.76995277404785 |
| 1. stem         | Train    |                      | 59036               | 203.4578869342804  |
| 2. rmStop       | Dev      |                      | 33622               | 59.74693298339844  |
| 3. noCheckSpell | Test     |                      | 60706               | 224.7171356678009  |
| 1. stem         | Train    |                      |                     |                    |
| 2. rmStop       | Dev      |                      |                     |                    |
| 3. checkSpell   | Test     |                      |                     |                    |

#### Just remove stop words

| Type  | Reduced Size | Time              |
| ----- | ------------ | ----------------- |
| Train | 72819        | 269.70441198349   |
| Dev   | 40808        | 53.82471513748169 |
| Test  | 74319        | 259.4161581993103 |

#### Remove stop words and stem

| Type  | Reduced Size | Time               |
| ----- | ------------ | ------------------ |
| Train | 58950        | 216.24050211906433 |
| Dev   | 33524        | 54.62634897232056  |
| Test  | 60619        | 236.47044587135315 |



#### Word Count Range

Only remove stop words: 1 - 9729

Remove stop words and stem: 



### Classify Accurance

#### Baseline

| Classifier          | Accurance           | Time                |
| ------------------- | ------------------- | ------------------- |
| Dummy_most_frequent | 0.25                | 0.1266167163848877  |
| Dummy_stratified    | 0.25176867831493194 | 0.14670515060424805 |

#### No preprocess + Mutal Information (without embedding)

| Classifier                | Top  | Accurance           | Time               |
| ------------------------- | ---- | ------------------- | ------------------ |
| MNB                       | 10   | 0.29491370993675636 | 0.6087250709533691 |
| LinearSVM                 | 10   | 0.2948869117804695  | 75.24470615386963  |
| LogiReg                   | 10   | 0.29491370993675636 | 47.14589810371399  |
| Ensemble Classifier(Hard) | 10   | 0.29491370993675636 | 98.37596607208252  |
| Random Forest             | 10   | 0.2946993246864616  | 0.6832027435302734 |
| XGBoost                   | 10   | 0.29464572837388786 | 26.57959222793579  |
| MNB                       | 50   | 0.30134526744559975 | 0.795403003692627  |
| LinearSVM                 | 50   | 0.2979151034408833  | 74.60034322738647  |
| Logi Reg                  | 50   | 0.30062171722585485 | 180.77869415283203 |
| Ensemble Classifier(Hard) | 50   | 0.3006485153821417  | 230.72853469848633 |
| Random Forest             | 50   | 0.300246543037839   | 3.7974488735198975 |
| XGBoost                   | 50   | 0.30105048772644444 | 133.95293712615967 |
| MNB                       | 100  | 0.3078304212670168  | 1.1424529552459717 |
| LinearSVM                 | 100  | 0.3038642941365634  | 134.71496891975403 |
| Logi Reg                  | 100  | 0.30405188123057136 | 384.2845768928528  |
| Ensemble Classifier(Hard) | 100  | 0.3039178904491371  | 473.57241797447205 |
| Random Forest             | 100  | 0.30247079000964733 | 10.399298667907715 |
| XGBoost                   | 100  | 0.3044270554185872  | 281.0866401195526  |

#### rmStop noStem noCheckSpell + WLH(without embedding) (p = 10, t = 2, len = 2)

| Classifier                | Top  | Size | Accurance           | Time               |
| ------------------------- | ---- | ---- | ------------------- | ------------------ |
| MNB                       | 10   | 199  | 0.27628899131739737 | 1.0421807765960693 |
| LinearSVM                 | 10   | 199  | 0.2762621931611105  | 3.8548407554626465 |
| Logi Reg                  | 10   | 199  | 0.27628899131739737 | 31.43122410774231  |
| Ensemble Classifier(Hard) | 10   | 199  | 0.27628899131739737 | 31.466897010803223 |
| Random Forest             | 10   | 199  | 0.2762621931611105  | 3.491147041320801  |
| XGBoost                   | 10   | 199  | 0.27521706506592347 | 151.30071711540222 |
| MNB                       | 30   |      |                     |                    |
| LinearSVM                 | 30   |      |                     |                    |
| Logi Reg                  | 30   |      |                     |                    |
| Ensemble Classifier(Hard) | 30   |      |                     |                    |
| Random Forest             | 30   |      |                     |                    |
| XGBoost                   | 30   |      |                     |                    |
| MNB                       | 60   | 1190 | 0.31785293171829776 | 4.886746883392334  |
| LinearSVM                 | 60   | 1190 | 0.31252009861721514 | 9.78895115852356   |
| Logi Reg                  | 60   | 1190 | 0.31265408939864936 | 236.72653675079346 |
| Ensemble Classifier(Hard) | 60   | 1190 | 0.3125736949297888  | 247.05391669273376 |
| Random Forest             | 60   | 1190 | 0.3126004930860757  | 66.00580978393555  |
| XGBoost                   | 60   | 1190 | 0.2978079108157359  | 1077.5234360694885 |
| MNB                       | 80   | 1586 | 0.3177725372494372  | 4.621535062789917  |
| LinearSVM                 | 80   | 1586 | 0.31444956586986816 | 9.473670244216919  |
| Logi Reg                  | 80   | 1586 | 0.3139671990567049  | 249.41507601737976 |
| Ensemble Classifier(Hard) | 80   | 1586 | 0.3141011898381391  | 254.31201601028442 |
| Random Forest             | 80   | 1586 | 0.31356522671240217 | 74.04016423225403  |
| XGBoost                   | 80   | 1586 | 0.29829027762889915 | 1056.800155878067  |
| MNB                       | 100  | 1982 | 0.31782613356201095 | 5.492473125457764  |
| LinearSVM                 | 100  | 1982 | 0.3163790331225212  | 10.40421175956726  |
| Logi Reg                  | 100  | 1982 | 0.31573587737163683 | 296.43895506858826 |
| Ensemble Classifier(Hard) | 100  | 1982 | 0.31603065709079214 | 309.34904980659485 |
| Random Forest             | 100  | 1982 | 0.3147175474327366  | 105.63942980766296 |
| XGBoost                   | 100  | 1982 | 0.2982098831600386  | 1319.1885120868683 |

#### rmStop stem noCheckSpell + WLH(without embedding, p = 10, t = 2, len = 2)

| Classifier                | Top  | Size | Accurance           | Time               |
| ------------------------- | ---- | ---- | ------------------- | ------------------ |
| MNB                       | 50   | 767  | 0.31766534462428986 | 3.22891902923584   |
| LinearSVM                 | 50   | 767  | 0.3157894736842105  | 7.691873788833618  |
| Logi Reg                  | 50   | 767  | 0.31616464787222637 | 117.06731295585632 |
| Ensemble Classifier(Hard) | 50   | 767  | 0.3162718404973738  | 150.25844478607178 |
| Random Forest             | 50   | 767  | 0.31573587737163683 | 33.177552223205566 |
| XGBoost                   | 50   | 767  | 0.30413227569943185 | 681.4222021102905  |

#### rmStop noStem noCheckSpell + WLH(p = 10, t = 1, len = 2)

| Classifier                | Top  | Size | Accurance          | Time              |
| ------------------------- | ---- | ---- | ------------------ | ----------------- |
| MNB                       | 10   | 805  | 0.2609872440776075 |                   |
| LinearSVM                 | 10   | 805  |                    |                   |
| Logi Reg                  | 10   | 805  |                    |                   |
| Ensemble Classifier(Hard) | 10   | 805  |                    |                   |
| Random Forest             | 10   | 805  |                    |                   |
| XGBoost                   | 10   | 805  |                    |                   |
| MNB                       | 30   | 2415 | 0.2611212348590417 | 52.67728519439697 |
| LinearSVM                 | 30   | 2415 | 0.2659181048343874 | 57.03435707092285 |
| Logi Reg                  | 30   | 2415 |                    |                   |
| Ensemble Classifier(Hard) | 30   | 2415 |                    |                   |
| Random Forest             | 30   | 2415 |                    |                   |
| XGBoost                   | 30   | 2415 |                    |                   |

#### rmStop noStem noCheckSpell + WLH(30%, p = 10, t = 1, len = 2) + Doc2vec Embedding

| Classifier                | Accurance          | Time              |
| ------------------------- | ------------------ | ----------------- |
| MNB                       | -                  | -                 |
| LinearSVM                 | 0.2982366813163254 | 563.2966248989105 |
| Logi Reg                  |                    |                   |
| Ensemble Classifier(Hard) |                    |                   |
| Random Forest             |                    |                   |
| XGBoost                   |                    |                   |

#### rmStop noStem noCheckSpell + WLH(10%, p = 10, t = 1, len = 2) + Doc2vec Embedding

| Classifier                | Accurance | Time |
| ------------------------- | --------- | ---- |
| MNB                       |           |      |
| LinearSVM                 |           |      |
| Logi Reg                  |           |      |
| Ensemble Classifier(Hard) |           |      |
| Random Forest             |           |      |
| XGBoost                   |           |      |

#### rmStop noStem + MY_WLH

| Classifier | Top  | Feature Size | Accurance | Time |
| ---------- | ---- | ------------ | --------- | ---- |
| MNB        | 1%   | 729          |           |      |
| LinearSVM  | 1%   | 729          |           |      |
| Logi Reg   | 1%   | 729          |           |      |



#### Predict Results on Kaggle

| Classifier | Methods                                          | Result  |
| ---------- | ------------------------------------------------ | ------- |
| MNB        | rmStop_noStem_noCheckSpell + WLH100%_noEmbedding | 0.30289 |
|            |                                                  |         |
|            |                                                  |         |

