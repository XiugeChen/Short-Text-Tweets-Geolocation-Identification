## Results Summary

### Preprocessing

| Method          | Data Set | Original Size (word) | Reduced Size (word) | Time ()            |
| --------------- | -------- | -------------------- | ------------------- | ------------------ |
| 1. noStem       | Train    | 94838                | 94838               | 506.6164360046387  |
| 2. noRmStop     | Dev      | 53190                | 53190               | 105.26869606971741 |
| 3. noCheckSpell | Test     | 99722                | 99722               | 575.5326058864594  |
|                 |          |                      |                     |                    |
|                 |          |                      |                     |                    |
|                 |          |                      |                     |                    |

### Classify Accurance

| Classifier          | Preprocess Method              | Feature Eng Method  | Accurance           | Time                |
| ------------------- | ------------------------------ | ------------------- | ------------------- | ------------------- |
| Dummy_most_frequent | noRmStop, noStem, noCheckSpell | mi, 10, noEmbedding | 0.25                | 0.1266167163848877  |
| Dummy_stratified    |                                |                     | 0.2506699539071712  | 0.14670515060424805 |
| MNB                 |                                |                     | 0.29491370993675636 | 0.6087250709533691  |
| LinearSVM           |                                |                     | 0.2948869117804695  | 75.24470615386963   |
| Logistic Regression |                                |                     | 0.29491370993675636 | 47.14589810371399   |
|                     |                                |                     |                     |                     |

