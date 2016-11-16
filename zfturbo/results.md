# Various results 


## First tests of zfturbo_script_mass_hashes_personal_recommendations.py

### Test 1:
**Dataset configuration :**
    - train : 201501-201502 (full), validation : 201503 (full)

**Options :**
    - `proba = values / nb_months`
    
    - `COMMON_RECOMMENDATIONS_WEIGHT = 0.5`

**Results:**
INFO:root:Predicted score: 0.0208096529837


### Test 2:

**Dataset configuration :**
    - train : 201501-201502 (full), validation : 201503 (full)

**Options :**
    - `proba = np.array(values) / (2.0 * nb_months) + 0.5`
    
    - `COMMON_RECOMMENDATIONS_WEIGHT = 0.5`

**Results:**
INFO:root:Predicted score: 0.0213302407694


### Test 3:

**Dataset configuration :**
    - train : 201501-201502 (250k), validation : 201503 (250k)

**Options : **
    - `proba = np.array(values) / (2.0 * nb_months) + 0.5`
    
    - `COMMON_RECOMMENDATIONS_WEIGHT = 0.5`

**Results:**
INFO:root:Predicted score: 0.548605911347 ?????



### Test 4:

**Dataset configuration :**
    - train : 201501-201505 (full), validation : 201507 (full)

**Options : **
    - `proba = np.array(values) / (2.0 * nb_months) + 0.5`
        
    - `COMMON_RECOMMENDATIONS_WEIGHT = 0.5`

**Results:**
INFO:root:Predicted score: 0.040358012212


### Test 5:

**Dataset configuration :**
    - train : 201505-201508 (full), validation : 201511 (full)

**Options :**
    - `proba = np.array(values) / (2.0 * nb_months) + 0.5`
        
    - `COMMON_RECOMMENDATIONS_WEIGHT = 0.5`

**Results:**
INFO:root:Predicted score: 0.0705896503752


### Test 6:

**Dataset configuration :**
    - train : 201505-201509 (full), validation : 201511 (full)

**Options :**
    - `proba = np.array(values) / (2.0 * nb_months) + 0.5`

    - `COMMON_RECOMMENDATIONS_WEIGHT = 0.5`

**Results:**
INFO:root:Predicted score: 0.0495554051219
Kaggle: 0.0161563


### Test 7:

**Dataset configuration :**
    - train : 201501-201604 (full), validation : 201605 (full)

**Options :**
    - `proba = np.array(values) / (2.0 * nb_months) + 0.5`

    - `COMMON_RECOMMENDATIONS_WEIGHT = 0.5`

**Results:**
INFO:root:Predicted score: 0.0172339792323
Kaggle: 0.0163384


## Tests on reduced dataset
### Test 1:

**Dataset configuration :**
    - train : 201601-201604 (full), validation : 201605 (full)

**Options :**
    - `proba = np.array(values) / (2.0 * nb_months) + 0.5`

    - `COMMON_RECOMMENDATIONS_WEIGHT = 0.5`

**Results:**
INFO:root:Predicted score: 0.0166102609575
Kaggle: 0.0163645

### Test 2:

**Dataset configuration :**
    - train : 201601-201604 (full), validation : 201605 (full)

**Options :**
    - `proba = np.array(values) / (2.0 * nb_months) + 0.5`

    - `COMMON_RECOMMENDATIONS_WEIGHT = 1.0`

**Results:**
INFO:root:Predicted score: 0.0181385522945

### Test 3
```
INFO:root:-- predict_score : personal_recommendations_weight=1.0
INFO:root:--- predict_score : map7=0.0134817770562

INFO:root:-- predict_score : personal_recommendations_weight=0.1
INFO:root:--- predict_score : map7=0.0178552748385

INFO:root:-- predict_score : personal_recommendations_weight=0.2
INFO:root:--- predict_score : map7=0.0173115588153

INFO:root:-- predict_score : personal_recommendations_weight=0.3
INFO:root:--- predict_score : map7=0.016741842923
```

### Test 4
15/11/2016 : Fix bugs etc
201601-201604
DEBUG:root:-- predict_score : personal_recommendations_weight=0.7
DEBUG:root:--- predict_score : map7=0.0220622619839
INFO:root:Predicted score: 0.0220622619839
Kaggle: 0.0162049

### Test 5
201501-201604
DEBUG:root:-- predict_score : personal_recommendations_weight=0.7
DEBUG:root:--- predict_score : map7=0.022
INFO:root:Predicted score: 0.022
Kaggle: 0.0147564

### Test 6
201501-201604 : Only ZFTurbo
DEBUG:root:-- predict_score : personal_recommendations_weight=0.0
DEBUG:root:--- predict_score : map7=0.0215788389906
INFO:root:Predicted score: 0.0215788389906