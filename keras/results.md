# Results 





Result without young clients on data trained without young clients
Kaggle : 0.0115578

Result with young clients on data trained without young clients
Kaggle : 0.0200227

Result with merged young and other clients predictions 
Kaggle : 0.0221373


#### Data:

- yearmonth_list = [201505, 201506]
- nb_clients = 100000

#### Model

- merged profiles with 'ave', sigmoid, binary_crossentropy, nadam, accuracy, 


**Conf:**
- Profiles : 0, 1, 8
- batch_size=1000

*Cross-validation results :* 

Nb epoch | Nb folds | Mean MAP@7 | STD(MAP@7)
--- | --- | --- | ---
150 | 5 | 0.030864 | 0.00731

Kaggle : 0.0213722




#### Data:

- yearmonth_list = [201505, 201506]
- nb_clients = max

#### Model

- merged profiles with 'ave', sigmoid, binary_crossentropy, nadam, accuracy, 


**Conf:**
- Profiles : 0, 1, 8
- batch_size=2000

*Cross-validation results :* 

Nb epoch | Nb folds | Min MAP@7 | Mean MAP@7 | Max MAP@7 | STD MAP@7
--- | --- | --- | --- | --- | ---
150 | 5 | 0.021769 | 0.030416 | 0.041625 | 0.00679 

Kaggle : 0.0209931
