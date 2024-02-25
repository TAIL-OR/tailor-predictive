### create database and get data
    ```
    cd scripts/
    python3 create_data.py

### how to run the models
    
    ```
    cd models/
    python3 run_model.py -m xgboost -nt 30 -hz 30

### Explicação dos Parâmetros:
* m: Define o modelo a ser usado para o treinamento. Pode ser xgboost ou lgbm.
* nt: Número de tentativas para encontrar os melhores hiperparâmetros durante o processo de otimização.
* hz: Define o horizonte de previsão desejado, ou seja, o número de períodos futuros que o modelo tentará prever.

