# Defect-detection-in-pipelines

# Detecção do Defeito "Ruptura" em Dutos com CNN

## 1. **Importação de Bibliotecas**
O código importa diversas bibliotecas para realizar o treinamento de um modelo de rede neural convolucional (CNN) para classificar imagens com defeitos de pipeline:

- **os**: Para manipulação de arquivos e diretórios.
- **numpy**: Para manipulação de arrays numéricos.
- **tensorflow.keras**: Para criação e treinamento de modelos de deep learning.
- **sklearn.model_selection**: Para dividir os dados em conjuntos de treinamento, validação e teste.
- **PIL**: Para manipulação e carregamento das imagens.
- **tensorflow.keras.callbacks**: Para definir callbacks (ex: early stopping) durante o treinamento.
- **tensorflow.keras.optimizers**: Para definir o otimizador do modelo.
- **tensorflow.keras.losses**: Para definir a função de perda do modelo.
- **tensorflow.keras.preprocessing.image**: Para realizar o data augmentation (aumento de dados).

## 2. **Função `f1_score_metric`**
A função `f1_score_metric` é definida para calcular a métrica F1-Score diretamente durante o treinamento do modelo. A métrica F1-Score combina precisão e recall de forma harmônica, sendo útil em cenários de classificação binária (como a detecção do defeito "Ruptura").

### Explicação do código:
- **`y_true` e `y_pred`**: São as variáveis que representam os valores reais e preditos das classes.
- **`K.cast`**: Garante que tanto `y_true` quanto `y_pred` sejam do tipo `float32`, o que é necessário para os cálculos.
- **`K.mean`**: Calcula a média dos valores ao longo das dimensões da variável, implementando o cálculo do F1-Score com a fórmula:
  \[
  F1 = \frac{2 \times (Precision \times Recall)}{Precision + Recall}
  \]
  Usando a função `K.epsilon()` para evitar divisão por zero.

## 3. **Carregamento e Processamento das Imagens**
A função `load_and_process_images` carrega as imagens e seus respectivos rótulos (labels) de defeitos a partir de arquivos no disco.

### Explicação do código:
- **Leitura das imagens**: O código lê as imagens de um diretório especificado (`images_path`), redimensionando-as para o tamanho definido em `img_size` (no caso, 256x256 pixels).
- **Normalização**: As imagens são convertidas para arrays NumPy e normalizadas (valores de pixels são divididos por 255 para ficarem no intervalo [0, 1]).
- **Leitura dos rótulos**: O arquivo de rótulo (um arquivo `.txt` associado à imagem) é lido para verificar se a classe do defeito é "Ruptura". Se for, o rótulo é marcado como `1`, caso contrário, `0`.

## 4. **Divisão dos Dados**
Os dados (imagens e rótulos) são divididos em três conjuntos: treinamento, validação e teste, utilizando a função `train_test_split` do `sklearn`.

### Explicação do código:
- **Divisão treino/validação/teste**:
  - **70% para treino**: A maior parte dos dados é usada para treinamento.
  - **15% para validação**: Usado para ajustar parâmetros do modelo durante o treinamento.
  - **15% para teste**: Usado para avaliar o desempenho do modelo após o treinamento.
  
  A divisão é feita de maneira estratificada para garantir que a distribuição das classes (especialmente a classe "Ruptura") seja equilibrada em todos os conjuntos.

## 5. **Data Augmentation**
A técnica de **Data Augmentation** é usada para aumentar a diversidade do conjunto de treinamento aplicando transformações aleatórias nas imagens.

### Explicação do código:
- **Transformações para aumento de dados**:
  - **Flip horizontal** (`horizontal_flip=True`)
  - **Rotação aleatória** (`rotation_range=20`)
  - **Deslocamento horizontal e vertical** (`width_shift_range=0.1`, `height_shift_range=0.1`)

  Essas transformações ajudam o modelo a generalizar melhor, evitando overfitting ao forçar o modelo a aprender características invariantes das imagens.

## 6. **Modelo CNN**
Um modelo de rede neural convolucional (CNN) é definido utilizando a API Keras.

### Explicação do código:
- **Camadas convolucionais**:
  - As camadas `Conv2D` aplicam filtros convolucionais para extrair características das imagens.
  - **BatchNormalization** é usada após cada camada convolucional para estabilizar o aprendizado.
  - **MaxPooling2D** reduz a dimensionalidade das imagens, mantendo apenas as características mais importantes.
  - **Dropout** é usado para evitar overfitting, desligando aleatoriamente neurônios durante o treinamento.
  
- **Camada densa**:
  - Após as camadas convolucionais, a rede é achatada com a camada `Flatten` e uma camada densa é adicionada para realizar a classificação.

- **Saída**:
  - A camada final (`Dense(1, activation='sigmoid')`) tem uma única unidade com função de ativação `sigmoid`, adequada para problemas de classificação binária (defeito "Ruptura" ou não).

## 7. **Compilação do Modelo**
O modelo é compilado com o **otimizador Adam**, a **função de perda BinaryCrossentropy** (apropriada para problemas de classificação binária) e as **métricas de avaliação**: precisão, recall, e F1-Score.

### Explicação do código:
- **Optimizer**: O otimizador Adam é utilizado com uma taxa de aprendizado pequena (`0.0001`), que é ideal para fine-tuning de modelos complexos.
- **Loss function**: A função de perda `BinaryCrossentropy` é adequada para problemas de classificação binária.
- **Métricas**: As métricas de avaliação incluem:
  - **Accuracy**: Precisão geral do modelo.
  - **Precision**: Precisão da classe positiva (Ruptura).
  - **Recall**: Sensibilidade, ou a capacidade de identificar corretamente as instâncias positivas.
  - **F1-Score**: Métrica combinada que leva em conta tanto a precisão quanto o recall.

## 8. **Callbacks**
O callback **EarlyStopping** é utilizado para interromper o treinamento caso o modelo não melhore após um número específico de épocas (`patience=5`).

### Explicação do código:
- **EarlyStopping**: Monitora a `val_loss` (perda de validação) e interrompe o treinamento se o modelo não melhorar após 5 épocas, restaurando os melhores pesos encontrados durante o treinamento.

## 9. **Treinamento do Modelo**
O modelo é treinado utilizando o método `fit`, com os dados de treinamento e validação fornecidos pelos geradores de imagens.

### Explicação do código:
- **Treinamento**: O treinamento é realizado por até 20 épocas, utilizando o gerador de dados de treinamento e validação. O modelo será ajustado com base na perda de validação e nas métricas de avaliação.

## 10. **Avaliação no Conjunto de Teste**
Após o treinamento, o modelo é avaliado no conjunto de teste para verificar seu desempenho final.

### Explicação do código:
- **`model.evaluate`**: A avaliação é realizada utilizando o gerador de dados de teste, que fornece as imagens e os rótulos correspondentes.
- **Exibição das Métricas**: As métricas de avaliação (perda, precisão, recall e F1-Score) são exibidas após a avaliação do modelo.