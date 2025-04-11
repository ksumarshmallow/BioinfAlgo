# BioinfAlgo
Добро пожаловать в репозиторий **BioinfAlgo**! Здесь будут реализованы домашние работы В рамках курса "Алгоритмы в биоинформатике" 🚀

---
## 📂 Структура проекта
```plaintext
BioinfAlgo/
├── hw1/                    # Домашняя работа 1: Алгоритм Витерби для химерных последовательностей
│   ├── src/                # Исходные скрипты
│   │   ├── sequence_processing.py
│   │   ├── chimera_generator.py
│   │   ├── hmm.py
│   │   └── viterbi.py
│   ├── results/            # Результаты работы скриптов
│   ├── run_chimera.py      # Пайплайн генерации химерной последовательности
│   ├── run_viterbi.py      # Запуск алгоритма Витерби
│   ├── config.yaml         # Конфиг для генерации химерной последовательности
│   ├── config_viterbi.yaml # Конфиг для Витерби
│   └── requirements.txt    # Зависимости для ДЗ1
├── hw2/                    # TBD
├── hw3/                    # TBD
├── data/                   # Входные данные
└── utils/                  # Вспомогательные модули
    ├── cli_argument_parser.py
    ├── config_parser.py
    └── logging.py
```
---
## 🛠️ Начало работы
1. **Клонируйте репозиторий**:
```bash
git clone https://github.com/ksumarshmallow/BioinfAlgo.git
cd BioinfAlgo
```

2. **Создайте виртуальное окружение** (рекомендуется Python 3.12):

```bash
python3.12 -m venv .venv
```

3. **Активируйте виртуальное окружение**
    - Для Linux/MacOS
    ```bash
    source .venv/bin/activate
    ```
    - Для Windows:
    ```bash
    .venv\Scripts\activate
    ```

4. **Установите зависимости**. Зависимости устанавливаются *отдельно* для каждой домашей работы. Учтите это при наименовании виртуального окружения (п.2 и 3). Например, утановка зависимостей для ДЗ1 (папка `hw1`):

```bash
pip install -r hw1/requirements.txt
```

## 📌HW1. Viterbi Algorithm
Возможно два ~~стула~~ варианта:
1. **Полный пайплайн**: Генерация химерной последовательности из исходных двух, определение параметров HMM, поиск наиболее вероятного пути состояний (алгоритм Витерби) и вычисление Error Rate.
2. **Только определение наиболее вероятного пути состояний** для последовательности. Должна быть задана матрица эмиссий!

---

### **Вариант 1: Генерация химерной последовательности** (Генерация химеры + Витерби)

#### 🚀 Запуск через CLI

Для выполнения пайплайна генерации химеры выполните команду из корня проекта:

```bash
python hw1/run_chimera.py \
  --seq1 data/seq1.fasta \
  --seq2 data/seq2.fasta \
  --config hw1/config.yaml
```

- `--seq1`: путь к первой последовательности (формат `fasta` или `txt`).
- `--seq2`: путь ко второй последовательности.
- `--config`: путь к конфигурационному файлу.

#### ⚙️ Конфигурационный файл (`config.yaml`)

```yaml
valid_nucleotides: ['A', 'T', 'G', 'C']

ChimeraGenerator:
  mean_fragment_length: 300     # Средняя длина фрагмента (нуклеотидов)
  max_seqlen: 10000             # Длина итоговой химерной последовательности
  seed: 42                      # Seed для воспроизводимости
  param_folder: 'hw1/params'    # Папка для сохранения матрицы эмиссий HMM
  output_folder: 'data'         # Папка для сохранения результатов
  output_name: 'chimera_seq'    # Базовое имя файлов вывода
  save_format: 'fasta'          # Формат сохранения химерной последовательности (.fasta или .txt)
```


#### 🔍 Логика работы

1. **Обработка входных данных**:
   - Фильтрация невалидных нуклеотидов.
   - Автоматическое определение формата файла (FASTA/TXT).

2. **Генерация химеры**:
   - Случайное чередование фрагментов из `seq1` и `seq2`.
   - Контроль длины и числа фрагментов через параметры.

3. **Построение HMM**:
   - **Переходная матрица**: Вероятности переключения между состояниями (seq1 ↔ seq2).
   - **Эмиссионная матрица**: Вероятности нуклеотидов для каждого состояния.
   - **Стационарное распределение**: Расчёт через собственные векторы.

4. **Алгоритм Витерби**:
   - Восстановление наиболее вероятного пути состояний.
   - Расчёт **Error Rate**: Доля ошибочных предсказаний.

#### 📊 Результаты
| Файл                     | Описание                          |
|--------------------------|-----------------------------------|
| `chimera_seq.fasta`      | Химерная последовательность       |
| `chimera_seq_states.txt` | Истинные метки состояний          |
| `chimera_seq_viterbi.txt`| Предсказанные метки Витерби       |
| `emission_matrix.npy`    | Эмиссионная матрица HMM (NumPy)   |
| `summary.txt`            | Статистика по химере и качеству   |


---

### 2. Запуск только алгоритма Витерби

Если вам необходимо протестировать работу алгоритма Витерби с готовыми матрицами переходов и эмиссий, воспользуйтесь скриптом `run_viterbi.py`.

#### 🚀 Запуск через CLI

```bash
python hw1/run_viterbi.py \
  --seq data/chimera_seq.fasta \
  --output results/predicted_states.txt \
  --config hw1/config_viterbi.yaml
```

- `--seq`: путь к последовательности, для которой необходимо предсказать путь состояний.
- `--output`: путь для сохранения предсказанного пути (текстовый файл).
- `--config`: путь к конфигурационному файлу.

#### ⚙️Конфигурационный файл (`config_viterbi.yaml`)

```yaml
# Порядок нуклеотидов для матрицы эмиссий
valid_nucleotides: ['A', 'T', 'G', 'C']
# Среднее число шагов в каждом из состояний
mean_steps: 300
# Эмиссионная матрица
emission_matrix_path: 'hw1/params/emission_matrix.npy'
```

  - `emission_matrix_path` указывает путь к сохраненной ранее эмиссионной матрице. Либо можно задать в файле собственную матрицу. 
  - Порядок нуклеотидов в `valid_nucleotides` должен соответствовать порядку колонок в эмиссионной матрице.
  - Переходная матрица генерируется таким образом, что среднее число шагов равно параметру `mean_steps`.

После выполнения скрипта алгоритм Витерби сохранит предсказанный путь состояний для каждого валидного нуклеотида из входной последовательности в указанный в аргументах файл.



