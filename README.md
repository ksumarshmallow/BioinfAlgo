# BioinfAlgo

## HW1
```markdown
hw1/
├── src/
│   ├── __init__.py
│   ├── sequence_processing.py  # обработка последовательностей
│   └── chimera_generator.py    # генерация химерной последовательности
│   └── hmm.py                  # определение модели HMM
│   └── viterbi_runner.py       # логика запуска Витерби
├── run_chimera.py          # запуск всего пайплайна 
└── run_viterbi.py          # запук Витерби с готовыми параметрами HMM
```

### Главный файл
Файл `run_chimera.py` содержит основную логику домашней работы:
1. **Процессинг файлов с последовательностями**: или `fasta`-файлов, или из `.txt`. В любом случае в последовательности остаются только нуклеотиды, указанные в конфигурационном файле (`hw1/config.yaml`). По умолчанию - **A**, **T**, **G**, **C**
2. **Генерация химерной последовательности** из процессированных последовательностей из п.1. Длина последовательности и средняя длина фрагмента указываются в конфигурационном файле. По умолчанию средняя длина фрагмента - 300 н., длина последовательности - 10 т.н.
3. **Построение HMM-модели** - определение матрицы эмиссий и переходов. 
    - Матрица переходов создается таким образом, что среднее число шагов в каждом из состояний равно средней длине фрагмента. 
    - Матрица эмиссий создается таким образом, что вероятность эмиссии для пиримидинов равна усредненной частоте пиримидинов в соответствующей последовательности. Аналогично для пуринов.
    - Изначальное стационарное состояние ищется как собственный вектор матрицы переходов, соответствующий собственному значению 1. При равномерном выборе последовательностей в генерации химерной последовательности, стационарное состояние также будет равномерным.
4. **Витерби-алгоритм**: поиск наиболее вероятного пути, с помощью которого была сгенерирована химерная последовательность. 

**Выполнение кода через CLI:**
```bash
python hw1/run_chimera.py --seq1 <path_to_seq1> --seq2 <path_to_seq2> --config <path_to_config>
```

Выполняется из корня проекта!
- `--seq1` - указывается путь к первой последовательности, из фрагментов которой будет генерироваться химерная последовательность. Формат `fasta` или `txt`.
- `--seq2` - указывается путь ко второй последовательности, из фрагментов которой будет генерироваться химерная последовательность. Формат `fasta` или `txt`.
- `--config` - указываетя путь к конфигурационному файлу. Пример в данном репозитории находится по относительному пути `hw1/config.yaml`


**Пример конфигурационного файла:**

```yaml
GeneralParams:
  num_sequences: 2
  valid_nucleotides: ['A', 'T', 'G', 'C']

ChimeraGenerator:
  mean_fragment_length: 300
  max_seqlen: 10000
  seed: 42
  output_folder: 'data'
  output_name: 'chimera_seq'
  save_format: 'fasta'
```
- `mean_fragment_length` - средняя длина фрагмента. Длина фрагмента генерируется из экспоненциального распределения с заданным средним. По умолчанию - 300 н.
- `max_seqlen` - длина выходной химерной последовательности. По умолчанию - 10 тысяч н.
- `seed` - для воспроизводимости результатов. 
- `output_folder` - название папки, в которую будет сохранена:
    - Сгенерированная химерная последовательность - с названием `<output_name>.<save_format>`. Возможны два формата сохранения: `fasta` (по умолчанию) и `txt`
    - Метки для каждого нуклеотида: из какой последовательности (`seq1` или `seq2`) он пришел. Это последовательность длиной `max_seqlen`, состоящая из **1** и **2** (в зависимости от того, соответствует нуклеотид фрагменту из `seq1` или `seq2 `). Сохраняется с названием `<output_name>_states.txt`

**Результат работы программы**
- Файл с химерной последовательностью. Сохраняется в папке `output_folder`, с названием `<output_name>.<save_format>`
- Файл с маппингом нуклеотидов химерной последовательности и их состояниями (из каких последовательностей пришли). Сохраняется в папке `output_folder`, с названием `<output_name>_states.txt`
- Файл с маппингом нуклеотидов химерной последовательности и их **предсказанными** Витерби состояниями. Сохраняется в папке `output_folder`, с названием `<output_name>_states_viterbi.txt`
- **Параметры HMM** - `hmm_params.pkl`. Представляет из себя словарь:
    - Множества состояний и наблюдений: `states_set` и `observations_set`
    - Матрицы переходов и эмиссий: `transition_matrix` и `emission_matrix`
    - Стационарное распределение: `pi`
    - Перевод закодированных состояний и наблюдений в "реальные": `mappings` (`obs2idx`, `idx2obs`, `state2idx`, `idx2state`)
- **Саммари** - файл `summary.txt` будет содержать:
    - **Информация о химерной последовательности**:
        - Изначальные частоты нуклеотидов в оригинальных последовательностях `seq1` и `seq2`. 
        - Частоты нуклеотидов в химерной последовательности
        - Сколько нуклеотидов (асболютно и отноительно в %) содержит химерная последовательности из оригинальных последоватестей
        - Средняя длина фрагмента в химерной последовательности
        - Количество фрагментов, из которых состоит химерная последовательность
    - **Информация о HMM**:
        - Матрица переходов
        - Матрица эмиссий
        - Стационарное распределение
    - **Качество лучшего пути, определенного Витерби**
        - Error rate - в какой доле позиций мы ошиблись в определении "источника" нуклеотида
        - Считается для химерной последовательности
        - Считается доверительный интервал для error rate для рандомных фрагментов из изначальных последовательностей: семплируем X фрагментов из каждой послдеовательности, для них определяем Витерби (по определенным ранее матрице эмиссий и перехов) -> считаем error rate -> так делаем X раз -> считаем дов интервал (перцентильный)


### Запуск Витерби
Можно проверить работу Витерби с готовыми матрицами переходов и эмиссий.
```bash
    python run_viterbi.py --seq <path_to_sequence> --hmm_params <path_to
```
