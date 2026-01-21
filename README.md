# IT Service Ticket Classifier

LSTM-нейросеть для автоматической классификации IT-тикетов по 8 категориям.

## Описание

Модель анализирует текст обращения в IT-поддержку и определяет его категорию:
- **Access** — проблемы с доступом и паролями
- **Administrative rights** — запросы на права администратора
- **HR Support** — кадровые вопросы
- **Hardware** — проблемы с оборудованием
- **Internal Project** — внутренние проекты
- **Miscellaneous** — прочее
- **Purchase** — закупки
- **Storage** — хранилище данных

## Результаты

- **Accuracy:** 82%
- **Macro F1-Score:** 0.82

### Classification Report

| Категория | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Access | 0.88 | 0.88 | 0.88 |
| Administrative rights | 0.65 | 0.81 | 0.72 |
| HR Support | 0.86 | 0.79 | 0.83 |
| Hardware | 0.84 | 0.76 | 0.80 |
| Internal Project | 0.74 | 0.90 | 0.81 |
| Miscellaneous | 0.76 | 0.82 | 0.79 |
| Purchase | 0.83 | 0.93 | 0.88 |
| Storage | 0.81 | 0.91 | 0.86 |

## Архитектура модели

- **Embedding:** 100-мерные векторы слов
- **LSTM:** 128 units с dropout 0.3
- **Dense:** 64 нейрона (ReLU) → 8 классов (Softmax)
- **Оптимизатор:** Adam (lr=0.001)
- **Loss:** Categorical Crossentropy

## Датасет

[IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset) — 47,837 тикетов.

## Установка и запуск

```bash
# Клонирование репозитория
git clone https://github.com/SmoukZ/nlp-word-processor.git
cd nlp-word-processor

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или venv\Scripts\activate  # Windows

# Установка зависимостей
pip install tensorflow keras pandas numpy matplotlib seaborn scikit-learn nltk kagglehub

# Запуск ноутбука
jupyter notebook main.ipynb
```

## Пример использования

```python
text = "My laptop screen is broken and I need a replacement"
predicted_class, confidence = predict_ticket(text)
# Результат: Hardware (98.82%)
```

## Технологии

- Python 3.11
- TensorFlow / Keras
- NLTK
- Scikit-learn
- Pandas, NumPy, Matplotlib
