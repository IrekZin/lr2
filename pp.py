from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


# Исходные данные
categories = ['food', 'technology', 'politics', 'travel']
train_data = ['This is a article about food','pizza, sushi and burgers are very tasty' ,'This is a technology article.', 'This is a political article.', 'This is a travel article.']
train_labels = [0, 0, 1, 2, 3]

# Векторизация текста

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data)

# Обучение модели
model = LinearSVC()
model.fit(train_vectors, train_labels)


# Определение моделей Pydantic для запроса и ответа
class TextRequest(BaseModel):
    text: str


class CategoryResponse(BaseModel):
    category: str


class EvaluationResponse(BaseModel):
    report: str


# Создание экземпляра приложения FastAPI
app = FastAPI()


# Определение маршрутов API
@app.post("/predict", response_model=CategoryResponse)
def predict_category(text_request: TextRequest):
    new_text = text_request.text
    vectorized_text = vectorizer.transform([new_text])
    predicted_category = model.predict(vectorized_text)
    category = categories[predicted_category[0]]
    return {"category": category}


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_model():
    test_data = ['This is another food article.', 'This is another technology article.', 'This is another political article.', 'This is another travel article.']
    test_labels = [0, 1, 2, 3]
    test_vectors = vectorizer.transform(test_data)
    predicted_labels = model.predict(test_vectors)
    report = classification_report(test_labels, predicted_labels, target_names=categories)
    return {"report": report}
