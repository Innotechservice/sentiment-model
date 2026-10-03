# Sentiment Model

A small sentiment classifier (positive / negative) built from scratch with PyTorch. The trained model is served by a FastAPI service and used by a Streamlit web interface.

## Live demo

https://innotechservice.streamlit.app (the app sleeps after 12 hours without visitors, the first visit afterwards may take a moment)

## Results

- Model: embedding + mean pooling + small feed-forward network (about 1.3 million parameters), no pretrained weights
- Data: IMDB movie reviews (English), 20,000 train / 5,000 validation / 25,000 test
- Test accuracy: 85.5% (F1 about 0.85 for both classes)
- Training: about 10 seconds per epoch on CPU, early stopping on validation loss

## Limitations

- English only. The vocabulary comes from English reviews, so other languages (for example Azerbaijani) are not supported. The API returns reliable=false when most words are unknown.
- The model ignores word order. It can fail on negation ("not bad at all") and sarcasm.
- The confidence value is the model's internal score. It does not measure how likely the answer is to be correct.
- Only two classes (positive, negative). There is no neutral class.

## Setup (Windows)

Tested with Python 3.14.

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python src\dataset.py
    python src\vocab.py
    python src\train.py bag 20
    python src\evaluate.py bag

dataset.py downloads IMDB (about 80 MB) and creates the train/validation/test CSV files. vocab.py builds the vocabulary. train.py saves the best model to models\bag.pt.

## Run

Terminal 1 (API):

    venv\Scripts\activate
    python -m uvicorn api.main:app

Terminal 2 (web interface):

    venv\Scripts\activate
    python -m streamlit run app\ui.py

Open http://localhost:8501 for the interface and http://127.0.0.1:8000/docs for the API documentation.

## API

POST /predict with JSON {"text": "This movie was great"} returns label ("musbet" = positive, "menfi" = negative), confidence, known_words, total_words and reliable. GET /health checks that the service is running. Text length is limited to 2000 characters.

## Project structure

    api/main.py      FastAPI service
    app/ui.py        Streamlit interface
    src/dataset.py   download, cleaning and splitting of the data
    src/vocab.py     vocabulary, text encoding, PyTorch Dataset
    src/model.py     model architectures (SentimentBag, SentimentLSTM)
    src/train.py     training with early stopping
    src/evaluate.py  test-set metrics
    src/predict.py   prediction for new sentences

## Notes

SentimentLSTM is also included. On CPU it was too slow (about 50 minutes per epoch) and reached only about 59% validation accuracy after one epoch, so SentimentBag is the model in use. Training the LSTM on a GPU is a possible next step.

Planned: an Azerbaijani dataset and model.

