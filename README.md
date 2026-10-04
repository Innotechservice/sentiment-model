# Sentiment Model

A small sentiment classifier (positive / negative) built from scratch with PyTorch. It supports English and Azerbaijani. The trained models are served by a FastAPI service and used by a Streamlit web interface.

## Live demo

https://innotechservice.streamlit.app (the app sleeps after 12 hours without visitors, the first visit afterwards may take a moment). Choose the language (English or Azerbaijani) at the top of the page.

## Results

Both languages use the same architecture: embedding + mean pooling + small feed-forward network, no pretrained weights. Each language has its own vocabulary and its own trained model.

| Language | Data | Test accuracy |
|---|---|---|
| English | IMDB movie reviews: 20,000 train / 5,000 validation / 25,000 test | 85.5% (F1 about 0.85 for both classes) |
| Azerbaijani | Azerbaijani reviews: 2,925 test reviews | 82.7% (validation accuracy 82.2%) |

- English model: about 1.3 million parameters, about 10 seconds per epoch on CPU, early stopping on validation loss.
- Azerbaijani model: confusion matrix on the test set: 1258 negative and 1160 positive reviews correct, 205 negative predicted as positive, 302 positive predicted as negative.

## Limitations

- Each model understands only its own language. There is no automatic language detection, the language is chosen by the user. The API returns reliable=false when most words are unknown, for example when English text is sent to the Azerbaijani model.
- The models ignore word order. They can fail on negation ("not bad at all") and sarcasm.
- The confidence value is the model's internal score. It does not measure how likely the answer is to be correct.
- Only two classes (positive, negative). Neutral sentences such as "the price is average" still get one of the two labels.
- The Azerbaijani model is less accurate than the English one (82.7% vs 85.5%).

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

The trained models are included in the repository: models\bag.pt and models\vocab.json (English), models\az_bag.pt and models\az_vocab.json (Azerbaijani). To check the Azerbaijani model on its test set:

    python src\az_evaluate.py

## Run

Terminal 1 (API):

    venv\Scripts\activate
    python -m uvicorn api.main:app

Terminal 2 (web interface that uses the API):

    venv\Scripts\activate
    python -m streamlit run app\ui.py

Open http://localhost:8501 for the interface and http://127.0.0.1:8000/docs for the API documentation.

Alternative without the API: app\cloud_ui.py loads both models directly.

    python -m streamlit run app\cloud_ui.py

Command line:

    python src\predict.py "This movie was great"
    python src\predict.py --lang az "your sentence"

On Windows PowerShell, letters such as ə, ı, ş and ç may be garbled when typed in the command line. Use the web interface or the API for Azerbaijani text.

## API

POST /predict with JSON {"text": "This movie was great", "lang": "en"} returns label ("musbet" = positive, "menfi" = negative), confidence, known_words, total_words, reliable and lang. The lang field is optional: "en" (default) or "az". GET /health checks that the service is running and lists the loaded languages. Text length is limited to 2000 characters.

Example for Azerbaijani: {"text": "your sentence", "lang": "az"}

## Project structure

    api/main.py          FastAPI service (both languages)
    app/ui.py            Streamlit interface that calls the API
    app/cloud_ui.py      Streamlit interface that loads the models directly
    src/dataset.py       download, cleaning and splitting of the English data
    src/vocab.py         vocabulary, text encoding, PyTorch Dataset
    src/model.py         model architectures (SentimentBag, SentimentLSTM)
    src/train.py         training with early stopping
    src/evaluate.py      test-set metrics (English)
    src/predict.py       prediction for new sentences (en and az)
    src/az_data.py       preparation of the Azerbaijani data
    src/az_train.py      training of the Azerbaijani model
    src/az_evaluate.py   test-set metrics and error samples (Azerbaijani)
    src/az_inspect.py    inspection of the Azerbaijani data

## Notes

SentimentLSTM is also included. On CPU it was too slow (about 50 minutes per epoch) and reached only about 59% validation accuracy after one epoch, so SentimentBag is the model in use. Training the LSTM on a GPU is a possible next step.

Planned: improve the Azerbaijani model by studying its errors, and add a neutral class.