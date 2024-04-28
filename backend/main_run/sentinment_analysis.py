from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import flair
import pandas as pd
import nltk
nltk.download('vader_lexicon')

def sentiment_analyzer(symbol):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    tickers = [symbol]
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')
    sentiment_positive_count = 0
    sentiment_negative_count = 0
    sentiment_positive_sum = 0
    sentiment_negative_sum = 0
    news_tables = {}
    for ticker in tickers:
        url = finviz_url + ticker
        req = Request(url=url, headers={'user-agent': 'my-app'})
        response = urlopen(req)
        html = BeautifulSoup(response, features='html.parser')
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    parsed_data = []

    for ticker, news_table in news_tables.items():

        for row in news_table.findAll('tr'):
            try:
                title = row.find('a', class_='tab-link-news').text
                date_data = row.find('td').text.replace("\r\n", "").split(' ')
                filtered_data = [item for item in date_data if item.strip() != '']
                if len(filtered_data) == 1:
                    time = filtered_data[0]
                else:
                    date = filtered_data[0]
                    time = filtered_data[1]

                # Use flair for sentiment analysis
                sentence = flair.data.Sentence(title)
                sentiment_model.predict(sentence)
                compound_score = sentence.labels[0].score
                compound_value = sentence.labels[0].value
                if compound_value == 'POSITIVE':
                  sentiment_positive_count = sentiment_positive_count+1
                  sentiment_positive_sum += compound_score
                else:
                  sentiment_negative_count= sentiment_negative_count+1
                  sentiment_negative_sum += compound_score

            except Exception as e:
                print(e)

            parsed_data.append([ticker, date, time, title, compound_value, compound_score])

    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title', 'compound_value', 'compound_score'])
    df['date'] = pd.to_datetime(df.date).dt.date
    # print(df.head())
    print('postive count: ')
    print(sentiment_positive_count)
    print('negative count: ')
    print(sentiment_negative_count)
    print('postive sum: ')
    print(sentiment_positive_sum)
    print('negative sum: ')
    print(sentiment_negative_sum)

    vader = SentimentIntensityAnalyzer()

    f = lambda title: vader.polarity_scores(title)['compound']
    df['valer'] = df['title'].apply(f)
    vader_sum = df['valer'].sum()
    print(df['valer'].sum())

    if sentiment_positive_count >= 55 and vader_sum >= 10:
        return True
    else:
        return False


sentiment_analyzer('AAPL')