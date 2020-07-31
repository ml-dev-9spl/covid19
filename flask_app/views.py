import logging
import requests
import json
from flask import Blueprint, render_template

from flask_app import cache


main_bp = Blueprint('main', __name__)

logger = logging.getLogger('flask.app')


@main_bp.route('/', methods=["GET"])
def index():

    payload = {'country': 'India'}  # or {'code': 'DE'}
    URL = 'https://api.statworx.com/covid'
    response = requests.post(url=URL, data=json.dumps(payload))
    json_response = json.loads(response.text)
    date = json_response['date']
    cases = json_response['cases']
    deaths = json_response['deaths']
    cases_cum = json_response['cases_cum']
    deaths_cum = json_response['deaths_cum']
    # Convert to data frame
    # df = pd.DataFrame.from_dict(json.loads(response.text))
    covid_data = {
        "date":date,
        "cases":cases,
        "deaths":deaths,
        # "cases_cum":cases_cum,
        "deaths_cum":deaths_cum,
    }
    data = []
    for item in zip(date, cases):
        data.append({ 'date': item[0], "value": item[1]})
    logger.info("User requested to infere the xray")

    return render_template('index.html', covid_data=covid_data,data=data)