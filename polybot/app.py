import boto3
import flask
from flask import request
import os
import json
from bot import ObjectDetectionBot
from botocore.exceptions import ClientError
from pymongo import MongoClient
from loguru import logger

app = flask.Flask(__name__)

client = MongoClient("mongodb://mongodb-0.mongodb:27017/")
db = client["predictions_db"]
predictions_collection = db["predictions"]

# pull telegram token from aws secret manager:
def get_secret():
    secret_name = "lidor-telegram-bot-token"
    region_name = "eu-north-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        print(f"Error retrieving secret: {e}")
        return None
    if 'SecretString' in get_secret_value_response:
        secret = get_secret_value_response['SecretString']
        secret_dict = json.loads(secret)
        return secret_dict.get("TELEGRAM_BOT_TOKEN")


TELEGRAM_TOKEN = get_secret()
S3_BUCKET = 'lidorbashari'
TELEGRAM_APP_URL = os.environ['TELEGRAM_APP_URL']
SQS_URL = os.environ['SQS_URL']


@app.route('/', methods=['GET'])
def index():
    return 'Ok'


@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'


@app.route(f'/results', methods=['POST'])
def results():
    data = request.json
    prediction_id = data.get("predictionId")

    if not prediction_id:
        return {"success": False, "message": "Missing predictionId"}, 400

    prediction = predictions_collection.find_one({"prediction_id": prediction_id}, {"_id": 0})

    if not prediction:
        return {"success": False, "message": "Prediction not found"}, 404

    chat_id = prediction.get("chat_id")
    detected_objects = prediction.get("labels", [])
    predicted_img_path = prediction.get("predicted_img_path")

    text_results = f"🔍 Prediction Results for ID: \n"
    if detected_objects:
        for obj in detected_objects:
            text_results += f"- {obj['class']}\n"
    else:
        text_results = "No objects detected in the image."
    # TODO use the prediction_id to retrieve results from MongoDB and send to the end-user

    bot.send_text(chat_id, text_results)

    if predicted_img_path:
        try:
            local_image_path = bot.download_photo_from_s3(predicted_img_path)
            if local_image_path:
                logger.info(f"Sending photo from local path: {local_image_path}")
                bot.send_photo(chat_id, local_image_path)  # שליחת קובץ מתוך הקונטיינר
            else:
                bot.send_text(chat_id, "Sorry, there was an issue retrieving the image.")
        except Exception as e:
            logger.error(f"Failed to send photo: {e}")
            bot.send_text(chat_id, f"Sorry, there was an issue sending the processed image. Error: {str(e)}")
    else:
        bot.send_text(chat_id, "Sorry, I couldn't find the processed image.")

    return 'Ok'


@app.route(f'/loadTest/', methods=['POST'])
def load_test():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'


if __name__ == "__main__":
    bot = ObjectDetectionBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET, SQS_URL)

    app.run(host='0.0.0.0', port=8443)
