import time
from pathlib import Path
from detect import run
import yaml
from loguru import logger
import os
import json
import boto3
from pymongo import MongoClient
import requests

# Environment variables
images_bucket = os.environ['BUCKET_NAME']
queue_name = os.environ['SQS_QUEUE_NAME']
local_download_path = '/photos/tmp'
os.makedirs(local_download_path, exist_ok=True)

if not images_bucket or not queue_name:
    raise ValueError("Missing required environment variables: BUCKET_NAME or SQS_QUEUE_NAME")

# AWS SQS client
sqs_client = boto3.client('sqs', region_name='eu-north-1')

# Load class names from YOLO dataset configuration
with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

# MongoDB client setup
client = MongoClient("mongodb://mongodb-0.mongodb:27017/")
db = client['predictions_db']
predictions_collection = db['predictions']

def save_prediction_to_mongo(prediction_summary):
    """Save the prediction results to MongoDB."""
    try:
        predictions_collection.insert_one(prediction_summary)
        logger.info(f"Prediction saved to MongoDB: {prediction_summary}")
    except Exception as e:
        logger.error(f"Failed to save prediction to MongoDB: {e}")


def download_photo_from_s3(bucket_name, object_key, local_path):
    """Download an image from S3 and save it locally."""
    s3_client = boto3.client('s3')
    try:
        local_file_path = os.path.join(local_path, os.path.basename(object_key))

        # Check if we have write permissions to the local path
        if not os.access(local_path, os.W_OK):
            logger.error(f"Write permission denied for local path: {local_path}")
            return None

        s3_object_key = object_key.replace(f'https://{bucket_name}.s3.eu-north-1.amazonaws.com/', '')
        logger.info(f"Attempting to download file: {bucket_name}, {s3_object_key}, {local_file_path}")
        s3_client.download_file(bucket_name, s3_object_key, local_file_path)
        logger.info(f"Downloaded {object_key} from bucket {bucket_name} to {local_file_path}")
        return local_file_path
    except Exception as e:
        logger.error(f"Failed to download {object_key} from bucket {bucket_name}: {e}")
        return None

def upload_to_s3(local_file_path, object_key):
    """Upload a processed image to S3."""
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(local_file_path, images_bucket, object_key)
        logger.info(f"Successfully uploaded {local_file_path} to S3 as {object_key}")
        return f's3://{images_bucket}/{object_key}'
    except Exception as e:
        logger.error(f"Failed to upload {local_file_path} to S3: {e}")
        return None

def consume():
    """Continuously process messages from the SQS queue."""
    while True:
        response = sqs_client.receive_message(QueueUrl=queue_name, MaxNumberOfMessages=1, WaitTimeSeconds=5)

        if 'Messages' in response:
            message = response['Messages'][0]
            message_body = json.loads(message['Body'])
            receipt_handle = message['ReceiptHandle']
            prediction_id = message['MessageId']

            logger.info(f'Prediction {prediction_id} - Start processing')

            # Extract required fields from the message
            img_name = message_body.get("photo_id")
            chat_id = message_body.get("chat_id")

            if not all([img_name, chat_id, message_body.get("file_path")]):
                logger.error(f"Missing required fields in SQS message: {message_body}")
                sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)
                continue

            s3_object_key = message_body.get("file_path").replace(f's3://{images_bucket}/', '')
            original_img_path = download_photo_from_s3(images_bucket, s3_object_key, local_download_path)

            if not original_img_path:
                continue

            logger.info(f'Prediction {prediction_id} - Download completed: {original_img_path}')

            # Run YOLO object detection
            try:
                run(
                    weights='yolov5s.pt',
                    data='data/coco128.yaml',
                    source=original_img_path,
                    project='static/data',
                    name=prediction_id,
                    save_txt=True
                )

                logger.info(f'Prediction {prediction_id} - YOLO processing completed')
            except Exception as e:
                logger.error(f"Failed to process image with YOLO: {e}")

            # Define paths for processed images and labels
            predicted_img_path = Path(f'static/data/{prediction_id}/{os.path.basename(original_img_path)}')

            if not predicted_img_path.exists():
                logger.error(f"Prediction {prediction_id} - Processed image not found: {predicted_img_path}")
                continue

            object_key = f"predicted/{prediction_id}/{os.path.basename(predicted_img_path)}"
            uploaded_predicted_img_path = upload_to_s3(predicted_img_path, object_key)

            # Parse prediction results
            pred_summary_path = Path(f'static/data/{prediction_id}/labels/{Path(original_img_path).stem}.txt')
            labels = []  # הגדרה ברירת מחדל ל-labels
            if pred_summary_path.exists():
                with open(pred_summary_path) as f:
                    labels = [line.split(' ') for line in f.read().splitlines()]
                    labels = [{
                        'class': names[int(l[0])],
                        'cx': float(l[1]),
                        'cy': float(l[2]),
                        'width': float(l[3]),
                        'height': float(l[4]),
                    } for l in labels]
                logger.info(f'Prediction {prediction_id} - Labels extracted: {labels}')

            # Save prediction summary to MongoDB with chat_id
            prediction_summary = {
                'prediction_id': prediction_id,
                'original_img_path': original_img_path,
                'predicted_img_path': uploaded_predicted_img_path,  # שימוש בנתיב S3
                'labels': labels,
                'time': time.time(),
                'chat_id': chat_id  # הוספת chat_id כדי שיהיה זמין ב-MongoDB
            }

            save_prediction_to_mongo(prediction_summary)

            # Send POST request to Polybot with prediction_id as query parameter
            polybot_endpoint = f"http://polybot-service/results?predictionId={prediction_id}"
            payload = {"predictionId": prediction_id}
            try:
                polybot_response = requests.post(polybot_endpoint, json=payload)
                if polybot_response.status_code == 200:
                    logger.info(f"Results fetched successfully from Polybot: {polybot_response.text}")
                else:
                    logger.error(f"Failed to fetch results from Polybot: {polybot_response.status_code} - Response: {polybot_response.text}")
            except Exception as e:
                logger.error(f"Error sending request to Polybot: {e}")

            # Delete processed message from SQS queue
            sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)

if __name__ == "__main__":
    consume()