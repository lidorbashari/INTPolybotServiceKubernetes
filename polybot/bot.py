import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import boto3
import json



class Bot:

    def __init__(self, token, telegram_chat_url, s3_bucket_name, sqs_url):
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)
        self.s3_bucket_name = s3_bucket_name
        self.sqs_url = sqs_url
        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def download_photo_from_s3(self, s3_path):
        try:
            path_parts = s3_path.replace("s3://", "").split("/", 1)
            bucket_name = path_parts[0]
            object_key = path_parts[1]
            s3_client = boto3.client('s3')
            local_dir = "/tmp/predictions"
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, os.path.basename(object_key))
            s3_client.download_file(bucket_name, object_key, local_path)
            return local_path

        except Exception as e:
            print(f"Error downloading file from S3: {e}")
            return None

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class ObjectDetectionBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            try:
                photo_path = self.download_user_photo(msg)
                photo_id = msg['photo'][-1]['file_id']
                chat_id = msg['chat']['id']

                s3_url = self.upload_photo_to_s3(photo_path)
                if s3_url:
                    self.send_event_to_sqs(self.sqs_url, photo_id, s3_url, chat_id)
                    self.send_text(chat_id, "Your image is being processed. Please wait...")
                else:
                    self.send_text(chat_id, "Failed to upload your image. Please try again later.")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.send_text(msg['chat']['id'], "There was an error processing your image.")
        else:
            self.send_text(msg['chat']['id'], "Please send a photo.")

            # TODO upload the photo to S3

    def upload_photo_to_s3(self, photo_path, object_name=None):
        if object_name is None:
            object_name = f"photos-k8s/{os.path.basename(photo_path)}"
        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file(photo_path, self.s3_bucket_name, object_name)
            url = f"https://{self.s3_bucket_name}.s3.eu-north-1.amazonaws.com/{object_name}"
            logger.info(f'Successfully uploaded {photo_path} to S3 bucket {self.s3_bucket_name} as {url}')
            return url
        except Exception as e:
            logger.error(f"Failed to upload {photo_path} to S3 bucket. Error: {e}")
            return None

            # TODO send a job to the SQS queue
    def send_event_to_sqs(self, queue_url, photo_id, photo_path, chat_id):
        sqs = boto3.client('sqs')
        message_body = json.dumps({
            "photo_id": photo_id,
            "file_path": photo_path,
            "chat_id": chat_id
        })

        try:
            response = sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=message_body
            )
            logger.info(f"Successfully sent message to SQS: {response['MessageId']}")
        except Exception as e:
            logger.error(f"Failed to send message to SQS: {e}")

        # TODO send message to the Telegram end-user (e.g. Your image is being processed. Please wait...)
