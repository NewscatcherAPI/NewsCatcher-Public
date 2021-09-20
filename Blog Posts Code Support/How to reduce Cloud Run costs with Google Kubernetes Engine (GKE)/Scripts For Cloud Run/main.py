"""import default packages"""
import os
import logging
import json
import base64


# Import Global Env Variables
Variable_1 = os.environ(['VARIABLE_1'])
Variable_2 = os.environ(['VARIABLE_2'])
project_id = os.environ(['PROJECT_ID'])
topic_id = os.environ(['TOPIC_ID'])

"""Import downloaded packages"""


from flask import Flask, request
from flask_cors import CORS
from flask_sslify import SSLify


"""Everything related to Google Pub/Sub"""
from google.cloud import pubsub_v1

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)


# Initialise flask app
app = Flask(__name__)
CORS(app, supports_credentials=True)
sslify = SSLify(app)


def transform_message(input_message):
    try:
        output_message = json.loads(base64.b64decode(input_message).decode('utf-8'))
        return output_message
    except:
        return None


@app.route("/", methods=["POST"])
def extract_article():
    logging.basicConfig(
        level='INFO',
        format='[%(levelname)-5s] %(asctime)s\t-\t%(message)s',
        datefmt='%d/%m/%Y %I:%M:%S %p'
    )

    envelope = request.get_json()
    if not envelope:
        msg = "no Pub/Sub message received"
        logging.info(f"ERROR: {msg}")
        return {'message': f"Bad Request: {msg}",
                'status': 'error'}, 400

    if not isinstance(envelope, dict) or "message" not in envelope:
        msg = "invalid Pub/Sub message format"
        logging.info(f"ERROR: {msg}")
        return {'message': {msg},
                'status': 'error'}, 400

    pubsub_message = envelope["message"]

    if isinstance(pubsub_message, dict) and "data" in pubsub_message:
        message = transform_message(pubsub_message["data"])
        if not message:
            logging.info(f'Impossible to consume the message {str(pubsub_message["data"])}')
            msg = "Impossible to transform the message from Pub/Sub"
            return {"message": f"Error: {msg}",
                    'status': 'error'}, 400

    """Function continues"""



    message_to_send = json.dumps({'title': 'Toyota reveals a hybrid pickup at last—but it\'s probably not what you think',
                                  'summary': '	After more than 20 years of hybrid sales in the U.S.,'
                                             ' it\'s hard to believe that Toyota hasn\'t made a single hybrid pickup.'
                                             ' The wait is at last over. But no, the hybrid pickup Toyota\'s delivering isn\'t'
                                             ' going to land anywhere close to the 40-mpg Ford Maverick hybrid pickup revealed'
                                             ' earlier this year.Toyota\'s jumping in the hybrid truck arena with the arrival'
                                             ' of a completely redesigned 2022 Toyota Tundra—its full-size truck'
                                             ' that has been one of the thirstiest models in its peer set'
                                             ' (14 mpg combined, for the 2021 4WD models).'}).encode('utf-8')

    try:
        publish_future = publisher.publish(topic_path, data=message_to_send)
        publish_future.result()
    except Exception as e:
        logging.info(f"ERROR: while sending a message")

if __name__ == "__main__":
    app.run(ssl_context="adhoc", host="0.0.0.0", port=5000)