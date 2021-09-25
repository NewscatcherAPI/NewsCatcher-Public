"""Import packages"""
import os
import json
import logging


# Import Global Env Variables
Variable_1 = os.environ(['VARIABLE_1'])
Variable_2 = os.environ(['VARIABLE_2'])
project_id = os.environ(['PROJECT_ID'])
topic_id = os.environ['TOPIC_ID']
subscription_id = os.environ(['SUBSRIPTION_ID'])
timeout = os.environ(['TIMEOUT'])
max_messages_sent = os.environ(['MAX_SENT'])

"""Everything related to Google Pub/Sub"""
from google.cloud import pubsub_v1

# Subscriber which extract news article message from the pull
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)
flow_control = pubsub_v1.types.FlowControl(max_messages=max_messages_sent)

# Topic where the extracted article is sent to
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)



def consume_message(message_input):
    message = json.loads(message_input.data.decode('utf-8'))

    """Function continue and extract data from an article"""

    message_to_send = json.dumps(
        {'title': 'Toyota reveals a hybrid pickup at last—but it\'s probably not what you think',
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
        logging.error(e)

    message_input.ack()


if __name__ == "__main__":
    while True:
        streaming_pull_future = subscriber.subscribe(subscription_path, callback=consume_message, flow_control=flow_control)
        try:
            streaming_pull_future.result(timeout=timeout)
        except Exception as e:
            logging.error(e)
            streaming_pull_future.cancel()
            subscriber.close()
            subscriber = pubsub_v1.SubscriberClient()
            subscription_path = subscriber.subscription_path(project_id, subscription_id)
            flow_control = pubsub_v1.types.FlowControl(max_messages=max_messages_sent)