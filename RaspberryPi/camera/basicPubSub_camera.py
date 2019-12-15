'''
/*
 * Copyright 2010-2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */
 '''

from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import logging
import time
import argparse
import json

AllowedActions = ['both', 'publish', 'subscribe']

# Custom MQTT message callback
def customCallback(client, userdata, message):
    print("Received a new message: ")
    print(message.payload)
    print("from topic: ")
    print(message.topic)
    print("--------------\n\n")


# Read in command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--endpoint", action="store", required=True, dest="host", help="Your AWS IoT custom endpoint")
parser.add_argument("-r", "--rootCA", action="store", required=True, dest="rootCAPath", help="Root CA file path")
parser.add_argument("-c", "--cert", action="store", dest="certificatePath", help="Certificate file path")
parser.add_argument("-k", "--key", action="store", dest="privateKeyPath", help="Private key file path")
parser.add_argument("-w", "--websocket", action="store_true", dest="useWebsocket", default=False,
                    help="Use MQTT over WebSocket")
parser.add_argument("-id", "--clientId", action="store", dest="clientId", default="basicPubSub",
                    help="Targeted client id")
parser.add_argument("-t", "--topic", action="store", dest="topic", default="sdk/test/Python", help="Targeted topic")
parser.add_argument("-m", "--mode", action="store", dest="mode", default="both",
                    help="Operation modes: %s"%str(AllowedActions))
parser.add_argument("-M", "--message", action="store", dest="message", default="Hello World!",
                    help="Message to publish")


# S3 parameters
import argparse, os, boto3, glob


parser.add_argument("-K", "--keyID", action="store", dest="accessKeyID", default='AKIAICFM4F6PFMZG4JKQ', help="Your AWS access key")
parser.add_argument("-S", "--secretkey", action="store", dest="accessSecretKey", default='MTf/tQtqKl6BFiC7XcVCPWqBEZqy/yhKYuD4mjMJ', help="Your AWS secret access key")
parser.add_argument("-B", "--bucket", action="store", default='raspberry22-backup', dest="bucketName", help="AWS bucket name")
#parser.add_argument("-f", "--filename", action="store", required=True, dest="backupFile", help="Trip_start part of you backup file")

args = parser.parse_args()
AWS_ACCESS = args.accessKeyID
AWS_SECRET = args.accessSecretKey
bucketName = args.bucketName

host = args.host
rootCAPath = args.rootCAPath
certificatePath = args.certificatePath
privateKeyPath = args.privateKeyPath
useWebsocket = args.useWebsocket
clientId = args.clientId
topic = args.topic

if args.mode not in AllowedActions:
    parser.error("Unknown --mode option %s. Must be one of %s" % (args.mode, str(AllowedActions)))
    exit(2)

if args.useWebsocket and args.certificatePath and args.privateKeyPath:
    parser.error("X.509 cert authentication and WebSocket are mutual exclusive. Please pick one.")
    exit(2)

if not args.useWebsocket and (not args.certificatePath or not args.privateKeyPath):
    parser.error("Missing credentials for authentication.")
    exit(2)

# Configure logging
logger = logging.getLogger("AWSIoTPythonSDK.core")
logger.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)

# Init AWSIoTMQTTClient
myAWSIoTMQTTClient = None
if useWebsocket:
    myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId, useWebsocket=True)
    myAWSIoTMQTTClient.configureEndpoint(host, 443)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath)
else:
    myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId)
    myAWSIoTMQTTClient.configureEndpoint(host, 8883)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath, privateKeyPath, certificatePath)

# AWSIoTMQTTClient connection configuration
myAWSIoTMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
myAWSIoTMQTTClient.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
myAWSIoTMQTTClient.configureDrainingFrequency(2)  # Draining: 2 Hz
myAWSIoTMQTTClient.configureConnectDisconnectTimeout(10)  # 10 sec
myAWSIoTMQTTClient.configureMQTTOperationTimeout(5)  # 5 sec


videotopic = 'sdk/test/videolists'

# Connect and subscribe to AWS IoT
myAWSIoTMQTTClient.connect()
if args.mode == 'both' or args.mode == 'subscribe':
    myAWSIoTMQTTClient.subscribe(topic, 1, customCallback)
time.sleep(2)

#videotopic == 'sdk/test/videolists'

#message = {}
#message['msg'] = " "
# Publish to the same topic in a loop forever
while True:
    if args.mode == 'both' or args.mode == 'publish':
#	print("Topic: %s" %args.topic)
#	if videotopic == 'sdk/test/videolists':
#		if args.mode == 'both' or args.mode == 'subscribe':
#			myAWSIoTMQTTClient.subscribe(videotopic, 1, customCallback)
	print(customCallback)
	message = {}
	message['msg'] = ' '
	print(message)
    	if message['msg'] == 'get list':

		session = boto3.Session(aws_access_key_id = AWS_ACCESS ,aws_secret_access_key = AWS_SECRET)
        	client = session.client('s3')

               	directory = os.popen('pwd').read().rstrip() + '/Camera' + '/'
               	filelists = [os.path.basename(x) for x in glob.glob(str(directory) + '*.avi')]
               	filename = "RPi_camera2.txt"

		file = open(directory + "RPi_camera2.txt", "wb")
		for f in filelists:
			file.write(f + ',')
		#file.close()

		print("Write file")
		print(directory+filename)
      		client.upload_file(directory+filename, bucketName, filename)
		print("Upload file")
                #print('Topic: %s, File name: %s, Bucket name: %s' %(args.topic, filename, bucketName))
		''' 
	        message = {}
	        message['msg'] = "Type 'get list': "
		if message['msg'] == 'get list': print("TTT")
		else: print("FFF")
	        #message['file'] = filelists
		#message['bucket'] = bucketName
		'''
	        messageJson = json.dumps(message)
	        myAWSIoTMQTTClient.publish(videotopic, messageJson, 1)
		#print(message)
	        if args.mode == 'publish':
			print('Published topic %s: %s\n' % (videotopic, messageJson))
