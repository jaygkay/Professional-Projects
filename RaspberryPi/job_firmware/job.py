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

args = parser.parse_args()
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

# Connect and subscribe to AWS IoT
myAWSIoTMQTTClient.connect()
if args.mode == 'both' or args.mode == 'subscribe':
    myAWSIoTMQTTClient.subscribe(topic, 1, customCallback)
time.sleep(2)

import ast, os, glob, boto3, botocore, time

def GetFileList(BUCKET_NAME):
    # get file list from bucket
    bucket = s3.Bucket(BUCKET_NAME)
    objs = [obj for obj in bucket.objects.all()]
    files = [obj.key for obj in objs]
    print("Bucket name: %s \nfile list: %s" %(BUCKET_NAME, files))
    return files

def UploadFile(directory, KEY, BUCKET_NAME):
    try: 
        if directory: os.chdir(directory)
        else: directory = os.popen('pwd').read().rstrip() + '/'

        filenames = [os.path.basename(x) for x in glob.glob(str(directory) + '*{}'.format(KEY))]
        for f in filenames:
            client.upload_file(directory+f, BUCKET_NAME, f)
            print("Uploading file successed!")
            print('File name: %s, Bucket name: %s' %(f,BUCKET_NAME))
    except botocore.exceptions.ClientError as e:
              print("Uploading file FAILED!\nERROR Message:", e)

def DownloadFile(directory, KEY, BUCKET_NAME, FILE_NAME):
    if directory: os.chdir(directory)
    else: directory = os.popen('pwd').read().rstrip() + '/'

    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY, FILE_NAME)
        print("File downloaded")
        print("File: %s" %(directory+FILE_NAME))
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


def CreateJob():
    global S3_REGION, JOB_BUCKET_NAME, JOBID, TARGET, JOB_DOC
    try:
        createJob_response = iot.create_job(
            jobId=JOBID,
            targets=[TARGET],
            documentSource='https://'+JOB_BUCKET_NAME+'.s3.'+S3_REGION+'.amazonaws.com/'+JOB_DOC,
            description=' ',
            presignedUrlConfig={},
            targetSelection='SNAPSHOT',
            jobExecutionsRolloutConfig={},
            documentParameters={})
        # Job document and job document source cannot be specified at the same time.

        # get the job and job document information
        info_job = iot.describe_job(jobId=JOBID) # job info
        info_job_doc = iot.get_job_document(jobId=JOBID)
        job_doc_dict = ast.literal_eval(info_job_doc['document']) # job doc info

        print("New Job is created!")
        print("===========================================================================")
        print("Job name: %s" %JOBID)
        print("Target: %s" %TARGET)
        print("Status: %s" %info_job['job']['status'])
        print("Document Sourse: \n%s" %info_job['documentSource'])
        print("Document:")
        print(job_doc_dict)
        print("Responses:")
        print(createJob_response)
        print()
        return info_job, job_doc_dict
              
    except botocore.exceptions.ClientError as e:
              print("Creating job FAILED!\nERROR Message:", e)
def GetJobInfo(JOBID):
	# get the job and job document information
        info_job = iot.describe_job(jobId=JOBID) # job info
        info_job_doc = iot.get_job_document(jobId=JOBID)
        job_doc_dict = ast.literal_eval(info_job_doc['document']) # job doc info

        print("New Job is created!")
        print("===========================================================================")
        print("Job name: %s" %JOBID)
        print("Target: %s" %TARGET)
        print("Status: %s" %info_job['job']['status'])
        print("Document Sourse: \n%s" %info_job['documentSource'])
        print("Document:")
        print(job_doc_dict)
        print("Responses:")
        print()
        return info_job, job_doc_dict


def VersionUpdate(directory, CURR_IMG_FILE):
    if directory: os.chdir(directory)
    else: directory = os.popen('pwd').read().rstrip() + '/'+ 'raspJob/'

    filenames = [os.path.basename(x) for x in glob.glob(str(directory) + '*{}'.format(CURR_IMG_FILE))]
    curr_version = filenames[0].split('.')[:]
    NEW_ver = int(info_job_doc['firmware']['version'])
    CURR_ver = int(curr_version[0][-2:])
    print("New Version: %d" %NEW_ver)
    print("Current Version: %d" %CURR_ver)
    print()
    
    if NEW_ver > CURR_ver : 
        print("New Version available!")
        DownloadFile(directory, IMG_KEY, IMG_BUCKET_NAME, 'rasp_img_{}.txt'.format(NEW_ver))
        print("Updating Version ...")
        print("=======================")
        while info_job['job']['status'] == 'IN_PROGRESS':
            print(info_job['job']['status']+'...')
            time.sleep(10)
        print()
        print("Updating Statue: %s" %info_job['job']['status'])
        print()
    else: print("Latest Version!")



# IMG and job information
S3_REGION = 'your_aws_region'
IMG_BUCKET_NAME = 'your_s3_bucket_name'
IMG_KEY = 'your_file_name'

JOB_BUCKET_NAME = 'your_job_bucket_name'
JOB_DOC = 'your_job_file_name'

directory = os.popen('pwd').read().rstrip() + '/'+ 'raspJob/'
# open S3 session with AWS access
session = boto3.Session(aws_access_key_id = AWS_ACCESS ,aws_secret_access_key = AWS_SECRET)
s3 = session.resource('s3')

client = session.client('s3')
#UploadFile(directory, IMG_KEY, IMG_BUCKET_NAME)

iot = session.client('iot', region_name = S3_REGION) # region name is from the endpoint of iot

# params
JOBID = 'your_job_ID'
TARGET = 'arn:aws:iot:your_s3_region:your_account_ID:thing/your_iot_topic'

#GetJobInfo(JOBID)
info_job, info_job_doc = GetJobInfo(JOBID)
NEW_ver = int(info_job_doc['firmware']['version'])

CURR_IMG_FILE = 'your_current_file_name'
print()
print()
print("current version: ", CURR_IMG_FILE)
print("Job doc version: ", NEW_ver)
VersionUpdate(directory, CURR_IMG_FILE)


''' 
# Publish to the same topic in a loop forever
loopCount = 0
while True:
    if args.mode == 'both' or args.mode == 'publish':
        message = {}
        message['message'] = args.message
        message['sequence'] = loopCount
        messageJson = json.dumps(message)
        myAWSIoTMQTTClient.publish(topic, messageJson, 1)
        if args.mode == 'publish':
            print('Published topic %s: %s\n' % (topic, messageJson))
        loopCount += 1
    time.sleep(1)
'''
