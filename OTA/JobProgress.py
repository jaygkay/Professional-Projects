

from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTThingJobsClient
from AWSIoTPythonSDK.core.jobs.thingJobManager import jobExecutionTopicType
from AWSIoTPythonSDK.core.jobs.thingJobManager import jobExecutionTopicReplyType
from AWSIoTPythonSDK.core.jobs.thingJobManager import jobExecutionStatus

import threading
import logging
import time
import datetime
import argparse
import json
from boto3.s3.transfer import S3Transfer

import ast, os,glob, boto3, botocore, time, sys

# time for 'ota_start'

from datetime import datetime
from pytz import timezone
ts = "%Y-%m-%d %H:%M:%S"
cts = "%Y%m%d_%H%M%S"

class JobsMessageProcessor(object):
    def __init__(self, awsIoTMQTTThingJobsClient, clientToken):
        #keep track of this to correlate request/responses
        self.clientToken = clientToken
        self.awsIoTMQTTThingJobsClient = awsIoTMQTTThingJobsClient
        self.done = False
        self.jobsStarted = 0
        self.jobsSucceeded = 0
        self.jobsRejected = 0
        self._setupCallbacks(self.awsIoTMQTTThingJobsClient)

    def _setupCallbacks(self, awsIoTMQTTThingJobsClient):
        self.awsIoTMQTTThingJobsClient.createJobSubscription(self.newJobReceived, jobExecutionTopicType.JOB_NOTIFY_NEXT_TOPIC)
        self.awsIoTMQTTThingJobsClient.createJobSubscription(self.startNextJobSuccessfullyInProgress, jobExecutionTopicType.JOB_START_NEXT_TOPIC, jobExecutionTopicReplyType.JOB_ACCEPTED_REPLY_TYPE)
        self.awsIoTMQTTThingJobsClient.createJobSubscription(self.startNextRejected, jobExecutionTopicType.JOB_START_NEXT_TOPIC, jobExecutionTopicReplyType.JOB_REJECTED_REPLY_TYPE)

        # '+' indicates a wildcard for jobId in the following subscriptions
        self.awsIoTMQTTThingJobsClient.createJobSubscription(self.updateJobSuccessful, jobExecutionTopicType.JOB_UPDATE_TOPIC, jobExecutionTopicReplyType.JOB_ACCEPTED_REPLY_TYPE, '+')
        self.awsIoTMQTTThingJobsClient.createJobSubscription(self.updateJobRejected, jobExecutionTopicType.JOB_UPDATE_TOPIC, jobExecutionTopicReplyType.JOB_REJECTED_REPLY_TYPE, '+')

    #call back on successful job updates
    def startNextJobSuccessfullyInProgress(self, client, userdata, message):
        payload = json.loads(message.payload.decode('utf-8'))
        if 'execution' in payload:
            self.jobsStarted += 1
            execution = payload['execution']
            self.executeJob(execution)
            statusDetails = {'HandledBy': 'ClientToken: {}'.format(self.clientToken)}
            threading.Thread(target = self.awsIoTMQTTThingJobsClient.sendJobsUpdate, kwargs = {'jobId': execution['jobId'], 'status': jobExecutionStatus.JOB_EXECUTION_SUCCEEDED, 'statusDetails': statusDetails, 'expectedVersion': execution['versionNumber'], 'executionNumber': execution['executionNumber']}).start()
        else:
            print('Start next saw no execution: ' + message.payload.decode('utf-8'))
            self.done = True

    def executeJob(self, execution):
        print('Executing job ID, version, number: {}, {}, {}'.format(execution['jobId'], execution['versionNumber'], execution['executionNumber']))
        print('With jobDocument: ' + json.dumps(execution['jobDocument']))
        print('New version: {}'.format(execution['jobDocument']['firmware']['version']))

    def newJobReceived(self, client, userdata, message):
        payload = json.loads(message.payload.decode('utf-8'))
        if 'execution' in payload:
            self._attemptStartNextJob()
        else:
            print('Notify next saw no execution')
            self.done = True

    def processJobs(self):
        self.done = False
        self._attemptStartNextJob()

    def startNextRejected(self, client, userdata, message):
        printf('Start next rejected:' + message.payload.decode('utf-8'))
        self.jobsRejected += 1

    def updateJobSuccessful(self, client, userdata, message):
        self.jobsSucceeded += 1

    def updateJobRejected(self, client, userdata, message):
        self.jobsRejected += 1

    def _attemptStartNextJob(self):
        statusDetails = {'StartedBy': 'ClientToken: {} on {}'.format(self.clientToken, datetime.now().isoformat())}
        threading.Thread(target=self.awsIoTMQTTThingJobsClient.sendJobsStartNext, kwargs = {'statusDetails': statusDetails}).start()

    def isDone(self):
        return self.done

    def getStats(self):
        stats = {}
        stats['jobsStarted'] = self.jobsStarted
        stats['jobsSucceeded'] = self.jobsSucceeded
        stats['jobsRejected'] = self.jobsRejected
        return stats

    def completedjob(self):
        self.done = True

def FileSize(directory, filename):
    os.chdir(directory)
    statinfo = os.stat(filename)
    return statinfo.st_size

def TimeStamp(timeZone = 'America/Chicago'):
        # return local time (default = Chicago)
        tzone = datetime.now(timezone(timeZone))
        return tzone

def GetJobInfo(JOBID):
    # get the job and job document information
    info_job = iot.describe_job(jobId=JOBID) # job info
    info_job_doc = iot.get_job_document(jobId=JOBID)
    job_doc_dict = ast.literal_eval(info_job_doc['document']) # job doc info

    print("\n\n==============================[Job Info]===================================")
    print("Job name: %s" %JOBID)
    print("Status: %s" %info_job['job']['status'])
    print("Firmware version: %s" %job_doc_dict['firmware']['url'].split('/')[-2])
    print("Firmware file name: %s" %job_doc_dict['firmware']['url'].split('/')[-1])
    print("Size: %s" %job_doc_dict['firmware']['size'])
    print("Document Sourse: \n%s" %info_job['documentSource'])
    print("Document:")
    print(job_doc_dict)
    return info_job, job_doc_dict

class OTAProgressBar(object):
    def __init__(self, bucket, key, filename):
        self.bucket = bucket
        self.key = key
        self._filename = filename
        self._size = client.head_object(Bucket=bucket, Key=key)['ContentLength'] # size of total firmware file
        self._downloaded_size = 0 # size of downloaded chuck
        self._lock = threading.Lock() 
        self.percentage = 0.00

    def __call__(self, bytes_amount):        
        with self._lock:
            if args.mode == 'both' or args.mode == 'publish':
                self._downloaded_size += bytes_amount
                diff = TimeStamp() - ota_start
                hours = int(diff.seconds // (60 * 60))
                mins = int((diff.seconds // 60) % 60)
                sec = diff.seconds - mins*60

                message = {}
                message['camera_id'] = thingName
                message['OTA_start_time'] = ota_start.strftime(ts)
                message['Download time'] = '%s hour %s min %s second' %(hours, mins, sec)
                message['Firmware version'] = FIRMWARE_VERSION
                message['Firmware filename'] = self._filename.split('/')[-1]
                message['Progress'] = str(round((float(self._downloaded_size) / float(self._size)) * 100, 2))+' %'
                message['Total size'] = self._size
                message['Downloading'] = self._downloaded_size

                messageJson = json.dumps(message)
                myAWSIoTMQTTClient.publish(topic, messageJson, 1)
                if args.mode == 'publish':
                    print('Published topic %s: %s\n' % (topic, messageJson))
            
AllowedActions = ['both', 'publish', 'subscribe']

# subscribe topics
def listener():
    if args.mode == 'both' or args.mode == 'subscribe':
        myAWSIoTMQTTClient.subscribe("sdk/test/firmare", 1, videoCallback)


# Read in command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--thingName", action="store", dest="thingName", help="Your AWS IoT ThingName to process jobs for")
parser.add_argument("-e", "--endpoint", action="store", required=True, dest="host", help="Your AWS IoT custom endpoint")
parser.add_argument("-r", "--rootCA", action="store", required=True, dest="rootCAPath", help="Root CA file path")
parser.add_argument("-c", "--cert", action="store", dest="certificatePath", help="Certificate file path")
parser.add_argument("-k", "--key", action="store", dest="privateKeyPath", help="Private key file path")
parser.add_argument("-p", "--port", action="store", dest="port", type=int, help="Port number override")
parser.add_argument("-w", "--websocket", action="store_true", dest="useWebsocket", default=False,
                    help="Use MQTT over WebSocket")
parser.add_argument("-id", "--clientId", action="store", dest="clientId", default="basicJobsSampleClient",
                    help="Targeted client id")

parser.add_argument("-t", "--topic", action="store", dest="topic", default="CarVi/test/firmware", help="Targeted topic")
parser.add_argument("-m", "--mode", action="store", dest="mode", default="both",
                    help="Operation modes: %s"%str(AllowedActions))
parser.add_argument("-M", "--message", action="store", dest="message", default="Hello World!",
                    help="Message to publish")


args = parser.parse_args()
host = args.host
rootCAPath = args.rootCAPath
certificatePath = args.certificatePath
privateKeyPath = args.privateKeyPath
port = args.port
useWebsocket = args.useWebsocket
clientId = args.clientId
thingName = args.thingName
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

# Port defaults
if args.useWebsocket and not args.port:  # When no port override for WebSocket, default to 443
    port = 443
if not args.useWebsocket and not args.port:  # When no port override for non-WebSocket, default to 8883
    port = 8883

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
    myAWSIoTMQTTClient.configureEndpoint(host, port)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath)
else:
    myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId)
    myAWSIoTMQTTClient.configureEndpoint(host, port)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath, privateKeyPath, certificatePath)

# AWSIoTMQTTClient connection configuration
myAWSIoTMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
myAWSIoTMQTTClient.configureConnectDisconnectTimeout(100)  # 10 sec
myAWSIoTMQTTClient.configureMQTTOperationTimeout(100)  # 5 sec

myAWSIoTMQTTClient.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
myAWSIoTMQTTClient.configureDrainingFrequency(2)  # Draining: 2 Hz

jobsClient = AWSIoTMQTTThingJobsClient(clientId, thingName, QoS=1, awsIoTMQTTClient=myAWSIoTMQTTClient)


# Custom MQTT message callback
# topic : CarVi/test/firmware/[IMEI]
topic = topic+'/'+thingName

def firmwareCallback(client, userdata, message):
    print("Received a new message: ")
    print(message.payload)
    print("from topic: ")
    print(message.topic)
    print("--------------\n\n")

myAWSIoTMQTTClient.connect()
if args.mode == 'both' or args.mode == 'subscribe':
    myAWSIoTMQTTClient.subscribe(topic, 1, firmwareCallback)
time.sleep(2)

print('Connecting to MQTT server and setting up callbacks...')
jobsClient.connect()

# AWS access
AWS_ACCESS = 'AKIAJIOF7V5POZV3XZFA'
AWS_SECRET = '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'

# firmware and job information
S3_REGION = 'us-west-2' #'ap-northeast-2'
directory = '/Users/dajeongjeon/Desktop'

# open S3 session with AWS access
session = boto3.Session(aws_access_key_id = AWS_ACCESS ,aws_secret_access_key = AWS_SECRET)
s3 = session.resource('s3')
client = session.client('s3')
iot = session.client('iot', region_name = S3_REGION) # region name is from the endpoint of iot


# params
JOBID = 'rasp-ota-test'# 'Newfirmware-OTA'

info_job, info_job_doc = GetJobInfo(JOBID)
firmware_total_size = int(info_job_doc['firmware']['size'])
firmware_url = info_job_doc['firmware']['url']

# get bucket, key, and firmware version
BUCKET_NAME, KEY= firmware_url.split('/',3)[-1].split('/',1)
FIRMWARE_VERSION = firmware_url.split('/')[-2]
FILE_NAME = KEY.split('/')[-1]
directory = '/Users/dajeongjeon/Desktop/'


print('\n\n===========================[Required Info for OTA]===========================')
print('FROM------>>')
print('Bucket: %s' %BUCKET_NAME)
print('Key: %s' %KEY)
print('File: %s' %FILE_NAME)
print('\n')
print('TO------>>')
print('Local Directory: %s' %directory)
print('=============================================================================\n\n')

# version comparesion
NEW_ver = int(info_job_doc['firmware']['version'])
CURR_ver = 41811019 # int(curr_version[0][-2:])


print("\n\n============================[Version Information]============================")
print("New Version: %d" %NEW_ver)
print("Current Version: %d" %CURR_ver)
print("=============================================================================\n\n")


jobsMsgProc = JobsMessageProcessor(jobsClient, clientId)
print('\n\nStarting to process jobs................\n\n')


if info_job['job']['status'] != 'IN_PROGRESS':
    print("Job is NOT 'IN PROGRESS'")
    print("Current Status: {}".format(info_job['job']['status']))
    print("\n")

else:
    # Process OTA
    if NEW_ver > CURR_ver : 
        print("New Version available!")
        print('DOWNLOADING FIRMWARE...')

        ota_start = TimeStamp()
        with open(directory+FILE_NAME, 'wb') as data:
            client.download_fileobj(BUCKET_NAME, KEY, data, Callback=OTAProgressBar(BUCKET_NAME, KEY, directory+FILE_NAME))
        jobsMsgProc.processJobs()

    elif NEW_ver == CURR_ver : 
        print("Latest Version!")

jobsMsgProc = JobsMessageProcessor(jobsClient, clientId)
print('\n\n*****************************************************************')
print('Done processing jobs')
print('Stats: ' + json.dumps(jobsMsgProc.getStats()))
print('*****************************************************************\n\n')
jobsClient.disconnect()



