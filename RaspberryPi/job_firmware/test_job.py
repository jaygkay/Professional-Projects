import argparse, os, boto3, glob

# Read in command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument("-k", "--keyID", action="store", dest="accessKeyID", default='AKIAICFM4F6PFMZG4JKQ', help="Your AWS access key")
parser.add_argument("-s", "--secretkey", action="store", dest="accessSecretKey", default='MTf/tQtqKl6BFiC7XcVCPWqBEZqy/yhKYuD4mjMJ', help="Your AWS secret access key")
parser.add_argument("-b", "--bucket", action="store", default='raspberry22-backup', dest="bucketName", help="AWS bucket name")

args = parser.parse_args()
AWS_ACCESS = args.accessKeyID
AWS_SECRET = args.accessSecretKey
bucketName = args.bucketName

# config
#AWS_ACCESS = 'AKIAICFM4F6PFMZG4JKQ'
#AWS_SECRET = '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'
#bucketName = 'raspberry22-backup'

session = boto3.Session(aws_access_key_id = AWS_ACCESS ,aws_secret_access_key = AWS_SECRET)
client = session.client('s3')


''' 
with open("camera_RPi.txt", "wb") as file:
	for f in filenames:
		file.write(f)
'''
''' 
for f in filenames:
    client.upload_file(directory+f, bucketName, f)
    print('File name: %s, Bucket name: %s' %(f,bucketName))
'''
#from aws.iot import jobs
import json
import boto3
import botocore

BUCKET_NAME = 'your_s3_bucket_name'
KEY = 'your_file_name'

s3 = session.resource('s3')



try:
        s3.Bucket(BUCKET_NAME).download_file(KEY, 'rasp_job_s3.json')
        print("File downloaded")
except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
        else:
                raise


with open("rasp_job.json",'rb') as fw:
        fw_current = dic = json.load(fw)

print(fw_current)
