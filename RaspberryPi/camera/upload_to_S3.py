import argparse, os, boto3, glob

# Read in command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument("-k", "--keyID", action="store", dest="accessKeyID", default='AKIAICFM4F6PFMZG4JKQ', help="Your AWS access key")
parser.add_argument("-s", "--secretkey", action="store", dest="accessSecretKey", default='MTf/tQtqKl6BFiC7XcVCPWqBEZqy/yhKYuD4mjMJ', help="Your AWS secret access key")
parser.add_argument("-b", "--bucket", action="store", default='raspberry22-backup', dest="bucketName", help="AWS bucket name")
parser.add_argument("-f", "--filename", action="store", required=True, dest="backupFile", help="Trip_start part of you backup file")

args = parser.parse_args()
AWS_ACCESS = args.accessKeyID
AWS_SECRET = args.accessSecretKey
bucketName = args.bucketName
tripStart = args.backupFile

# config
#AWS_ACCESS = 
#AWS_SECRET = 
#bucketName = 

session = boto3.Session(aws_access_key_id = AWS_ACCESS ,aws_secret_access_key = AWS_SECRET)
client = session.client('s3')

directory = os.popen('pwd').read().rstrip() + '/Camera' + '/'
filenames = [os.path.basename(x) for x in glob.glob(str(directory) + '*{}.avi'.format(tripStart))]


''' #Get avi file list ended with tripStart 
with open("camera_RPi.txt", "wb") as file:
	for f in filenames:
		file.write(f)
'''

for f in filenames:
    client.upload_file(directory+f, bucketName, f)
    print('File name: %s, Bucket name: %s' %(f,bucketName))
