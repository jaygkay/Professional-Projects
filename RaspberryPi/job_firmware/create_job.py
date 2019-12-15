import ast, os,glob, boto3, botocore, time

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

def DeleteJob(JOBID):
    try:
        iot.delete_job(jobId=JOBID,force=True)
        print("Job has been deleted!")
        print("Job name: %s" %JOBID)
    except botocore.exceptions.ClientError as e:
              print("Cannot delete job!\nERROR Message:", e)
            

# IMG and job information
S3_REGION = 'your_s3_region'
IMG_BUCKET_NAME = 'your_s3_bucket_name'
IMG_KEY = 'your_file_name'

JOB_BUCKET_NAME = 'your_job_bucket_name'
JOB_DOC = 'your_job_file_name'

directory = os.popen('pwd').read().rstrip() + '/'+ 'raspJob/'

# open S3 session with AWS access
session = boto3.Session(aws_access_key_id = AWS_ACCESS ,aws_secret_access_key = AWS_SECRET)
s3 = session.resource('s3')

client = session.client('s3')
UploadFile(directory, IMG_KEY, IMG_BUCKET_NAME)

UploadFile(directory, JOB_DOC, JOB_BUCKET_NAME)

iot = session.client('iot', region_name = S3_REGION) 

# params
JOBID = 'your_job_ID'
TARGET = 'arn:aws:iot:your_s3_region:your_account_ID:thing/your_iot_topic'
